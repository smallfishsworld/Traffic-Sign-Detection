"""Dynamic micro-batching on an asyncio queue.

Requests put one prepared (letterboxed) image each into a bounded queue. A single worker task
takes the first waiting image, collects more until ``max_batch_size`` images or until
``max_wait_ms`` after that first image *arrived* (so an image that already waited during the
previous batch is not held back further), and runs the batch on a dedicated inference thread, so
the event loop never blocks on inference. With ``adaptive`` waiting the worker does not wait at
all when nothing else is queued: a lone request (concurrency 1) pays no batching delay, and
under load the queue fills while the previous batch runs anyway. A full queue rejects new work
immediately (the HTTP layer turns that into 429) instead of letting latency grow without bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tsd.serve.metrics import ServeMetrics


class QueueFullError(RuntimeError):
    """The batching queue is full."""


@dataclass(frozen=True)
class BatchInfo:
    """What a request learns about how it was served."""

    batch_size: int
    queue_wait_ms: float
    inference_ms: float


@dataclass
class _Job[In, Out]:
    item: In
    future: asyncio.Future[tuple[Out, BatchInfo]]
    enqueued: float  # perf_counter() at submit, for the reported queue wait
    arrived: float  # event-loop time at submit, for the batching deadline


class MicroBatcher[In, Out]:
    def __init__(
        self,
        infer: Callable[[Sequence[In]], Sequence[Out]],
        max_batch_size: int = 8,
        max_wait_ms: float = 5.0,
        queue_size: int = 64,
        metrics: ServeMetrics | None = None,
        adaptive: bool = False,
    ) -> None:
        if max_batch_size < 1 or queue_size < 1 or max_wait_ms < 0:
            raise ValueError("invalid batching parameters")
        self._infer = infer
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1e3
        self._queue: asyncio.Queue[_Job[In, Out]] = asyncio.Queue(maxsize=queue_size)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tsd-infer")
        self._task: asyncio.Task[None] | None = None
        self._inflight: list[_Job[In, Out]] = []  # taken from the queue, not yet answered
        self._metrics = metrics
        self.adaptive = adaptive

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run(), name="tsd-batcher")

    async def stop(self) -> None:
        """Stop the worker and fail every request that has not been answered: those still in
        the queue and those of the batch being collected or run when the worker was cancelled."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        pending = list(self._inflight)
        while not self._queue.empty():
            pending.append(self._queue.get_nowait())
        for job in pending:
            if not job.future.done():
                job.future.set_exception(RuntimeError("service is shutting down"))
        self._inflight = []
        await asyncio.to_thread(self._executor.shutdown, True)

    async def submit(self, item: In) -> tuple[Out, BatchInfo]:
        """Queue ``item`` and wait for its result; raises :class:`QueueFullError` at once if the
        queue is full."""
        if self._task is None:
            raise RuntimeError("batcher is not running")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[Out, BatchInfo]] = loop.create_future()
        try:
            self._queue.put_nowait(_Job(item, future, time.perf_counter(), loop.time()))
        except asyncio.QueueFull:
            if self._metrics is not None:
                self._metrics.rejected.labels("queue_full").inc()
            raise QueueFullError("inference queue is full") from None
        self._set_depth()
        return await future

    def _set_depth(self) -> None:
        if self._metrics is not None:
            self._metrics.queue_depth.set(self._queue.qsize())

    async def _collect(self) -> list[_Job[In, Out]]:
        batch = self._inflight = []
        batch.append(await self._queue.get())
        if self.adaptive and self._queue.empty():
            return batch  # nobody else is waiting: run now
        loop = asyncio.get_running_loop()
        deadline = batch[0].arrived + self.max_wait_s
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    batch.append(await self._queue.get())
            except TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            jobs = [j for j in await self._collect() if not j.future.done()]  # drop cancelled
            self._set_depth()
            if not jobs:
                continue
            started = time.perf_counter()
            try:
                outputs = await loop.run_in_executor(
                    self._executor, self._infer, [j.item for j in jobs]
                )
                if len(outputs) != len(jobs):
                    raise RuntimeError("inference returned a wrong number of results")
            except Exception as err:  # deliver the failure to every waiting request
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(err)
                continue
            inference_ms = (time.perf_counter() - started) * 1e3
            if self._metrics is not None:
                self._metrics.batch_size.observe(len(jobs))
                self._metrics.inference_latency.observe(inference_ms / 1e3)
            for job, out in zip(jobs, outputs, strict=True):
                wait_ms = (started - job.enqueued) * 1e3
                if self._metrics is not None:
                    self._metrics.queue_wait.observe(wait_ms / 1e3)
                if not job.future.done():
                    job.future.set_result((out, BatchInfo(len(jobs), wait_ms, inference_ms)))
