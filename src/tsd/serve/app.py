"""HTTP inference service.

* ``POST /v1/detect`` (multipart field ``image``) -> detections with class names and scores;
* ``GET /healthz`` -> 200 once the model is loaded and warmed up (with the batching and admission
  limits and the runtime's effective settings), else 503;
* ``GET /metrics`` -> Prometheus metrics.

The model is loaded once at start-up and warmed up (every batch size up to the largest) before the
server accepts connections. A pure ASGI middleware (:class:`RequestGate`) sits in front of
everything: it checks a detection request's ``Content-Length`` and admits it (at most
``max_inflight`` requests being received, decoded, queued or inferred at once) *before* any of
the upload is read, so an overload is answered with 429 instead of piling up decode work, and it
counts and times every response, validation errors included. Image dimensions are read from the
file header before decoding (decompression bombs are refused), decoding and letterboxing run on
a small thread pool, inference runs on the micro-batcher's own thread, so the event loop only
shuffles bytes, and a request whose client has disconnected is dropped from the queue (499).
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image as PILImage
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tsd.infer import Detector, LetterboxInfo, NMSConfig, load_detector
from tsd.io import Image, ImageDecodeError, decode_image
from tsd.serve.batcher import MicroBatcher, QueueFullError
from tsd.serve.metrics import ServeMetrics
from tsd.serve.settings import ServeSettings

Prepared = tuple[Image, LetterboxInfo]
_MULTIPART_OVERHEAD = 64 * 1024
MIN_IMAGE_SIDE = 8  # px; thinner images cannot show a sign and are refused with 400
ENDPOINTS = ("/v1/detect", "/healthz", "/metrics")
CLIENT_CLOSED = 499  # nginx's code for "client closed the request"


class ImageTooLargeError(ValueError):
    pass


class ImageTooSmallError(ValueError):
    pass


class Detection(BaseModel):
    class_id: int
    class_name: str
    score: float
    box_xyxy: list[float]


class DetectResponse(BaseModel):
    model: str
    backend: str
    width: int  # of the decoded image (EXIF orientation applied), the frame of the boxes
    height: int
    detections: list[Detection]
    batch_size: int
    queue_wait_ms: float
    inference_ms: float
    # Handler time: from the start of the endpoint (upload already received and parsed) to the
    # response body; the full request time is the tsd_http_request_duration_seconds histogram.
    total_ms: float


def image_size(data: bytes, max_pixels: int) -> tuple[int, int]:
    """Width and height from the file header, without decoding the pixels."""
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            width, height = im.size
    except PILImage.DecompressionBombError as err:
        raise ImageTooLargeError(str(err)) from err
    except Exception as err:  # Pillow raises several types for unreadable input
        raise ImageDecodeError("not a supported image file") from err
    if width * height > max_pixels:
        raise ImageTooLargeError(f"{width}x{height} exceeds the limit of {max_pixels} pixels")
    return width, height


class Admission:
    """Non-blocking count of admitted, unfinished detection requests (event-loop only)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def try_acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active -= 1


class RequestGate:
    """Pure ASGI middleware: upload checks and admission control for ``POST /v1/detect`` before
    the body is read (411, 413, 429 with ``Retry-After``), and a request counter and duration
    histogram by endpoint and status for every HTTP response (from the moment the request
    reaches the application to the end of the response)."""

    def __init__(
        self, app: ASGIApp, cfg: ServeSettings, metrics: ServeMetrics, admission: Admission
    ) -> None:
        self.app = app
        self.cfg = cfg
        self.metrics = metrics
        self.admission = admission

    def _check_upload(self, scope: Scope) -> Response | None:
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length", b"").decode("latin-1")
        if not length.isdigit():
            return JSONResponse({"detail": "a Content-Length header is required"}, 411)
        if int(length) > self.cfg.max_upload_bytes + _MULTIPART_OVERHEAD:
            return JSONResponse(
                {"detail": f"upload larger than {self.cfg.max_upload_bytes} bytes"}, 413
            )
        if not self.admission.try_acquire():
            self.metrics.rejected.labels("admission").inc()
            return JSONResponse(
                {"detail": "too many requests in flight"}, 429, headers={"Retry-After": "1"}
            )
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        started = time.perf_counter()
        status = 500

        async def send_and_record(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        admitted = False
        try:
            if scope["method"] == "POST" and path == "/v1/detect":
                rejection = self._check_upload(scope)
                if rejection is not None:
                    await rejection(scope, receive, send_and_record)
                    return
                admitted = True
            await self.app(scope, receive, send_and_record)
        finally:
            if admitted:
                self.admission.release()
            endpoint = path if path in ENDPOINTS else "other"
            self.metrics.requests.labels(endpoint, str(status)).inc()
            self.metrics.http_duration.labels(endpoint, str(status)).observe(
                time.perf_counter() - started
            )


@dataclass
class ServiceState:
    ready: bool = False
    detector: Detector | None = None
    batcher: MicroBatcher[Prepared, Any] | None = None
    pool: ThreadPoolExecutor | None = None


async def _wait_for_disconnect(request: Request) -> None:
    """Returns when the client goes away (the body has been read, so the next ASGI message is
    ``http.disconnect``, sent on disconnect or after the response)."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def create_app(settings: ServeSettings | None = None, detector: Detector | None = None) -> FastAPI:
    """Build the service; ``detector`` overrides loading ``settings.model_path`` (tests)."""
    cfg = settings or ServeSettings()
    metrics = ServeMetrics()
    nms = NMSConfig(
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        max_det=cfg.max_detections,
    )
    state = ServiceState()
    limit = cfg.max_inflight or cfg.queue_size + cfg.max_batch_size + cfg.preprocess_threads
    admission = Admission(limit)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if detector is not None:
            det = detector
        else:
            det = await asyncio.to_thread(
                load_detector, cfg.model_path, cfg.threads, nms, cfg.openvino_precision
            )
        # Every batch size the batcher can form: the first call at a new size is much slower
        # (the runtime prepares that shape), which showed up as the p99 at concurrency 8 when
        # only sizes 1 and max were warmed (bench/results/serving/batch_shape_probe.json).
        dummy = np.full((480, 640, 3), 114, np.uint8)
        for batch in range(1, cfg.max_batch_size + 1):
            for _ in range(cfg.warmup_iterations):
                await asyncio.to_thread(det.predict, [dummy] * batch, nms)
        batcher: MicroBatcher[Prepared, Any] = MicroBatcher(
            lambda items: det.infer_prepared(items, nms),
            max_batch_size=cfg.max_batch_size,
            max_wait_ms=cfg.max_wait_ms,
            queue_size=cfg.queue_size,
            metrics=metrics,
            adaptive=cfg.batch_wait == "adaptive",
        )
        batcher.start()
        pool = ThreadPoolExecutor(cfg.preprocess_threads, thread_name_prefix="tsd-pre")
        metrics.model_info.labels(model=cfg.model_path.name, backend=det.name).set(1)
        state.detector, state.batcher, state.pool, state.ready = det, batcher, pool, True
        try:
            yield
        finally:
            state.ready = False
            await batcher.stop()
            await asyncio.to_thread(pool.shutdown, True)

    app = FastAPI(title="Traffic-sign detection service", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg
    app.state.metrics = metrics
    app.state.admission = admission
    app.add_middleware(RequestGate, cfg=cfg, metrics=metrics, admission=admission)

    def prepare(det: Detector, data: bytes) -> tuple[tuple[int, int], Prepared]:
        image_size(data, cfg.max_image_pixels)  # refuse decompression bombs before decoding
        image = decode_image(data)
        height, width = image.shape[:2]  # the decoded frame (EXIF orientation applied)
        if min(width, height) < MIN_IMAGE_SIDE:
            raise ImageTooSmallError(f"{width}x{height} is smaller than {MIN_IMAGE_SIDE} px")
        return (width, height), det.prepare(image)

    @app.post("/v1/detect", response_model=DetectResponse)
    async def detect(
        request: Request,
        image: Annotated[UploadFile, File(description="An image file (JPEG, PNG, ...).")],
    ) -> Response:
        t0 = time.perf_counter()
        if not state.ready or state.detector is None or state.batcher is None:
            raise HTTPException(503, "model is not loaded")
        det = state.detector
        data = await image.read()
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(413, f"image larger than {cfg.max_upload_bytes} bytes")
        loop = asyncio.get_running_loop()
        try:
            (width, height), prepared = await loop.run_in_executor(state.pool, prepare, det, data)
        except ImageTooLargeError as err:
            raise HTTPException(413, str(err)) from err
        except (ImageDecodeError, ImageTooSmallError) as err:
            raise HTTPException(400, str(err)) from err
        submit = asyncio.ensure_future(state.batcher.submit(prepared))
        gone = asyncio.ensure_future(_wait_for_disconnect(request))
        try:
            await asyncio.wait({submit, gone}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            gone.cancel()
        if not submit.done():
            submit.cancel()  # the batcher skips a cancelled job
            metrics.disconnects.inc()
            return JSONResponse({"detail": "client disconnected"}, CLIENT_CLOSED)
        try:
            dets, info = submit.result()
        except QueueFullError as err:
            raise HTTPException(429, str(err), headers={"Retry-After": "1"}) from err
        handler_ms = (time.perf_counter() - t0) * 1e3
        metrics.handler_latency.observe(handler_ms / 1e3)
        body = DetectResponse(
            model=cfg.model_path.name,
            backend=det.name,
            width=width,
            height=height,
            detections=[Detection(**d) for d in dets.to_dicts(det.names)],
            batch_size=info.batch_size,
            queue_wait_ms=round(info.queue_wait_ms, 3),
            inference_ms=round(info.inference_ms, 3),
            total_ms=round(handler_ms, 3),
        )
        return JSONResponse(body.model_dump())

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        if not state.ready or state.detector is None or state.batcher is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return JSONResponse(
            {
                "status": "ok",
                "model": cfg.model_path.name,
                "backend": state.detector.name,
                "max_batch_size": cfg.max_batch_size,
                "max_wait_ms": cfg.max_wait_ms,
                "batch_wait": cfg.batch_wait,
                "max_inflight": admission.limit,
                "queue_depth": state.batcher.depth,
                # effective runtime settings (threads, precision hint, pinning), as queried
                "runtime": state.detector.runtime_settings(),
            }
        )

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
