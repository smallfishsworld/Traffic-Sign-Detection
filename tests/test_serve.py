"""HTTP service and micro-batcher: endpoints, limits, back-pressure, metrics."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="install the 'serve' extra")

from fastapi.testclient import TestClient

from tsd.infer import Detector
from tsd.serve import ServeSettings, create_app
from tsd.serve.batcher import MicroBatcher, QueueFullError


class FakeBackend:
    """One confident 'class 1' box centred in the input; optional per-call delay."""

    name = "fake"

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.batch_sizes: list[int] = []

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(batch.shape[0])
        time.sleep(self.delay_s)
        raw = np.zeros((batch.shape[0], 4 + 3, 10), np.float32)
        raw[:, :4, 0] = [320, 320, 64, 32]
        raw[:, 4 + 1, 0] = 0.9
        return raw


def _jpeg(h: int = 480, w: int = 640) -> bytes:
    ok, enc = cv2.imencode(".jpg", np.full((h, w, 3), 90, np.uint8))
    assert ok
    return enc.tobytes()


def _client(backend: FakeBackend, **settings: object) -> TestClient:
    options: dict[str, object] = {"model_path": Path("fake.xml"), "warmup_iterations": 1}
    options.update(settings)
    cfg = ServeSettings(**options)  # type: ignore[arg-type]
    return TestClient(create_app(cfg, detector=Detector(backend, ["a", "b", "c"])))


def test_detect_health_and_metrics() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["backend"] == "fake"
        assert health.json()["max_inflight"] == 64 + 8 + 2  # queue + batch + decode threads
        assert health.json()["runtime"] == {}  # a fake backend reports no runtime settings
        r = client.post("/v1/detect", files={"image": ("x.jpg", _jpeg(), "image/jpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["width"], body["height"]) == (640, 480)
        (det,) = body["detections"]
        assert det["class_name"] == "b"
        assert det["score"] == pytest.approx(0.9)
        assert det["box_xyxy"] == pytest.approx([288, 224, 352, 256])
        assert body["batch_size"] == 1
        metrics = client.get("/metrics").text
        assert 'tsd_requests_total{endpoint="/v1/detect",status="200"} 1.0' in metrics
        assert "tsd_batch_size_count 1.0" in metrics
        assert 'tsd_model_info{backend="fake",model="fake.xml"} 1.0' in metrics
    assert backend.batch_sizes[:8] == list(range(1, 9))  # warm-up at every batch size


def test_bad_inputs_are_rejected() -> None:
    with _client(FakeBackend(), max_upload_bytes=4096, max_image_pixels=10_000) as client:
        bad = client.post("/v1/detect", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
        assert bad.status_code == 400
        big = client.post("/v1/detect", files={"image": ("x.jpg", b"\0" * 200_000, "image/jpeg")})
        assert big.status_code == 413
        too_many_pixels = client.post(
            "/v1/detect", files={"image": ("x.jpg", _jpeg(200, 200), "image/jpeg")}
        )
        assert too_many_pixels.status_code == 413
        assert "pixels" in too_many_pixels.json()["detail"]
        missing = client.post("/v1/detect")
        assert missing.status_code == 422
        metrics = client.get("/metrics").text
        assert 'status="413"} 2.0' in metrics


def test_full_queue_returns_429() -> None:
    backend = FakeBackend(delay_s=0.3)
    with _client(
        backend, queue_size=1, max_batch_size=1, max_wait_ms=0, max_inflight=64
    ) as client:  # admission never refuses: every 429 here is the full queue's

        def post(_: int) -> tuple[int, str | None]:
            files = {"image": ("x.jpg", _jpeg(), "image/jpeg")}
            r = client.post("/v1/detect", files=files)
            return r.status_code, r.headers.get("retry-after")

        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(post, range(6)))
        codes = [code for code, _ in results]
        assert 200 in codes
        assert 429 in codes
        assert set(codes) <= {200, 429}
        assert {retry for code, retry in results if code == 429} == {"1"}
        metrics = client.get("/metrics").text
        assert 'tsd_rejected_total{reason="queue_full"}' in metrics


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_batcher_groups_concurrent_requests() -> None:
    calls: list[int] = []

    def infer(items: Sequence[int]) -> list[int]:
        calls.append(len(items))
        return [2 * i for i in items]

    async def main() -> list[int]:
        batcher: MicroBatcher[int, int] = MicroBatcher(infer, max_batch_size=4, max_wait_ms=50)
        batcher.start()
        results = await asyncio.gather(*(batcher.submit(i) for i in range(6)))
        await batcher.stop()
        assert all(info.batch_size in (2, 4) for _, info in results)
        return [out for out, _ in results]

    assert _run(main()) == [0, 2, 4, 6, 8, 10]
    assert calls == [4, 2]


def test_batch_size_one_disables_batching() -> None:
    calls: list[int] = []

    def infer(items: Sequence[int]) -> list[int]:
        calls.append(len(items))
        return list(items)

    async def main() -> None:
        batcher: MicroBatcher[int, int] = MicroBatcher(infer, max_batch_size=1, max_wait_ms=50)
        batcher.start()
        await asyncio.gather(*(batcher.submit(i) for i in range(5)))
        await batcher.stop()

    _run(main())
    assert calls == [1] * 5


def test_batcher_back_pressure_errors_and_shutdown() -> None:
    release = threading.Event()

    def blocking(items: Sequence[int]) -> list[int]:
        release.wait(5)
        if 99 in items:
            raise ValueError("boom")
        return list(items)

    async def main() -> None:
        batcher: MicroBatcher[int, int] = MicroBatcher(
            blocking, max_batch_size=1, max_wait_ms=0, queue_size=2
        )
        with pytest.raises(RuntimeError, match="not running"):
            await batcher.submit(1)
        batcher.start()
        first = asyncio.create_task(batcher.submit(1))
        await asyncio.sleep(0.05)  # the worker has taken it and is blocked in inference
        queued = [asyncio.create_task(batcher.submit(i)) for i in (2, 99)]
        await asyncio.sleep(0.05)
        assert batcher.depth == 2
        with pytest.raises(QueueFullError):
            await batcher.submit(3)
        release.set()
        assert (await first)[0] == 1
        assert (await queued[0])[0] == 2
        with pytest.raises(ValueError, match="boom"):
            await queued[1]
        # Shutdown while a batch is running and another request waits: nobody hangs.
        release.clear()
        running = asyncio.create_task(batcher.submit(4))
        waiting = asyncio.create_task(batcher.submit(5))
        await asyncio.sleep(0.05)
        stopping = asyncio.create_task(batcher.stop())
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(stopping, 5)
        for task in (running, waiting):
            done, _ = await asyncio.wait({task}, timeout=5)
            assert done, "a request was left waiting after stop()"
            if task.exception() is not None:
                assert "shutting down" in str(task.exception())

    _run(main())
    with pytest.raises(ValueError, match="invalid"):
        MicroBatcher(list, max_batch_size=0)


@pytest.mark.weights
@pytest.mark.slow
def test_real_model_end_to_end(weights_path: Path, sample_images: list[np.ndarray]) -> None:
    cfg = ServeSettings(model_path=weights_path, threads=2, warmup_iterations=1)
    with TestClient(create_app(cfg)) as client:
        ok, enc = cv2.imencode(".jpg", sample_images[0])
        assert ok
        r = client.post("/v1/detect", files={"image": ("bus.jpg", enc.tobytes(), "image/jpeg")})
        assert r.status_code == 200
        names = {d["class_name"] for d in r.json()["detections"]}
        assert {"bus", "person"} <= names
        assert r.json()["backend"] == "pytorch-fp32"


# --- admission control, disconnects, frames, thin images, metrics -------------------------------


class BlockingBackend(FakeBackend):
    """Blocks every inference until ``release`` is set."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self.release = release

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self.release.wait(10)
        return super().infer(batch)


def _app(backend: FakeBackend, **settings: object) -> object:
    cfg = ServeSettings(model_path=Path("fake.xml"), warmup_iterations=0, **settings)  # type: ignore[arg-type]
    return create_app(cfg, detector=Detector(backend, ["a", "b", "c"]))


def _multipart(data: bytes) -> tuple[bytes, bytes]:
    boundary = "tsd-test-boundary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="x.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode(), (
        f"multipart/form-data; boundary={boundary}".encode()
    )


RESPONSE_HEADERS: dict[int, dict[str, str]] = {}  # last response headers by status (raw posts)


async def _raw_post(
    app: object, data: bytes, disconnect: asyncio.Event | None = None, content_length: bool = True
) -> tuple[int, int]:
    """POST /v1/detect straight through ASGI; returns (status, receive() calls) and keeps the
    response headers in ``RESPONSE_HEADERS``. After the body, receive() blocks until
    ``disconnect`` is set (a client that goes away) or forever."""
    body, ctype = _multipart(data)
    headers = [(b"content-type", ctype), (b"host", b"test")]
    if content_length:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/v1/detect", "raw_path": b"/v1/detect", "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 80),
        "headers": headers,
    }  # fmt: skip
    calls = 0
    status = 0

    async def receive() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "http.request", "body": body, "more_body": False}
        await (disconnect or asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = int(message["status"])  # type: ignore[call-overload]
            raw = message.get("headers") or []
            RESPONSE_HEADERS[status] = {k.decode(): v.decode() for k, v in raw}  # type: ignore[attr-defined]

    await app(scope, receive, send)  # type: ignore[operator]
    return status, calls


def test_admission_rejects_before_reading_or_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import tsd.serve.app as app_module

    decoded: list[int] = []
    real_decode = app_module.decode_image
    monkeypatch.setattr(app_module, "decode_image", lambda d: decoded.append(1) or real_decode(d))
    release = threading.Event()
    app = _app(BlockingBackend(release), max_inflight=1, max_batch_size=1)

    async def main() -> None:
        async with app.router.lifespan_context(app):  # type: ignore[attr-defined]
            first = asyncio.create_task(_raw_post(app, _jpeg()))
            await asyncio.sleep(0.2)  # admitted, decoded, blocked in inference
            status, receive_calls = await asyncio.wait_for(_raw_post(app, _jpeg()), 5)
            assert status == 429
            assert receive_calls == 0  # rejected before a byte of the upload was read
            assert RESPONSE_HEADERS[429]["retry-after"] == "1"
            release.set()
            assert (await asyncio.wait_for(first, 5))[0] == 200
            assert app.state.admission.active == 0  # type: ignore[attr-defined]

    asyncio.run(main())
    assert decoded == [1]


def test_missing_content_length_is_refused_before_reading() -> None:
    app = _app(FakeBackend())

    async def main() -> None:
        async with app.router.lifespan_context(app):  # type: ignore[attr-defined]
            status, receive_calls = await asyncio.wait_for(
                _raw_post(app, _jpeg(), content_length=False), 5
            )
            assert status == 411
            assert receive_calls == 0  # not a byte of the upload was read

    asyncio.run(main())


def test_decode_bound_overload_returns_429() -> None:
    release = threading.Event()
    backend = BlockingBackend(release)
    with _client(
        backend, max_inflight=2, max_batch_size=1, queue_size=8, warmup_iterations=0
    ) as client:

        def post(_: int) -> int:
            files = {"image": ("x.jpg", _jpeg(), "image/jpeg")}
            return client.post("/v1/detect", files=files).status_code

        with ThreadPoolExecutor(6) as pool:
            futures = [pool.submit(post, i) for i in range(6)]
            time.sleep(0.5)
            release.set()
            codes = [f.result() for f in futures]
        assert codes.count(200) == 2
        assert codes.count(429) == 4
        assert 'tsd_rejected_total{reason="admission"} 4.0' in client.get("/metrics").text


def test_disconnected_client_is_dropped_before_inference() -> None:
    release = threading.Event()
    backend = BlockingBackend(release)
    app = _app(backend, max_batch_size=1)

    async def main() -> None:
        async with app.router.lifespan_context(app):  # type: ignore[attr-defined]
            first = asyncio.create_task(_raw_post(app, _jpeg()))
            await asyncio.sleep(0.2)  # occupies the inference thread
            gone = asyncio.Event()
            second = asyncio.create_task(_raw_post(app, _jpeg(), disconnect=gone))
            await asyncio.sleep(0.2)  # decoded and queued behind the first
            gone.set()
            assert (await asyncio.wait_for(second, 5))[0] == 499
            release.set()
            assert (await asyncio.wait_for(first, 5))[0] == 200
            await asyncio.sleep(0.1)

    asyncio.run(main())
    assert backend.batch_sizes == [1]  # the abandoned request never reached the model
    text = __import__("prometheus_client").generate_latest(app.state.metrics.registry).decode()  # type: ignore[attr-defined]
    assert "tsd_client_disconnects_total 1.0" in text
    assert 'tsd_requests_total{endpoint="/v1/detect",status="499"} 1.0' in text


def test_size_is_reported_in_the_frame_of_the_boxes() -> None:
    import io

    from PIL import Image as PILImage

    exif = PILImage.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    buf = io.BytesIO()
    PILImage.fromarray(np.full((40, 80, 3), 90, np.uint8)).save(buf, "JPEG", exif=exif)
    with _client(FakeBackend()) as client:
        r = client.post("/v1/detect", files={"image": ("x.jpg", buf.getvalue(), "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert (body["width"], body["height"]) == (40, 80)
        for d in body["detections"]:
            x1, y1, x2, y2 = d["box_xyxy"]
            assert 0 <= x1 <= x2 <= 40
            assert 0 <= y1 <= y2 <= 80


@pytest.mark.parametrize(("h", "w"), [(1, 4000), (2000, 1)])
def test_thin_images_are_refused(h: int, w: int) -> None:
    ok, enc = cv2.imencode(".png", np.full((h, w, 3), 90, np.uint8))
    assert ok
    with _client(FakeBackend()) as client:
        r = client.post("/v1/detect", files={"image": ("x.png", enc.tobytes(), "image/png")})
        assert r.status_code == 400
        assert "smaller than" in r.json()["detail"]


def test_every_status_is_counted_and_timed() -> None:
    with _client(FakeBackend()) as client:
        assert client.post("/v1/detect").status_code == 422
        client.post("/v1/detect", files={"image": ("x.jpg", _jpeg(), "image/jpeg")})
        text = client.get("/metrics").text
        assert 'tsd_requests_total{endpoint="/v1/detect",status="422"} 1.0' in text
        assert (
            'tsd_http_request_duration_seconds_count{endpoint="/v1/detect",status="422"} 1.0'
            in text
        )
        assert "tsd_detect_handler_seconds_count 1.0" in text


def test_adaptive_wait_skips_the_window_for_a_lone_request() -> None:
    def infer(items: Sequence[int]) -> list[int]:
        return list(items)

    async def lone(adaptive: bool) -> float:
        batcher: MicroBatcher[int, int] = MicroBatcher(
            infer, max_batch_size=8, max_wait_ms=300, adaptive=adaptive
        )
        batcher.start()
        t0 = time.perf_counter()
        await batcher.submit(1)
        elapsed = time.perf_counter() - t0
        await batcher.stop()
        return elapsed

    assert _run(lone(adaptive=True)) < 0.15  # type: ignore[operator]
    assert _run(lone(adaptive=False)) >= 0.29  # type: ignore[operator]


def test_wait_window_starts_when_the_first_request_arrives() -> None:
    release = threading.Event()
    calls: list[int] = []

    def infer(items: Sequence[int]) -> list[int]:
        if not calls:
            release.wait(5)
        calls.append(len(items))
        return list(items)

    async def main() -> float:
        batcher: MicroBatcher[int, int] = MicroBatcher(infer, max_batch_size=8, max_wait_ms=300)
        batcher.start()
        first = asyncio.create_task(batcher.submit(0))
        await asyncio.sleep(0.35)  # its 300 ms window has closed; it is blocked in inference
        second = asyncio.create_task(batcher.submit(1))  # waits behind the blocked batch
        await asyncio.sleep(0.4)  # longer than the window: its deadline has passed
        release.set()
        await first
        t0 = time.perf_counter()
        await second
        elapsed = time.perf_counter() - t0
        await batcher.stop()
        return elapsed

    assert _run(main()) < 0.15  # type: ignore[operator]
    assert calls == [1, 1]
