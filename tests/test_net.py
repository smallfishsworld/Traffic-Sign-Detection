"""Downloader tests against an in-process mock HTTP transport (no network)."""

from __future__ import annotations

import base64
import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tsd.net import (
    Downloader,
    DownloadError,
    DownloadItem,
    DownloadPolicy,
    RateLimiter,
    server_md5,
)

BODY = bytes(range(256)) * 40  # 10240 bytes
URL = "https://example.test/files/data.bin"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def ranged_handler(
    body: bytes, calls: list[httpx.Request]
) -> Callable[[httpx.Request], httpx.Response]:
    """A server that honours ``Range: bytes=N-`` like S3/GCS."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        rng = request.headers.get("range")
        if rng:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            if start >= len(body):
                return httpx.Response(416)
            headers = {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"}
            return httpx.Response(206, headers=headers, content=body[start:])
        return httpx.Response(200, content=body)

    return handler


def make_downloader(
    handler: Callable[[httpx.Request], httpx.Response],
    policy: DownloadPolicy | None = None,
    sleeps: list[float] | None = None,
) -> Downloader:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    recorded = sleeps if sleeps is not None else []
    return Downloader(
        policy or DownloadPolicy(retries=2),
        client=client,
        limiter=RateLimiter(1e12),
        sleep=recorded.append,
    )


def test_rate_limiter_enforces_long_run_rate() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1000.0, burst=100.0, clock=clock, sleep=clock.sleep)
    limiter.acquire(100)  # within the burst
    assert clock.sleeps == []
    for _ in range(99):
        limiter.acquire(100)
    # 9900 bytes beyond the burst at 1000 B/s.
    assert sum(clock.sleeps) == pytest.approx(9.9)


def test_rate_limiter_refills_while_idle() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1000.0, burst=500.0, clock=clock, sleep=clock.sleep)
    limiter.acquire(500)
    clock.now += 10.0  # idle: bucket refills, but never above the burst
    limiter.acquire(500)
    assert clock.sleeps == []
    limiter.acquire(500)
    assert clock.sleeps == [pytest.approx(0.5)]


def test_policy_limits() -> None:
    assert DownloadPolicy().max_connections == 1
    assert DownloadPolicy().max_bytes_per_second == 2_000_000
    with pytest.raises(ValueError, match="max_connections"):
        DownloadPolicy(max_connections=3)
    with pytest.raises(ValueError, match="max_connections"):
        DownloadPolicy(max_connections=0)
    with pytest.raises(ValueError, match="positive"):
        DownloadPolicy(max_bytes_per_second=0)


def test_download_then_skip(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    dest = tmp_path / "out" / "data.bin"
    with make_downloader(ranged_handler(BODY, calls)) as dl:
        first = dl.fetch(DownloadItem(URL, dest, expected_size=len(BODY)))
        second = dl.fetch(DownloadItem(URL, dest, expected_size=len(BODY)))
    assert first.status == "downloaded"
    assert dest.read_bytes() == BODY
    assert first.sha256 == hashlib.sha256(BODY).hexdigest()
    assert not dest.with_name("data.bin.part").exists()
    assert second.status == "skipped"
    assert second.sha256 == first.sha256
    assert len(calls) == 1  # the skip made no request
    assert calls[0].headers["user-agent"].startswith("tsd-downloader")


def test_resume_from_partial_file(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    dest = tmp_path / "data.bin"
    dest.with_name("data.bin.part").write_bytes(BODY[:3000])
    with make_downloader(ranged_handler(BODY, calls)) as dl:
        result = dl.fetch(DownloadItem(URL, dest, expected_size=len(BODY)))
    assert result.status == "resumed"
    assert dest.read_bytes() == BODY
    assert calls[0].headers["range"] == "bytes=3000-"


def test_complete_partial_file_is_finalised_on_416(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    dest = tmp_path / "data.bin"
    dest.with_name("data.bin.part").write_bytes(BODY)
    with make_downloader(ranged_handler(BODY, calls)) as dl:
        dl.fetch(DownloadItem(URL, dest, expected_size=len(BODY)))
    assert dest.read_bytes() == BODY


def test_server_ignoring_range_restarts_cleanly(tmp_path: Path) -> None:
    dest = tmp_path / "data.bin"
    dest.with_name("data.bin.part").write_bytes(b"garbage" * 10)
    with make_downloader(lambda request: httpx.Response(200, content=BODY)) as dl:
        dl.fetch(DownloadItem(URL, dest, expected_size=len(BODY)))
    assert dest.read_bytes() == BODY


def test_interrupted_transfer_is_resumed_on_retry(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    full = ranged_handler(BODY, calls)

    def flaky(request: httpx.Request) -> httpx.Response:
        if not calls:  # first request: the connection "drops" after 4000 bytes
            calls.append(request)
            return httpx.Response(
                200, headers={"Content-Length": str(len(BODY))}, content=BODY[:4000]
            )
        return full(request)

    sleeps: list[float] = []
    dest = tmp_path / "data.bin"
    with make_downloader(flaky, sleeps=sleeps) as dl:
        dl.fetch(DownloadItem(URL, dest))
    assert dest.read_bytes() == BODY
    assert calls[1].headers["range"] == "bytes=4000-"
    assert len(sleeps) == 1


def test_size_mismatch_is_an_error(tmp_path: Path) -> None:
    dest = tmp_path / "data.bin"
    with (
        make_downloader(ranged_handler(BODY, [])) as dl,
        pytest.raises(DownloadError, match="size"),
    ):
        dl.fetch(DownloadItem(URL, dest, expected_size=len(BODY) + 1))
    assert not dest.exists()
    assert not dest.with_name("data.bin.part").exists()


def test_wrong_size_existing_file_is_replaced(tmp_path: Path) -> None:
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"short")
    with make_downloader(ranged_handler(BODY, [])) as dl:
        result = dl.fetch(DownloadItem(URL, dest, expected_size=len(BODY)))
    assert result.status == "downloaded"
    assert dest.read_bytes() == BODY


def test_md5_from_server_header_is_verified(tmp_path: Path) -> None:
    good = base64.b64encode(hashlib.md5(BODY).digest()).decode()
    bad = base64.b64encode(hashlib.md5(b"other").digest()).decode()

    def serve(md5: str) -> Callable[[httpx.Request], httpx.Response]:
        return lambda request: httpx.Response(
            200, headers={"x-goog-hash": f"crc32c=AAAAAA==,md5={md5}"}, content=BODY
        )

    with make_downloader(serve(good)) as dl:
        dl.fetch(DownloadItem(URL, tmp_path / "ok.bin"))
    sleeps: list[float] = []
    with make_downloader(serve(bad), sleeps=sleeps) as dl, pytest.raises(DownloadError):
        dl.fetch(DownloadItem(URL, tmp_path / "bad.bin"))
    assert not (tmp_path / "bad.bin").exists()
    assert len(sleeps) == 2  # retried twice before giving up


def test_pinned_md5_overrides_server(tmp_path: Path) -> None:
    with make_downloader(ranged_handler(BODY, [])) as dl, pytest.raises(DownloadError):
        dl.fetch(DownloadItem(URL, tmp_path / "x.bin", expected_md5="0" * 32))
    with make_downloader(ranged_handler(BODY, [])) as dl:
        dl.fetch(DownloadItem(URL, tmp_path / "y.bin", expected_md5=hashlib.md5(BODY).hexdigest()))


def test_transient_errors_back_off_and_retry(tmp_path: Path) -> None:
    statuses = iter([503, 429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(status, content=BODY if status == 200 else b"")

    sleeps: list[float] = []
    with make_downloader(handler, DownloadPolicy(retries=3, backoff_s=1.5), sleeps) as dl:
        dl.fetch(DownloadItem(URL, tmp_path / "data.bin"))
    assert sleeps == [1.5, 7.0]  # exponential backoff, then the server's Retry-After


def test_permanent_errors_are_not_retried(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    with make_downloader(handler) as dl, pytest.raises(DownloadError, match="404"):
        dl.fetch(DownloadItem(URL, tmp_path / "data.bin"))
    assert len(calls) == 1


def test_fetch_all_respects_connection_limit(tmp_path: Path) -> None:
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return httpx.Response(200, content=request.url.path.encode())

    items = [DownloadItem(f"https://example.test/{i}", tmp_path / f"{i}.txt") for i in range(8)]
    seen: list[str] = []
    with make_downloader(handler, DownloadPolicy(max_connections=2)) as dl:
        results = dl.fetch_all(items, on_result=lambda r: seen.append(r.dest.name))
    assert [r.dest for r in results] == [i.dest for i in items]
    assert (tmp_path / "3.txt").read_text() == "/3"
    assert peak <= 2
    assert seen == [f"{i}.txt" for i in range(8)]


def test_rate_limiter_sees_every_byte(tmp_path: Path) -> None:
    acquired: list[int] = []

    class Recorder(RateLimiter):
        def acquire(self, n: int) -> None:
            acquired.append(n)

    client = httpx.Client(transport=httpx.MockTransport(ranged_handler(BODY, [])))
    policy = DownloadPolicy(chunk_size=1000)
    with Downloader(policy, client=client, limiter=Recorder(1.0)) as dl:
        dl.fetch(DownloadItem(URL, tmp_path / "data.bin"))
    assert sum(acquired) == len(BODY)


def test_server_md5_parsing() -> None:
    raw = hashlib.md5(b"abc").digest()
    gcs = httpx.Headers(
        [("x-goog-hash", "crc32c=xyz=="), ("x-goog-hash", "md5=" + base64.b64encode(raw).decode())]
    )
    assert server_md5(gcs, "storage.googleapis.com") == raw.hex()
    s3 = httpx.Headers({"etag": f'"{raw.hex()}"'})
    assert server_md5(s3, "open-images-dataset.s3.amazonaws.com") == raw.hex()
    assert server_md5(s3, "example.com") is None  # ETag semantics are server specific
    multipart = httpx.Headers({"etag": '"0123456789abcdef0123456789abcdef-3"'})
    assert server_md5(multipart, "bucket.s3.amazonaws.com") is None


def test_corrupt_same_size_file_fails_its_pinned_md5_and_is_replaced(tmp_path: Path) -> None:
    dest = tmp_path / "meta.csv"
    dest.write_bytes(bytes(len(BODY)))  # right size, wrong content
    calls: list[httpx.Request] = []
    item = DownloadItem(
        URL, dest, expected_size=len(BODY), expected_md5=hashlib.md5(BODY).hexdigest()
    )
    with make_downloader(ranged_handler(BODY, calls)) as dl:
        result = dl.fetch(item)
        again = dl.fetch(item)
    assert result.status == "downloaded"
    assert dest.read_bytes() == BODY
    assert again.status == "skipped"
    assert len(calls) == 1
    assert calls[0].headers["accept-encoding"] == "identity"


@pytest.mark.parametrize(
    "bad", [{"retries": -1}, {"backoff_s": -0.5}, {"chunk_size": 0}, {"timeout_s": 0.0}]
)
def test_policy_fields_are_validated(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        DownloadPolicy(**bad)  # type: ignore[arg-type]
