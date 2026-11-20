"""Polite, resumable, integrity-checked HTTP downloads.

The defaults are conservative on purpose (the code is meant to run on a campus network that
flags heavy or bursty traffic):

* one connection by default, never more than two (``MAX_CONNECTIONS``);
* a token-bucket bandwidth cap shared by all connections (default 2 MB/s, small burst);
* resumable: data goes to ``<dest>.part`` and an interrupted transfer continues with an HTTP
  ``Range`` request; the file is renamed into place only after it has been verified;
* size-checked against the expected size and the server's ``Content-Length``/``Content-Range``,
  and MD5-checked against a pinned hash or, when the server publishes one, the server's hash
  (GCS ``x-goog-hash``, S3 single-part ``ETag``);
* files that are already present (and of the expected size) are skipped, so a rerun re-downloads
  nothing;
* retries only transient failures (connection errors, 429, 5xx) with exponential backoff and
  honours ``Retry-After``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

import httpx

log = logging.getLogger(__name__)

MAX_CONNECTIONS = 2
DEFAULT_BYTES_PER_SECOND = 2_000_000
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


class DownloadError(RuntimeError):
    """A download failed permanently (bad status, size or checksum mismatch after retries)."""


@dataclass(frozen=True)
class DownloadPolicy:
    max_connections: int = 1
    max_bytes_per_second: float = DEFAULT_BYTES_PER_SECOND
    timeout_s: float = 60.0
    retries: int = 4
    backoff_s: float = 2.0
    chunk_size: int = 64 * 1024
    user_agent: str = "tsd-downloader/0.1"

    def __post_init__(self) -> None:
        if not 1 <= self.max_connections <= MAX_CONNECTIONS:
            raise ValueError(f"max_connections must be between 1 and {MAX_CONNECTIONS}")
        if self.max_bytes_per_second <= 0:
            raise ValueError("max_bytes_per_second must be positive")
        if self.retries < 0 or self.backoff_s < 0:
            raise ValueError("retries and backoff_s must be >= 0")
        if self.chunk_size <= 0 or self.timeout_s <= 0:
            raise ValueError("chunk_size and timeout_s must be positive")


class RateLimiter:
    """Thread-safe token bucket. ``acquire(n)`` blocks until ``n`` bytes may be consumed.

    Tokens may go negative ("debt"): a caller that takes more than is available sleeps until the
    debt is repaid, so the long-run rate across all threads stays at ``rate`` bytes/s and bursts
    never exceed ``burst`` bytes.
    """

    def __init__(
        self,
        rate: float,
        burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate = float(rate)
        self.burst = float(burst if burst is not None else max(rate / 4, 64 * 1024))
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.burst
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self, n: int) -> None:
        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
            self._last = now
            self._tokens -= n
            debt = -self._tokens
        if debt > 0:
            self._sleep(debt / self.rate)


@dataclass(frozen=True)
class DownloadItem:
    url: str
    dest: Path
    expected_size: int | None = None
    expected_md5: str | None = None  # hex digest


@dataclass(frozen=True)
class DownloadResult:
    url: str
    dest: Path
    bytes: int
    sha256: str
    status: Literal["downloaded", "resumed", "skipped"]


def server_md5(headers: httpx.Headers, host: str) -> str | None:
    """MD5 of the full object as published by the server, as a hex digest, if any.

    GCS sends ``x-goog-hash: md5=<base64>``. For S3 the ETag of a single-part upload is the hex
    MD5 of the content (multipart ETags contain a ``-`` and are not checksums), so a 32-hex-digit
    ETag is trusted only from an ``amazonaws.com`` host.
    """
    for value in headers.get_list("x-goog-hash", split_commas=True):
        key, _, digest = value.strip().partition("=")
        if key == "md5":
            try:
                return base64.b64decode(digest, validate=True).hex()
            except (binascii.Error, ValueError):
                return None
    etag = str(headers.get("etag", "")).strip('"')
    if re.fullmatch(r"[0-9a-f]{32}", etag) and host.endswith("amazonaws.com"):
        return etag
    return None


def file_digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Downloader:
    """Downloads files according to a :class:`DownloadPolicy`. Use as a context manager."""

    def __init__(
        self,
        policy: DownloadPolicy | None = None,
        client: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy = policy or DownloadPolicy()
        self.limiter = limiter or RateLimiter(self.policy.max_bytes_per_second)
        self._sleep = sleep
        self._own_client = client is None
        self._client = client or httpx.Client(
            timeout=self.policy.timeout_s,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=self.policy.max_connections,
                max_keepalive_connections=self.policy.max_connections,
            ),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def fetch_all(
        self,
        items: Iterable[DownloadItem],
        on_result: Callable[[DownloadResult], None] | None = None,
    ) -> list[DownloadResult]:
        """Download ``items`` with at most ``policy.max_connections`` transfers in flight."""
        items = list(items)
        results: list[DownloadResult] = []
        with ThreadPoolExecutor(max_workers=self.policy.max_connections) as pool:
            for result in pool.map(self.fetch, items):
                if on_result is not None:
                    on_result(result)
                results.append(result)
        return results

    def fetch(self, item: DownloadItem) -> DownloadResult:
        if item.dest.exists():
            size = item.dest.stat().st_size
            size_ok = item.expected_size is None or size == item.expected_size
            md5_ok = item.expected_md5 is None or (
                file_digest(item.dest, "md5") == item.expected_md5.lower()
            )
            if size_ok and md5_ok:
                return DownloadResult(
                    item.url, item.dest, size, file_digest(item.dest, "sha256"), "skipped"
                )
            log.warning(
                "%s (%d bytes) fails its %s check: re-downloading",
                item.dest, size, "size" if not size_ok else "MD5",
            )  # fmt: skip
            item.dest.unlink()
        item.dest.parent.mkdir(parents=True, exist_ok=True)
        part = item.dest.with_name(item.dest.name + ".part")
        resumed = part.exists() and part.stat().st_size > 0
        attempts = self.policy.retries + 1
        for attempt in range(attempts):
            try:
                self._transfer(item, part)
                break
            except _RetryableError as err:
                if attempt == attempts - 1:
                    msg = f"{item.url}: giving up after {attempts} attempts: {err}"
                    raise DownloadError(msg) from err
                delay = err.retry_after or self.policy.backoff_s * 2**attempt
                log.warning(
                    "%s: %s (attempt %d), retrying in %.1fs", item.url, err, attempt + 1, delay
                )
                self._sleep(delay)
        part.replace(item.dest)
        size = item.dest.stat().st_size
        status: Literal["downloaded", "resumed"] = "resumed" if resumed else "downloaded"
        return DownloadResult(item.url, item.dest, size, file_digest(item.dest, "sha256"), status)

    def _transfer(self, item: DownloadItem, part: Path) -> None:
        offset = part.stat().st_size if part.exists() else 0
        if item.expected_size is not None and offset > item.expected_size:
            part.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        total: int | None
        with self._stream(item.url, headers) as resp:
            if resp.status_code == 416 and offset and offset == item.expected_size:
                total = offset  # the partial file is already complete
            elif resp.status_code == 416:
                part.unlink(missing_ok=True)
                raise _RetryableError("range not satisfiable; restarting from byte 0")
            else:
                total = self._write_body(resp, part, offset)
            md5 = item.expected_md5 or server_md5(resp.headers, resp.url.host)
        size = part.stat().st_size
        if total is not None and size < total:
            raise _RetryableError(f"connection ended at {size} of {total} bytes")
        if (total is not None and size != total) or (
            item.expected_size is not None and size != item.expected_size
        ):
            part.unlink()
            raise DownloadError(
                f"{item.url}: size {size} != expected {item.expected_size or total}"
            )
        if md5 is not None and file_digest(part, "md5") != md5.lower():
            part.unlink()
            raise _RetryableError("MD5 mismatch; restarting from byte 0")

    def _write_body(self, resp: httpx.Response, part: Path, offset: int) -> int | None:
        if resp.status_code in _RETRYABLE_STATUS:
            raise _RetryableError(f"HTTP {resp.status_code}", _retry_after(resp))
        encoding = resp.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):  # sizes, ranges and hashes would not match
            raise DownloadError(f"{resp.url}: server sent Content-Encoding {encoding}")
        if resp.status_code == 206:
            match = _CONTENT_RANGE.fullmatch(resp.headers.get("content-range", ""))
            if not match or int(match.group(1)) != offset:
                part.unlink(missing_ok=True)
                raise _RetryableError("unexpected Content-Range; restarting from byte 0")
            total = None if match.group(3) == "*" else int(match.group(3))
            mode = "ab"
        elif resp.status_code == 200:
            length = resp.headers.get("content-length")
            total = int(length) if length is not None else None
            mode = "wb"  # the server ignored the Range header: start over
        else:
            raise DownloadError(f"{resp.url}: HTTP {resp.status_code}")
        try:
            with part.open(mode) as f:
                for chunk in resp.iter_bytes(self.policy.chunk_size):
                    self.limiter.acquire(len(chunk))
                    f.write(chunk)
        except httpx.TransportError as err:
            raise _RetryableError(f"transfer interrupted: {err}") from err
        return total

    @contextmanager
    def _stream(self, url: str, headers: dict[str, str]) -> Iterator[httpx.Response]:
        # identity: sizes, Range offsets and hashes all refer to the bytes on disk
        headers = {"User-Agent": self.policy.user_agent, "Accept-Encoding": "identity", **headers}
        try:
            with self._client.stream("GET", url, headers=headers) as resp:
                yield resp
        except httpx.TransportError as err:
            raise _RetryableError(f"transport error: {err}") from err


class _RetryableError(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if value is None:
        return None
    try:
        return min(float(value), 300.0)
    except ValueError:
        return None
