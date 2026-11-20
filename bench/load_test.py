"""Closed-loop HTTP load test for ``POST /v1/detect`` (asyncio + httpx).

    uv run python bench/load_test.py --concurrency 8 --duration 20 --out result.json

Safety limits (see the README):

* the target must resolve to a loopback address (127.0.0.0/8, ::1) unless ``--allow-remote``,
  and proxy environment variables (``HTTP_PROXY``, ``ALL_PROXY``, ...) are ignored, so loopback
  traffic can never be routed to a proxy on another machine;
* concurrency (requests in flight) defaults to 8 and is capped at 64; ``--max-rps`` caps the
  request rate (default: no cap on loopback, 10 requests/s with ``--allow-remote``);
* warm-up (at most 60 s) plus duration (default 20 s) is at most 600 s, there are no retries, a
  429 response's ``Retry-After`` is honoured (capped at 1 s; seconds or an HTTP date), a worker
  backs off exponentially (50 ms doubling to 1 s) after a transport error or a 5xx, and the run
  stops early once more than half of at least 20 requests have failed.

Each of ``concurrency`` workers sends one image, waits for the answer, and sends the next. The
report has throughput (successful responses per second), latency percentiles of successful
responses, the error rate, the server's mean batch size during the run (from ``/metrics``),
and a latency breakdown from the timings each response reports (time outside the handler:
upload and parsing, response, client; decode + letterbox; queue wait; inference), overall and
for the slowest 1% and 5% of requests.
"""

from __future__ import annotations

import argparse
import asyncio
import email.utils
import ipaddress
import json
import math
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

MAX_CONCURRENCY = 64
MAX_DURATION_S = 600.0
MAX_WARMUP_S = 60.0
REMOTE_DEFAULT_MAX_RPS = 10.0
ABORT_ERROR_RATE = 0.5
ABORT_MIN_REQUESTS = 20
BACKOFF_START_S = 0.05
BACKOFF_MAX_S = 1.0


class TargetNotAllowedError(ValueError):
    pass


def check_target(url: str, allow_remote: bool = False) -> None:
    """Refuse targets that do not resolve exclusively to loopback addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise TargetNotAllowedError(f"not an http(s) URL: {url}")
    if allow_remote:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as err:
        raise TargetNotAllowedError(f"cannot resolve {parts.hostname}") from err
    addresses = {str(info[4][0]) for info in infos}
    remote = sorted(a for a in addresses if not ipaddress.ip_address(a.split("%")[0]).is_loopback)
    if remote:
        raise TargetNotAllowedError(
            f"{parts.hostname} resolves to non-loopback {remote}; pass --allow-remote to load-test"
            " another machine"
        )


def check_limits(
    concurrency: int, duration: float, warmup: float = 0.0, max_rps: float | None = None
) -> None:
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be in 1..{MAX_CONCURRENCY}")
    if not (math.isfinite(duration) and 0 < duration <= MAX_DURATION_S):
        raise ValueError(f"duration must be in (0, {MAX_DURATION_S}] seconds")
    if not (math.isfinite(warmup) and 0 <= warmup <= MAX_WARMUP_S):
        raise ValueError(f"warmup must be in [0, {MAX_WARMUP_S}] seconds")
    if warmup + duration > MAX_DURATION_S:
        raise ValueError(f"warmup + duration must be at most {MAX_DURATION_S} seconds")
    if max_rps is not None and not (math.isfinite(max_rps) and max_rps > 0):
        raise ValueError("max_rps must be a positive number")


def retry_after_seconds(value: str | None, cap: float = 1.0) -> float:
    """A ``Retry-After`` header (delta seconds or an HTTP date) as seconds, capped; 1 s (or the
    cap) when missing or unparseable."""
    default = min(1.0, cap)
    if not value:
        return default
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return default
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds):
        return default
    return min(max(seconds, 0.0), cap)


class RateLimiter:
    """At most ``rate`` request starts per second over all workers (None: no cap)."""

    def __init__(self, rate: float | None) -> None:
        self.interval = 0.0 if rate is None else 1.0 / rate
        self.next_slot = 0.0

    async def wait(self) -> None:
        if self.interval == 0.0:
            return
        now = time.monotonic()
        slot = max(now, self.next_slot)
        self.next_slot = slot + self.interval
        if slot > now:
            await asyncio.sleep(slot - now)


def breakdown(samples: list[dict[str, float]]) -> dict[str, Any]:
    """Where the time of successful requests went, from the timings each response reports:
    ``outside`` = client latency - handler time (upload + parsing, response, client),
    ``preprocess`` = handler - queue wait - inference (decode + letterbox + its thread-pool
    wait), ``queue_wait`` and ``inference`` (of the whole batch). Means overall and over the
    slowest 1% and 5% by client latency, plus percentiles of each part."""
    if not samples:
        return {}
    parts = ("latency", "outside", "preprocess", "queue_wait", "inference", "batch_size")
    arrays = {k: np.array([s[k] for s in samples], dtype=np.float64) for k in parts}
    order = np.argsort(arrays["latency"])
    out: dict[str, Any] = {"n": len(samples)}
    for name, frac in (("all", 1.0), ("slowest_5pct", 0.05), ("slowest_1pct", 0.01)):
        idx = order[-max(1, round(len(samples) * frac)) :]
        out[name] = {k: round(float(arrays[k][idx].mean()), 3) for k in parts}
        out[name]["count"] = len(idx)
    out["percentiles"] = {
        k: {f"p{q}": round(float(np.percentile(arrays[k], q)), 3) for q in (50, 95, 99)}
        for k in parts
    }
    return out


def parse_metric(text: str, name: str) -> float:
    """Sum of all samples of ``name`` in a Prometheus text exposition."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and (line[len(name)] in " {"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def summarize(latencies_ms: list[float], statuses: list[int], seconds: float) -> dict[str, Any]:
    ok = [lat for lat, s in zip(latencies_ms, statuses, strict=True) if s == 200]
    errors: dict[str, int] = {}
    for s in statuses:
        if s != 200:
            errors[str(s)] = errors.get(str(s), 0) + 1
    out: dict[str, Any] = {
        "requests": len(statuses),
        "ok": len(ok),
        "errors": errors,
        "error_rate": (len(statuses) - len(ok)) / len(statuses) if statuses else 0.0,
        "seconds": seconds,
        "throughput_rps": len(ok) / seconds if seconds > 0 else 0.0,
    }
    if ok:
        p50, p95, p99 = np.percentile(ok, [50, 95, 99])
        out["latency_ms"] = {
            "mean": float(np.mean(ok)),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "max": float(np.max(ok)),
        }
    return out


async def run_load(
    url: str,
    images: list[bytes],
    concurrency: int = 8,
    duration: float = 20.0,
    warmup: float = 2.0,
    timeout: float = 30.0,
    max_rps: float | None = None,
) -> dict[str, Any]:
    check_limits(concurrency, duration, warmup, max_rps)
    base = url.rstrip("/")
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    limiter = RateLimiter(max_rps)
    # trust_env=False: never route (loopback) traffic through HTTP_PROXY / ALL_PROXY.
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
        counter = 0
        aborted = False

        def next_image() -> bytes:
            nonlocal counter
            counter += 1
            return images[counter % len(images)]

        async def worker(
            stop_at: float,
            latencies: list[float],
            statuses: list[int],
            samples: list[dict[str, float]],
        ) -> None:
            nonlocal aborted
            backoff = 0.0
            while time.monotonic() < stop_at and not aborted:
                await limiter.wait()
                files = {"image": ("image.jpg", next_image(), "image/jpeg")}
                t0 = time.perf_counter()
                try:
                    response = await client.post(f"{base}/v1/detect", files=files)
                    status = response.status_code
                except httpx.HTTPError:
                    status = 599  # transport error / timeout
                    response = None
                latency = (time.perf_counter() - t0) * 1e3
                latencies.append(latency)
                statuses.append(status)
                if status == 200 and response is not None:
                    backoff = 0.0
                    try:
                        body = response.json()
                        handler = float(body["total_ms"])
                        queue, infer = float(body["queue_wait_ms"]), float(body["inference_ms"])
                        samples.append(
                            {
                                "latency": latency,
                                "outside": latency - handler,
                                "preprocess": handler - queue - infer,
                                "queue_wait": queue,
                                "inference": infer,
                                "batch_size": float(body["batch_size"]),
                            }
                        )
                    except (ValueError, KeyError, TypeError):
                        pass
                elif status == 429 and response is not None:
                    await asyncio.sleep(retry_after_seconds(response.headers.get("retry-after")))
                elif status >= 500:
                    backoff = min(BACKOFF_MAX_S, max(BACKOFF_START_S, 2 * backoff))
                    await asyncio.sleep(backoff)
                failed = sum(1 for s in statuses if s != 200)
                if (
                    len(statuses) >= ABORT_MIN_REQUESTS
                    and failed / len(statuses) > ABORT_ERROR_RATE
                ):
                    aborted = True

        if warmup > 0:
            await asyncio.gather(
                *(worker(time.monotonic() + warmup, [], [], []) for _ in range(concurrency))
            )
            aborted = False
        before = (await client.get(f"{base}/metrics")).text
        latencies: list[float] = []
        statuses: list[int] = []
        samples: list[dict[str, float]] = []
        t0 = time.monotonic()
        await asyncio.gather(
            *(worker(t0 + duration, latencies, statuses, samples) for _ in range(concurrency))
        )
        seconds = time.monotonic() - t0
        after = (await client.get(f"{base}/metrics")).text
    result = summarize(latencies, statuses, seconds)
    batches = parse_metric(after, "tsd_batch_size_count") - parse_metric(
        before, "tsd_batch_size_count"
    )
    images_done = parse_metric(after, "tsd_batch_size_sum") - parse_metric(
        before, "tsd_batch_size_sum"
    )
    result["server_mean_batch_size"] = images_done / batches if batches else None
    result["concurrency"] = concurrency
    result["max_rps"] = max_rps
    result["aborted_on_errors"] = aborted
    result["breakdown_ms"] = breakdown(samples)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8, help=f"1..{MAX_CONCURRENCY}")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds (<= 600)")
    parser.add_argument("--warmup", type=float, default=2.0, help="seconds, not recorded")
    parser.add_argument("--images-dir", type=Path, default=Path("data/openimages-traffic/images"))
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--allow-remote", action="store_true", help="allow non-loopback targets")
    parser.add_argument(
        "--max-rps",
        type=float,
        help=f"request-rate cap (default: none on loopback, {REMOTE_DEFAULT_MAX_RPS} remote)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.max_rps is None and args.allow_remote:
        args.max_rps = REMOTE_DEFAULT_MAX_RPS
    try:
        check_target(args.url, args.allow_remote)
        check_limits(args.concurrency, args.duration, args.warmup, args.max_rps)
    except ValueError as err:
        parser.error(str(err))
    from tsd.benchmark import pick_images

    paths = pick_images(list(args.images_dir.rglob("*.jpg")), args.num_images)
    if not paths:
        parser.error(f"no .jpg images under {args.images_dir}")
    images = [p.read_bytes() for p in paths]
    result = asyncio.run(
        run_load(
            args.url, images, args.concurrency, args.duration, args.warmup, max_rps=args.max_rps
        )
    )
    result["images"] = [p.name for p in paths]
    from tsd.provenance import provenance

    result["provenance"] = provenance()
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
