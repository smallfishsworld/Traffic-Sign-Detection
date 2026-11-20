"""The benchmark scripts' safety limits and retry policy (no network, no models)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tsd.benchmark import allowed_other_load

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))

import run_benchmarks  # noqa: E402

# --- load_test.py ------------------------------------------------------------------------------


@pytest.fixture
def load_test() -> ModuleType:
    pytest.importorskip("httpx")
    import load_test

    return load_test


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://127.0.0.5", "http://localhost"])
def test_loopback_targets_are_allowed(load_test: ModuleType, url: str) -> None:
    load_test.check_target(url)


@pytest.mark.parametrize("url", ["http://10.1.2.3:8000", "https://192.0.2.1/v1"])
def test_remote_targets_need_the_flag(load_test: ModuleType, url: str) -> None:
    with pytest.raises(load_test.TargetNotAllowedError, match="allow-remote"):
        load_test.check_target(url)
    load_test.check_target(url, allow_remote=True)


def test_non_http_targets_are_refused(load_test: ModuleType) -> None:
    with pytest.raises(load_test.TargetNotAllowedError):
        load_test.check_target("ftp://127.0.0.1/", allow_remote=True)


@pytest.mark.parametrize(
    ("concurrency", "duration", "ok"),
    [(1, 1.0, True), (64, 600.0, True), (0, 10.0, False), (65, 10.0, False), (8, 601.0, False)],
)
def test_concurrency_and_duration_limits(
    load_test: ModuleType, concurrency: int, duration: float, ok: bool
) -> None:
    if ok:
        load_test.check_limits(concurrency, duration)
    else:
        with pytest.raises(ValueError, match="must be in"):
            load_test.check_limits(concurrency, duration)


def test_metric_parsing_and_summary(load_test: ModuleType) -> None:
    text = (
        "# HELP tsd_batch_size images per batch\n"
        'tsd_batch_size_count{model="a"} 3.0\n'
        "tsd_batch_size_count_other 100.0\n"
        "tsd_batch_size_sum 12.0\n"
    )
    assert load_test.parse_metric(text, "tsd_batch_size_count") == 3.0
    assert load_test.parse_metric(text, "tsd_batch_size_sum") == 12.0
    summary = load_test.summarize([10.0, 20.0, 30.0, 5.0], [200, 200, 200, 429], 2.0)
    assert summary["ok"] == 3
    assert summary["errors"] == {"429": 1}
    assert summary["error_rate"] == 0.25
    assert summary["throughput_rps"] == 1.5
    assert summary["latency_ms"]["p50"] == 20.0  # 429s are not in the latency statistics


# --- run_benchmarks.measure: quiet pre-check and contamination retries -------------------------

FAKE = """
import json, sys
from pathlib import Path
counter, out, plan = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
attempt = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(attempt + 1))
step = plan[min(attempt, len(plan) - 1)]
if step == "busy":
    sys.exit(75)
if step == "fail":
    sys.exit(3)
out.write_text(json.dumps({"contamination": {"other_processes_cores": step}}))
"""


def _measure(tmp_path: Path, plan: list[object], attempts: int = 5) -> dict[str, object]:
    out = tmp_path / "result.json"
    cmd = [sys.executable, "-c", FAKE, str(tmp_path / "n"), str(out), json.dumps(plan)]
    run_benchmarks.measure(cmd, dict(os.environ), out, 4, attempts, 0.0)
    result: dict[str, object] = json.loads(out.read_text())
    return result


LIMIT = allowed_other_load(4, os.cpu_count() or 1)


def test_busy_machine_is_retried_and_contaminated_results_are_kept(tmp_path: Path) -> None:
    result = _measure(tmp_path, ["busy", LIMIT + 1.0, 0.0])
    # attempts count measured sessions only; the busy pre-check is counted separately
    assert result == {"contamination": {"other_processes_cores": 0.0}, "attempt": 2,
                      "busy_skips": 1, "contaminated": False}  # fmt: skip
    kept = json.loads((tmp_path / "result.attempt1.json").read_text())  # first kept file
    assert kept["contamination"]["other_processes_cores"] == LIMIT + 1.0
    assert (kept["attempt"], kept["busy_skips"]) == (1, 1)
    assert sorted(p.name for p in tmp_path.glob("result.attempt*.json")) == [
        "result.attempt1.json"  # attempt k accepted <=> k - 1 kept attempt files
    ]


def test_least_contaminated_attempt_is_used_and_flagged(tmp_path: Path) -> None:
    result = _measure(tmp_path, [LIMIT + 3.0, LIMIT + 1.0, LIMIT + 2.0], attempts=3)
    assert result["attempt"] == 2
    assert result["contaminated"] is True
    assert sorted(p.name for p in tmp_path.glob("result.attempt*.json")) == [
        "result.attempt1.json",
        "result.attempt2.json",
        "result.attempt3.json",
    ]


def test_never_quiet_and_failures_abort(tmp_path: Path) -> None:
    for name in ("busy", "fail"):
        (tmp_path / name).mkdir()
    with pytest.raises(SystemExit, match="no result for"):
        _measure(tmp_path / "busy", ["busy"], attempts=2)
    with pytest.raises(SystemExit, match="measurement failed"):
        _measure(tmp_path / "fail", ["fail"])


def test_benchmark_command_holds_the_lock_and_sets_threads(tmp_path: Path) -> None:
    cmd, env = run_benchmarks.bench_command(
        Path("/tmp/x.lock"), REPO / "models/m.onnx", "f32", "single", 2, 1, 5, 10, 0, 60.0,
        tmp_path / "o.json",
    )  # fmt: skip
    assert cmd[:2] == ["flock", "/tmp/x.lock"]
    assert cmd[cmd.index("--model") + 1] == "models/m.onnx"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "2"


# --- shared retry helper: attempt files are never overwritten -----------------------------------


def test_attempts_from_earlier_invocations_are_kept(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    old = '{"contamination": {"other_processes_cores": 9}}'
    (tmp_path / "result.attempt1.json").write_text(old)
    cmd = [sys.executable, "-c", FAKE, str(tmp_path / "n"), str(out), json.dumps([LIMIT + 1, 0.0])]
    run_benchmarks.measure(cmd, dict(os.environ), out, 4, 3, 0.0)
    assert json.loads((tmp_path / "result.attempt1.json").read_text())["contamination"] == {
        "other_processes_cores": 9
    }
    kept = json.loads((tmp_path / "result.attempt2.json").read_text())
    assert kept["contamination"]["other_processes_cores"] == LIMIT + 1
    assert json.loads(out.read_text())["contaminated"] is False


def test_rounds_measure_every_configuration_before_repeating() -> None:
    order = run_benchmarks.planned(["n", "s"], ["a", "b"], 2)
    assert order[:4] == [(1, "n", "a"), (1, "n", "b"), (1, "s", "a"), (1, "s", "b")]
    assert order[4][0] == 2


def test_int8_with_the_bf16_hint_has_its_own_label() -> None:
    from tsd.infer.detector import openvino_label

    assert run_benchmarks.BACKENDS["openvino-int8-bf16"][1] == "bf16"
    assert openvino_label("int8", "bf16", None) == "int8-bf16"
    assert openvino_label("fp32", "bf16", None) == "bf16"
    assert openvino_label("int8", "f32", False) == "int8-nopin"


# --- load_test.py: proxies, warm-up bounds, Retry-After -----------------------------------------


@pytest.mark.parametrize(
    ("warmup", "duration"),
    [(float("inf"), 10.0), (float("nan"), 10.0), (-1.0, 10.0), (61.0, 10.0), (60.0, 590.0)],
)
def test_warmup_is_bounded(load_test: ModuleType, warmup: float, duration: float) -> None:
    with pytest.raises(ValueError, match="warmup"):
        load_test.check_limits(4, duration, warmup)
    load_test.check_limits(4, 540.0, 60.0)


def test_retry_after_forms(load_test: ModuleType) -> None:
    import email.utils
    import time

    assert load_test.retry_after_seconds("0.25") == 0.25
    assert load_test.retry_after_seconds("120") == 1.0  # capped
    assert load_test.retry_after_seconds(None) == 1.0
    assert load_test.retry_after_seconds("soon") == 1.0
    past = email.utils.formatdate(time.time() - 60, usegmt=True)
    assert load_test.retry_after_seconds(past) == 0.0
    future = email.utils.formatdate(time.time() + 3600, usegmt=True)
    assert load_test.retry_after_seconds(future) == 1.0


def test_load_test_ignores_proxy_variables(
    load_test: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loopback run must not be routed through HTTP_PROXY / ALL_PROXY (here: a dead port)."""
    import asyncio

    body = b'{"total_ms": 2, "queue_wait_ms": 0, "inference_ms": 1, "batch_size": 1}'

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":")[1])
                await reader.readexactly(length)
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()

    async def main() -> dict[str, object]:
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            result: dict[str, object] = await load_test.run_load(
                f"http://127.0.0.1:{port}", [b"x"], concurrency=2, duration=0.3, warmup=0
            )
        return result

    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    result = asyncio.run(main())
    assert result["errors"] == {}
    assert int(result["ok"]) > 0  # type: ignore[call-overload]
    assert result["breakdown_ms"]["all"]["inference"] == 1.0  # type: ignore[index]


# --- serve_compare.py: medians and ranges over runs -------------------------------------------


def test_serving_aggregate_reports_medians_and_ranges(tmp_path: Path) -> None:
    pytest.importorskip("httpx")
    import serve_compare

    for run, (rps, p99) in enumerate([(30.0, 300.0), (32.0, 480.0), (31.0, 350.0)], 1):
        doc = {
            "setting": "fixed", "concurrency": 8, "run": run, "throughput_rps": rps,
            "error_rate": 0.0, "server_mean_batch_size": 4.0, "other_processes_cores": 0.1,
            "latency_ms": {"p50": 240.0, "p95": 280.0, "p99": p99},
            "contaminated": run == 2,
            "breakdown_ms": {"slowest_1pct": {
                "latency": p99, "outside": 5.0, "preprocess": 20.0, "queue_wait": 150.0,
                "inference": 100.0, "batch_size": 6.0}},
        }  # fmt: skip
        (tmp_path / f"fixed_c8.run{run}.json").write_text(json.dumps(doc))
    (tmp_path / "fixed_c8.run1.attempt1.json").write_text("{}")  # ignored
    (row,) = serve_compare.aggregate(tmp_path)
    assert row["runs"] == 3
    assert row["throughput_rps"] == {"median": 31.0, "min": 30.0, "max": 32.0}
    assert row["p99_ms"]["median"] == 350.0
    text = (tmp_path / "serving.md").read_text()
    assert "| 8 | fixed* | 3 | 31.0 (30.0-32.0) |" in text
    assert "| 8 | fixed | 2 | 480.0 | 5.0 | 20.0 | 150.0 | 100.0 | 6.0 |" in text
