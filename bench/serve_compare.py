"""Micro-batching off vs on (fixed and adaptive wait) under load, against a real ``tsd serve``.

    uv run python bench/serve_compare.py              # 3 settings x 3 runs, resumable
    uv run python bench/serve_compare.py --aggregate-only

Settings: ``off`` (max batch 1), ``fixed`` (max batch 8, always waits up to 5 ms after the first
queued request arrived) and ``adaptive`` (max batch 8, 5 ms, but no wait when nothing else is
queued). One *session* = one server lifetime for one setting and run index, under
``flock <lock>``: start ``tsd serve`` pinned to CPUs 0-2 (2 inference threads, 1 decode thread,
OpenVINO INT8 YOLO11n by default), wait for ``/healthz``, run ``bench/load_test.py``'s closed
loop pinned to CPU 3 at each concurrency level (1, 4, 8, 16 by default; 20 s each after 3 s of
warm-up), stop the server. A session holds the lock for about 2 minutes. Sessions run in rounds
(run 1 of every setting, then run 2, ...), and the tables report the median over the runs with
the range.

Like ``run_benchmarks.py``, every level records the CPU use of processes other than the server
and the load generator; a level above 0.3 cores is re-run in a later session (attempts are kept
as ``*.attempt<k>.json``, never overwritten), and after ``--max-attempts`` the least
contaminated attempt is used and flagged (``tsd.benchmark.measure_with_retries``). The script
exits non-zero if a level ends without a result, and refuses to measure uncommitted code unless
``--allow-dirty``. Results: ``<out>/<setting>_c<concurrency>.run<k>.json``,
``<out>/serving.json`` and ``<out>/serving.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import check_target, run_load
from tsd.benchmark import (
    EX_TEMPFAIL,
    cpu_sample,
    measure_with_retries,
    model_sha256,
    pick_images,
    quiet_check,
)
from tsd.provenance import LOCK_ENV, DirtyTreeError, git_state, provenance, require_clean

REPO = Path(__file__).resolve().parents[1]
SETTINGS = {
    "off": {"TSD_MAX_BATCH_SIZE": "1", "TSD_MAX_WAIT_MS": "0", "TSD_BATCH_WAIT": "fixed"},
    "fixed": {"TSD_MAX_BATCH_SIZE": "8", "TSD_MAX_WAIT_MS": "5", "TSD_BATCH_WAIT": "fixed"},
    "adaptive": {"TSD_MAX_BATCH_SIZE": "8", "TSD_MAX_WAIT_MS": "5", "TSD_BATCH_WAIT": "adaptive"},
}
MAX_OTHER_LOAD = 0.3
NAME = re.compile(r"^(?P<setting>[a-z]+)_c(?P<level>\d+)\.run(?P<run>\d+)\.json$")


def process_cpu_seconds(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")  # utime + stime


def wait_healthy(url: str, server: subprocess.Popen[bytes], timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with {server.returncode}")
        try:  # trust_env=False: never through a proxy from the environment
            if httpx.get(f"{url}/healthz", timeout=2, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become healthy")


def session(args: argparse.Namespace) -> None:
    """One server lifetime for one batching setting (run under flock by ``main``)."""
    url = f"http://127.0.0.1:{args.port}"
    check_target(url)
    if quiet_check(3.0) > MAX_OTHER_LOAD:
        sys.exit(EX_TEMPFAIL)
    env = {
        **os.environ,
        **SETTINGS[args.setting],
        "TSD_MODEL_PATH": str(args.model),
        "TSD_THREADS": "2",
        "TSD_PREPROCESS_THREADS": "1",
        "TSD_QUEUE_SIZE": "64",
        "TSD_HOST": "127.0.0.1",
        "TSD_PORT": str(args.port),
        "OMP_NUM_THREADS": "2",
    }
    tsd = Path(sys.executable).parent / "tsd"
    server = subprocess.Popen(
        ["taskset", "-c", "0-2", str(tsd), "serve"],
        env=env,
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_healthy(url, server)
        # the server's effective runtime settings, as it reports them (threads, precision hint,
        # pinning), and exactly which model files and images this session used
        health = httpx.get(f"{url}/healthz", timeout=5, trust_env=False).json()
        model_files = {"model": str(args.model), "sha256": model_sha256(args.model)}
        os.sched_setaffinity(0, {3})
        paths = pick_images(list(args.images_dir.rglob("*.jpg")), 16)
        images = [p.read_bytes() for p in paths]
        for level in args.levels:
            before, server_before = cpu_sample(), process_cpu_seconds(server.pid)
            result = asyncio.run(run_load(url, images, level, args.duration, warmup=3.0))
            after, server_after = cpu_sample(), process_cpu_seconds(server.pid)
            wall = after.wall_s - before.wall_s
            other = (after.busy_s - before.busy_s) - (after.own_s - before.own_s)
            other -= server_after - server_before
            result.update(
                setting=args.setting,
                run=args.run,
                server_env={k: v for k, v in env.items() if k.startswith("TSD_")},
                other_processes_cores=round(max(other, 0.0) / wall, 3),
                server_cpu_cores=round((server_after - server_before) / wall, 3),
                client_cpu_cores=round((after.own_s - before.own_s) / wall, 3),
                started_unix=round(time.time(), 1),
                server_health=health,
                model_files=model_files,
                images=[p.name for p in paths],
                provenance=provenance(
                    {"server_cpus": "0-2", "client_cpus": "3", "server": health.get("runtime")}
                ),
            )
            out = args.out / f"{args.setting}_c{level}.run{args.run}.json"
            out.write_text(json.dumps(result, indent=2) + "\n")
            lat = result.get("latency_ms", {})
            nan = float("nan")
            print(
                f"{args.setting} c={level}: {result['throughput_rps']:.1f} req/s,"
                f" p50 {lat.get('p50', nan):.1f} ms, p99 {lat.get('p99', nan):.1f},"
                f" errors {result['error_rate']:.3f}, batch {result['server_mean_batch_size']},"
                f" other load {result['other_processes_cores']:.2f}"
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def _stats(values: list[float]) -> dict[str, float]:
    import numpy as np

    return {"median": float(np.median(values)), "min": min(values), "max": max(values)}


def aggregate(out: Path) -> list[dict[str, Any]]:
    """Median and range over the runs of each (setting, concurrency)."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for p in sorted(out.glob("*_c*.run*.json")):
        m = NAME.match(p.name)
        if m is None:
            continue
        groups.setdefault((m["setting"], int(m["level"])), []).append(json.loads(p.read_text()))
    rows = []
    for (setting, level), runs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ok = [r for r in runs if r.get("latency_ms")]
        row: dict[str, Any] = {
            "setting": setting,
            "concurrency": level,
            "runs": len(runs),
            "throughput_rps": _stats([r["throughput_rps"] for r in runs]),
            "error_rate_max": max(r["error_rate"] for r in runs),
            "mean_batch": _stats([r["server_mean_batch_size"] or 0.0 for r in runs]),
            "other_load_max": max(r["other_processes_cores"] for r in runs),
            "contaminated": any(r.get("contaminated") for r in runs),
            "per_run": [
                {
                    "run": r.get("run"),
                    "throughput_rps": r["throughput_rps"],
                    **{f"{q}_ms": r.get("latency_ms", {}).get(q) for q in ("p50", "p95", "p99")},
                    "other_load": r["other_processes_cores"],
                    "slowest_1pct_ms": r.get("breakdown_ms", {}).get("slowest_1pct"),
                }
                for r in runs
            ],
        }
        for q in ("p50", "p95", "p99"):
            if ok:
                row[f"{q}_ms"] = _stats([r["latency_ms"][q] for r in ok])
        rows.append(row)
    (out / "serving.json").write_text(
        json.dumps({"aggregated_by": git_state(), "rows": rows}, indent=2) + "\n"
    )

    def cell(stat: dict[str, float] | None, digits: int = 1) -> str:
        if not stat:
            return "-"
        return f"{stat['median']:.{digits}f} ({stat['min']:.{digits}f}-{stat['max']:.{digits}f})"

    lines = [
        "Median over the runs, in brackets the range (min-max). `*` = at least one run stayed"
        " above the 0.3-core contamination limit.",
        "",
        "| concurrency | batching | runs | req/s | p50 ms | p95 ms | p99 ms | max error rate"
        " | mean batch | max other load (cores) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        flag = "*" if r["contaminated"] else ""
        lines.append(
            f"| {r['concurrency']} | {r['setting']}{flag} | {r['runs']}"
            f" | {cell(r['throughput_rps'])} | {cell(r.get('p50_ms'))} | {cell(r.get('p95_ms'))}"
            f" | {cell(r.get('p99_ms'))} | {r['error_rate_max']:.3f}"
            f" | {cell(r['mean_batch'], 2)} | {r['other_load_max']:.2f} |"
        )
    lines += [
        "",
        "Slowest 1% of requests at each level, mean per run (ms): client latency = outside the"
        " handler (upload, parsing, response) + decode/letterbox (incl. waiting for the decode"
        " thread) + queue wait + inference of its batch.",
        "",
        "| concurrency | batching | run | latency | outside | decode | queue wait | inference"
        " | batch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for pr in r["per_run"]:
            b = pr.get("slowest_1pct_ms")
            if not b:
                continue
            lines.append(
                f"| {r['concurrency']} | {r['setting']} | {pr['run']} | {b['latency']:.1f}"
                f" | {b['outside']:.1f} | {b['preprocess']:.1f} | {b['queue_wait']:.1f}"
                f" | {b['inference']:.1f} | {b['batch_size']:.1f} |"
            )
    (out / "serving.md").write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/yolo11n_openvino_int8/yolo11n.xml")
    )
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--levels", default="1,4,8,16")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--images-dir", type=Path, default=REPO / "data/openimages-traffic/images")
    parser.add_argument("--out", type=Path, default=REPO / "bench/results/serving")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("TSD_BENCH_LOCK", "/home/user/work/bench.lock")),
    )
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--session", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--setting", choices=list(SETTINGS), help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.levels = [int(v) for v in str(args.levels).split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.session:
        session(args)
        return
    if not args.aggregate_only:
        try:
            require_clean(args.allow_dirty)
        except DirtyTreeError as err:
            parser.error(str(err))
        env = {
            **os.environ,
            "TSD_PARENT_COMMAND": shlex.join([Path(sys.executable).name, *sys.argv]),
            LOCK_ENV: str(args.lock),  # recorded in the result's provenance (bench_lock)
        }
        for run in range(1, args.runs + 1):
            for setting in args.settings.split(","):
                by_path = {args.out / f"{setting}_c{lv}.run{run}.json": lv for lv in args.levels}
                todo = [p for p in by_path if not p.exists()]

                def command(
                    pending: list[Path],
                    setting: str = setting,
                    run: int = run,
                    by_path: dict[Path, int] = by_path,
                ) -> list[str]:
                    return [
                        "flock", str(args.lock), sys.executable, __file__, "--session",
                        "--setting", setting, "--run", str(run),
                        "--levels", ",".join(str(by_path[p]) for p in pending),
                        "--model", str(args.model), "--duration", str(args.duration),
                        "--port", str(args.port), "--images-dir", str(args.images_dir),
                        "--out", str(args.out),
                    ]  # fmt: skip

                if todo:
                    print(f"run {run}: {setting} levels {[by_path[p] for p in todo]}", flush=True)
                    measure_with_retries(
                        command, todo, lambda r: float(r["other_processes_cores"]),
                        MAX_OTHER_LOAD, args.max_attempts, args.retry_wait, env=env, cwd=REPO,
                    )  # fmt: skip
    rows = aggregate(args.out)
    expected = len(args.levels) * len(args.settings.split(","))
    if not args.aggregate_only and len(rows) < expected:
        raise SystemExit(f"only {len(rows)} of {expected} (setting, level) results")
    print((args.out / "serving.md").read_text())


if __name__ == "__main__":
    main()
