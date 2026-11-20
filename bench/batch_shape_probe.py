"""Is the first batch of a new size slow? (Explains the serving p99 at concurrency 8.)

    uv run python bench/batch_shape_probe.py      # takes the lock itself; delete the output
                                                  # file to measure again

Loads the service's default model like ``tsd serve`` does (OpenVINO INT8 YOLO11n, 2 inference
threads, CPUs 0-2), warms it up the way the service's start-up did before the fix (batch 1 and
the maximum batch, 3 calls each), then calls every batch size from 2 to 7 six times in a row and
records the first call's time against the median of the other five. Then the same for a fresh
detector warmed up at every size (the service's current start-up).

Like ``run_benchmarks.py``, the measurement is a separate process under ``flock <lock>`` (default
``/home/user/work/bench.lock``; the path is recorded in the result's ``provenance.bench_lock``)
that exits with 75 if other processes use more than 0.3 cores during a 3 s pre-check (the lock is
released and the run retried after ``--retry-wait`` seconds) and records the other processes' CPU
use during the run; a run above 0.3 cores is kept as ``*.attempt<k>.json`` and measured again,
and after ``--max-attempts`` the least contaminated attempt is used and flagged
(``tsd.benchmark.measure_with_retries``). Refuses to measure uncommitted code unless
``--allow-dirty``. Output: ``bench/results/serving/batch_shape_probe.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tsd.benchmark import (
    EX_TEMPFAIL,
    cpu_sample,
    measure_with_retries,
    model_sha256,
    other_load,
    pick_images,
    quiet_check,
)
from tsd.infer import load_detector
from tsd.io import load_image
from tsd.provenance import LOCK_ENV, DirtyTreeError, provenance, require_clean

REPO = Path(__file__).resolve().parents[1]
MAX_OTHER_LOAD = 0.3


def probe(model: Path, images: list[np.ndarray], threads: int, warm_all: bool) -> dict[str, Any]:
    det = load_detector(model, threads=threads)
    warm = range(1, 9) if warm_all else (1, 8)
    for b in warm:
        for _ in range(3):
            det.predict(images[:b])
    sizes: dict[str, Any] = {}
    for b in range(2, 8):
        times = []
        for _ in range(6):
            t0 = time.perf_counter()
            det.predict(images[:b])
            times.append((time.perf_counter() - t0) * 1e3)
        rest = statistics.median(times[1:])
        sizes[str(b)] = {
            "first_ms": round(times[0], 2),
            "median_of_next_5_ms": round(rest, 2),
            "first_over_rest": round(times[0] / rest, 2),
        }
    return {"warmed_sizes": list(warm), "sizes": sizes, "runtime": det.runtime_settings()}


def one(args: argparse.Namespace) -> None:
    """The measurement itself (run under flock by ``main``)."""
    os.sched_setaffinity(0, {int(c) for c in args.cpus.split(",")})
    if quiet_check(3.0) > MAX_OTHER_LOAD:
        sys.exit(EX_TEMPFAIL)
    paths = pick_images(sorted((REPO / "data/openimages-traffic/images").rglob("*.jpg")), 8)
    images = [load_image(p) for p in paths]
    before = cpu_sample()
    results = {
        "service_start_up_before_fix": probe(args.model, images, args.threads, warm_all=False),
        "every_size_warmed": probe(args.model, images, args.threads, warm_all=True),
    }
    doc = {
        "model": str(args.model),
        "model_sha256": model_sha256(args.model),
        "threads": args.threads,
        "cpus": args.cpus,
        "images": [p.name for p in paths],
        **results,
        "contamination": other_load(before, cpu_sample()),
        "provenance": provenance(),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(doc, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/yolo11n_openvino_int8/yolo11n.xml")
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", default="0,1,2")
    parser.add_argument(
        "--out", type=Path, default=REPO / "bench/results/serving/batch_shape_probe.json"
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("TSD_BENCH_LOCK", "/home/user/work/bench.lock")),
    )
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        one(args)
        return
    if args.out.exists():
        parser.error(f"{args.out} exists; delete it to measure again")
    try:
        require_clean(args.allow_dirty)
    except DirtyTreeError as err:
        parser.error(str(err))
    env = {
        **os.environ,
        "TSD_PARENT_COMMAND": shlex.join([Path(sys.executable).name, *sys.argv]),
        LOCK_ENV: str(args.lock),
    }
    cmd = [
        "flock", str(args.lock), sys.executable, __file__, "--one",
        "--model", str(args.model), "--threads", str(args.threads), "--cpus", args.cpus,
        "--result", str(args.out),
    ]  # fmt: skip
    measure_with_retries(
        lambda _pending: cmd,
        [args.out],
        lambda r: float(r["contamination"]["other_processes_cores"]),
        MAX_OTHER_LOAD,
        args.max_attempts,
        args.retry_wait,
        env=env,
        cwd=REPO,
    )
    doc = json.loads(args.out.read_text())
    for name in ("service_start_up_before_fix", "every_size_warmed"):
        sizes = doc[name]["sizes"]
        print(name, {b: (s["first_ms"], s["median_of_next_5_ms"]) for b, s in sizes.items()})
    print("other load", doc["contamination"]["other_processes_cores"], "attempt", doc["attempt"])


if __name__ == "__main__":
    main()
