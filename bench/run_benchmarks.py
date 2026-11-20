"""Run the CPU inference benchmark matrix (repeated) and write JSON + Markdown results.

    uv run python bench/run_benchmarks.py                    # whole matrix, 3 rounds, resumable
    uv run python bench/run_benchmarks.py --models yolo11n   # a chunk
    uv run python bench/run_benchmarks.py --aggregate-only   # rebuild the tables

The matrix is measured ``--runs`` times in rounds (round 1 = every configuration once, then round
2, ...), so slow drifts of the shared machine spread over all configurations instead of biasing
one. Every table entry is the median over the rounds with the run-to-run range, and a speedup is
the median of the per-round paired ratios (run k against PyTorch's run k).

Every measurement is a separate process wrapped in ``flock <lock> ...`` (default
``/home/user/work/bench.lock``, shared with other benchmarks on this machine), so the lock is
held only while one configuration is timed (at most about 70 s, the PyTorch 1-thread runs of
YOLO11s; ``--max-seconds`` caps the timed loop). Configurations with a result file are skipped
unless ``--force``, so the matrix can be run in chunks.

The machine is shared. ``bench_one.py`` refuses to start while other processes keep the CPUs
busy (exit 75; the lock is released and the run retried after ``--retry-wait`` seconds), and
every result records the other processes' CPU use during the timed loop. A result above
``tsd.benchmark.allowed_other_load`` is kept as ``<id>.run<k>.attempt<j>.json`` and re-measured;
after ``--max-attempts`` the least contaminated attempt is used and flagged in the tables
(``tsd.benchmark.measure_with_retries``).

Every raw result records the git commit, a dirty flag, the exact command lines (this script's
as ``TSD_PARENT_COMMAND``) and the effective runtime settings. The script refuses to measure
uncommitted code unless ``--allow-dirty``.

Results: ``<out>/raw/<id>.run<k>.json`` per measurement, ``<out>/benchmarks.json`` and
``<out>/benchmarks.md`` aggregated (joined with the accuracy report from ``tsd eval``), and
``<out>/lscpu.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from tsd.benchmark import (
    aggregate,
    allowed_other_load,
    config_id,
    cpu_info,
    markdown_report,
    measure_with_retries,
    run_to_run,
)
from tsd.provenance import LOCK_ENV, DirtyTreeError, git_state, require_clean

REPO = Path(__file__).resolve().parents[1]

BACKENDS = {  # name -> (model path template, OpenVINO precision hint, OpenVINO thread pinning)
    "pytorch-fp32": ("models/{m}.pt", "f32", True),
    "onnxruntime-fp32": ("models/{m}.onnx", "f32", True),
    "openvino-fp32": ("models/{m}_openvino_fp32/{m}.xml", "f32", True),
    "openvino-bf16": ("models/{m}_openvino_fp32/{m}.xml", "bf16", True),
    "openvino-int8": ("models/{m}_openvino_int8/{m}.xml", "f32", True),
    "openvino-int8-bf16": ("models/{m}_openvino_int8/{m}.xml", "bf16", True),
    # Ablation: OpenVINO pins its threads by default, PyTorch and ONNX Runtime do not.
    "openvino-fp32-nopin": ("models/{m}_openvino_fp32/{m}.xml", "f32", False),
    "openvino-int8-bf16-nopin": ("models/{m}_openvino_int8/{m}.xml", "bf16", False),
}
# (mode, threads, batch size, warm-up calls, timed calls, tile size)
MATRIX = [
    ("single", 1, 1, 20, 200, 0),
    ("single", 4, 1, 20, 200, 0),
    ("batch", 4, 8, 3, 30, 0),
]
# Only the 4-thread single-image setting for the pinning ablation.
ONLY = {"openvino-fp32-nopin": ["single_t4_b1"], "openvino-int8-bf16-nopin": ["single_t4_b1"]}
# Latency cost of sliced inference (P2), for the backend its accuracy was evaluated with.
TILED = {"openvino-fp32": [("single", 4, 1, 5, 60, 512)]}


def bench_command(
    lock: Path,
    model: Path,
    hint: str,
    mode: str,
    threads: int,
    batch: int,
    warmup: int,
    iterations: int,
    tiles: int,
    max_seconds: float,
    out: Path,
    run: int = 1,
    cpu_pinning: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """``flock <lock> python bench/bench_one.py ...`` for one configuration, and its environment
    (thread-count variables set to ``threads``; this script's command line for provenance)."""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TSD_PARENT_COMMAND": shlex.join([Path(sys.executable).name, *sys.argv]),
        LOCK_ENV: str(lock),  # recorded in the result's provenance (bench_lock)
    }
    cmd = [
        "flock", str(lock),
        sys.executable, str(REPO / "bench/bench_one.py"),
        "--model", str(model.relative_to(REPO)),
        "--threads", str(threads),
        "--mode", mode,
        "--batch-size", str(batch),
        "--precision-hint", hint,
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--max-seconds", str(max_seconds),
        "--out", str(out),
        "--tiles", str(tiles),
        "--run", str(run),
    ]  # fmt: skip
    if not cpu_pinning:
        cmd.append("--no-cpu-pinning")
    return cmd, env


def measure(
    cmd: list[str], env: dict[str, str], out: Path, threads: int, attempts: int, wait: float
) -> None:
    """Run ``cmd`` until it produces a result with acceptable contamination (see above)."""
    measure_with_retries(
        lambda _pending: cmd,
        [out],
        lambda r: float(r["contamination"]["other_processes_cores"]),
        allowed_other_load(threads, os.cpu_count() or 1),
        attempts,
        wait,
        env=env,
        cwd=REPO,
    )


def planned(models: list[str], backends: list[str], runs: int) -> list[tuple[int, str, str]]:
    """(run, model, backend) in measuring order: whole rounds, one after the other."""
    return [(k, m, b) for k in range(1, runs + 1) for m in models for b in backends]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--models", default="yolo11n,yolo11s")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--runs", type=int, default=3, help="rounds of the whole matrix")
    parser.add_argument("--out", type=Path, default=REPO / "bench/results/benchmarks")
    parser.add_argument(
        "--accuracy", type=Path, default=REPO / "bench/results/eval/coco_classes.json"
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("TSD_BENCH_LOCK", "/home/user/work/bench.lock")),
    )
    parser.add_argument("--max-seconds", type=float, default=300.0, help="cap of each timed loop")
    parser.add_argument("--force", action="store_true", help="re-measure existing results")
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--retry-wait", type=float, default=30.0, help="seconds between attempts")
    parser.add_argument("--allow-dirty", action="store_true", help="measure uncommitted code")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        try:
            require_clean(args.allow_dirty)
        except DirtyTreeError as err:
            parser.error(str(err))
        for run, model, backend in planned(
            args.models.split(","), args.backends.split(","), args.runs
        ):
            template, hint, pinning = BACKENDS[backend]
            path = REPO / template.format(m=model)
            if not path.exists():
                print(f"skip {backend} {model}: {path} not found", file=sys.stderr)
                continue
            for mode, threads, batch, warmup, iterations, tiles in MATRIX + TILED.get(backend, []):
                slot = f"{mode}_t{threads}_b{batch}"
                if backend in ONLY and (slot not in ONLY[backend] or tiles):
                    continue
                cid = config_id(path, hint, mode, threads, batch, tiles)
                cid += "" if pinning else "_nopin"
                out = raw_dir / f"{cid}.run{run}.json"
                if out.exists() and not args.force:
                    continue
                cmd, env = bench_command(
                    args.lock, path, hint, mode, threads, batch, warmup, iterations, tiles,
                    args.max_seconds, out, run, pinning,
                )  # fmt: skip
                print(f"run {run}: {cid}", flush=True)
                measure(cmd, env, out, threads, args.max_attempts, args.retry_wait)
    lscpu = subprocess.run(
        ["lscpu"], capture_output=True, text=True, check=True, env={"LC_ALL": "C"}
    ).stdout
    (args.out / "lscpu.txt").write_text(lscpu)
    raw = [
        json.loads(p.read_text())
        for p in sorted(raw_dir.glob("*.json"))
        if ".attempt" not in p.name
    ]
    accuracy = json.loads(args.accuracy.read_text()) if args.accuracy.exists() else None
    rows = aggregate(raw, accuracy)
    cpu = cpu_info(lscpu)
    commits = sorted({str(r.get("provenance", {}).get("git", {}).get("commit")) for r in raw})
    dirty = any(r.get("provenance", {}).get("git", {}).get("dirty") for r in raw)
    doc = {
        "cpu": cpu,
        "accuracy_report": str(args.accuracy.relative_to(REPO)),
        "measured_at_commits": commits,
        "any_dirty": dirty,
        "aggregated_by": {"command": shlex.join(sys.argv), "git": git_state()},
        "run_to_run": run_to_run(rows),
        "rows": rows,
    }
    note = f"Measured at commit(s) {', '.join(c[:12] for c in commits)}" + (
        " (some from a dirty tree)." if dirty else " (clean tree)."
    )
    (args.out / "benchmarks.json").write_text(json.dumps(doc, indent=2) + "\n")
    (args.out / "benchmarks.md").write_text(markdown_report(rows, cpu, note))


if __name__ == "__main__":
    main()
