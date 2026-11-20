"""Where OpenVINO spends the time: per-node execution kernels of every OpenVINO configuration.

    uv run python bench/profile_ops.py                  # yolo11n and yolo11s, resumable
    uv run python bench/profile_ops.py --aggregate-only

Each configuration of ``run_benchmarks.BACKENDS`` that runs on OpenVINO (FP32 IR with the f32 or
bf16 hint, INT8 IR with the f32 or bf16 hint) is built through ``OpenVINOBackend`` exactly like
the benchmark (CPU plugin, latency hint, one stream, 4 threads, explicit precision hint) plus
``PERF_COUNT``, run 30 times untimed and 100 times profiled on one letterboxed evaluation image
(batch 1). The per-node ``real_time`` is summed by kernel (``exec_type``, e.g.
``brgconv_avx512_amx_I8``), by node type and by the precision suffix of the kernel name. The
infer request's wall time is recorded too (``request_wall_ms``). The two need not agree: the
node sum leaves out the request's own overhead (input/output copies, scheduling), but per-node
times are measured by the plugin and can add up to more than the wall time (measured here for
YOLO11s), so they show where the time goes, not an exact decomposition. Profiling adds some
overhead of its own; compare configurations, not these numbers with the benchmark's.

Like the benchmark, each configuration is a separate process under ``flock <lock>`` that first
checks that the machine is quiet (exit 75 otherwise) and records the other processes' CPU use
during the profiled runs; contaminated results are re-measured
(``tsd.benchmark.measure_with_retries``). Output: ``<out>/profile/<model>_<label>.json`` per
configuration and ``<out>/profile_<model>.json`` / ``.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_benchmarks import BACKENDS
from tsd.benchmark import (
    EX_TEMPFAIL,
    allowed_other_load,
    cpu_sample,
    measure_with_retries,
    other_load,
    quiet_check,
)
from tsd.provenance import LOCK_ENV, DirtyTreeError, provenance, require_clean

REPO = Path(__file__).resolve().parents[1]
CONFIGS = {  # label -> (IR path template, precision hint): the pinned OpenVINO benchmark configs
    name: (template, hint)
    for name, (template, hint, pinning) in BACKENDS.items()
    if name.startswith("openvino-") and pinning
}
PRECISION = re.compile(r"_(f32|fp32|bf16|f16|fp16|i8|u8|i32|i64)$", re.IGNORECASE)


def profile(
    xml: Path, hint: str, label: str, image: Path, threads: int, runs: int
) -> dict[str, Any]:
    import numpy as np
    import openvino as ov
    import openvino.properties as props

    from tsd.infer.backends import OpenVINOBackend
    from tsd.infer.preprocess import preprocess
    from tsd.io import load_image

    backend = OpenVINOBackend(
        xml, threads, "bf16" if hint == "bf16" else "f32", label.removeprefix("openvino-"),
        extra_config={props.enable_profiling: True},
    )  # fmt: skip
    request = backend.request
    batch, _ = preprocess([load_image(image)])
    for _ in range(30):
        request.infer({0: batch})
    by_kernel: dict[str, float] = defaultdict(float)
    by_type: dict[str, float] = defaultdict(float)
    by_precision: dict[str, float] = defaultdict(float)
    nodes_by_precision: dict[str, int] = defaultdict(int)
    totals, walls = [], []
    before = cpu_sample()
    for i in range(runs):
        t0 = time.perf_counter()
        request.infer({0: batch})
        walls.append((time.perf_counter() - t0) * 1e3)
        total = 0.0
        for info in request.get_profiling_info():
            if info.status != ov.ProfilingInfo.Status.EXECUTED:
                continue
            ms = info.real_time.total_seconds() * 1e3
            total += ms
            m = PRECISION.search(info.exec_type)
            precision = m.group(1).upper().replace("FP", "F") if m else "other"
            by_kernel[info.exec_type] += ms
            by_type[info.node_type] += ms
            by_precision[precision] += ms
            if i == 0:
                nodes_by_precision[precision] += 1
        totals.append(total)
    contamination = other_load(before, cpu_sample())

    def share(d: dict[str, float]) -> list[dict[str, Any]]:
        grand = sum(d.values())
        return [
            {"name": k, "ms_per_inference": round(v / runs, 3), "share": round(v / grand, 4)}
            for k, v in sorted(d.items(), key=lambda kv: -kv[1])
        ]

    return {
        "label": label,
        "model": str(xml.relative_to(REPO)),
        "precision_hint": hint,
        "threads": threads,
        "runs": runs,
        "image": image.name,
        "sum_of_node_times_ms": {
            "median": round(float(np.median(totals)), 3),
            "min": round(min(totals), 3),
        },
        "request_wall_ms": {
            "median": round(float(np.median(walls)), 3),
            "min": round(min(walls), 3),
        },
        "by_precision": share(by_precision),
        "executed_nodes_by_precision": dict(nodes_by_precision),
        "by_node_type": share(by_type)[:15],
        "by_kernel": share(by_kernel)[:15],
        "contamination": contamination,
        "openvino": ov.get_version(),
        "provenance": provenance(backend.settings()),
    }


def markdown(model: str, results: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"# OpenVINO per-node profile ({model}, batch 1, 4 threads)",
        "",
        "Sum of per-node `real_time` (median over the profiled runs), the infer request's wall",
        "time (median; per-node times are the plugin's and need not add up to it), and the split",
        "of node time by the precision of the kernel that ran each node. Profiling overhead",
        "included.",
        "",
        "| configuration | node sum (ms) | request wall (ms) | time by kernel precision"
        " | nodes by precision | other load (cores) |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in results.items():
        prec = ", ".join(f"{p['name']} {p['share']:.0%}" for p in r["by_precision"])
        nodes = ", ".join(f"{k} {v}" for k, v in sorted(r["executed_nodes_by_precision"].items()))
        lines.append(
            f"| {label} | {r['sum_of_node_times_ms']['median']:.2f}"
            f" | {r['request_wall_ms']['median']:.2f} | {prec} | {nodes}"
            f" | {r['contamination']['other_processes_cores']:.2f} |"
        )
    types = sorted({t["name"] for r in results.values() for t in r["by_node_type"][:8]})
    lines += ["", "## Time by node type (ms per inference)", ""]
    lines += ["| node type | " + " | ".join(results) + " |", "|---" * (len(results) + 1) + "|"]
    for t in types:
        cells = []
        for r in results.values():
            hit = next((x for x in r["by_node_type"] if x["name"] == t), None)
            cells.append("-" if hit is None else f"{hit['ms_per_inference']:.2f}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    for label, r in results.items():
        lines += ["", f"## {label}: top kernels", "", "| kernel | ms | share |", "|---|---|---|"]
        lines += [
            f"| `{k['name']}` | {k['ms_per_inference']:.2f} | {k['share']:.0%} |"
            for k in r["by_kernel"][:8]
        ]
    lines += [
        "",
        "Measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units).",
        "",
    ]
    return "\n".join(lines)


def one(args: argparse.Namespace) -> None:
    """Profile one configuration (run under flock by ``main``)."""
    limit = allowed_other_load(args.threads, os.cpu_count() or 1)
    if quiet_check(3.0) > limit:
        sys.exit(EX_TEMPFAIL)
    template, hint = CONFIGS[args.one]
    image = args.image if args.image.is_file() else sorted(args.image.glob("*.jpg"))[0]
    xml = REPO / template.format(m=args.model)
    result = profile(xml, hint, args.one, image, args.threads, args.runs)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", default="yolo11n,yolo11s")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--image", type=Path, default=REPO / "data/openimages-traffic/images/validation"
    )
    parser.add_argument("--out", type=Path, default=REPO / "bench/results/benchmarks")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("TSD_BENCH_LOCK", "/home/user/work/bench.lock")),
    )
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--model", default="yolo11n", help=argparse.SUPPRESS)
    parser.add_argument("--one", choices=list(CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        one(args)
        return
    if not args.aggregate_only:
        try:
            require_clean(args.allow_dirty)
        except DirtyTreeError as err:
            parser.error(str(err))
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(args.threads),
        "TSD_PARENT_COMMAND": shlex.join([Path(sys.executable).name, *sys.argv]),
        LOCK_ENV: str(args.lock),  # recorded in the result's provenance (bench_lock)
    }
    for model in args.models.split(","):
        results = {}
        for label in CONFIGS:
            path = args.out / "profile" / f"{model}_{label}.json"
            if not args.aggregate_only and not path.exists():
                cmd = [
                    "flock", str(args.lock), sys.executable, __file__, "--one", label,
                    "--model", model, "--threads", str(args.threads), "--runs", str(args.runs),
                    "--image", str(args.image), "--result", str(path),
                ]  # fmt: skip

                def command(_pending: list[Path], cmd: list[str] = cmd) -> list[str]:
                    return cmd

                measure_with_retries(
                    command,
                    [path],
                    lambda r: float(r["contamination"]["other_processes_cores"]),
                    allowed_other_load(args.threads, os.cpu_count() or 1),
                    args.max_attempts,
                    args.retry_wait,
                    env=env,
                    cwd=REPO,
                )
            if path.exists():
                results[label] = json.loads(path.read_text())
        if not results:
            continue
        (args.out / f"profile_{model}.json").write_text(json.dumps(results, indent=2) + "\n")
        (args.out / f"profile_{model}.md").write_text(markdown(model, results))
        for label, r in results.items():
            print(model, label, r["sum_of_node_times_ms"], r["request_wall_ms"])


if __name__ == "__main__":
    main()
