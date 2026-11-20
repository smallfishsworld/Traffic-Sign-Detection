"""Measure one configuration (called by run_benchmarks.py under flock; usable on its own).

Exits with status 75 (EX_TEMPFAIL) without measuring if other processes keep the CPUs busy
during a short pre-check, so the caller can release the lock and retry later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tsd.benchmark import (
    EX_TEMPFAIL,
    allowed_other_load,
    pick_images,
    quiet_check,
    run_measurement,
)
from tsd.infer.tiling import TilingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision-hint", choices=["f32", "bf16"], default="f32")
    parser.add_argument("--images-dir", type=Path, default=Path("data/openimages-traffic/images"))
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet-window", type=float, default=3.0, help="0 skips the pre-check")
    parser.add_argument("--tiles", type=int, default=0, help="tile size for sliced inference")
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--run", type=int, default=1, help="index of a repeated run (recorded)")
    parser.add_argument(
        "--no-cpu-pinning", action="store_true", help="turn off OpenVINO's thread pinning"
    )
    args = parser.parse_args()
    if args.quiet_window > 0:
        busy = quiet_check(args.quiet_window)
        limit = allowed_other_load(args.threads, os.cpu_count() or 1)
        if busy > limit:
            print(f"busy: other processes use {busy:.2f} cores (limit {limit:.2f})")
            sys.exit(EX_TEMPFAIL)
    images = pick_images(list(args.images_dir.rglob("*.jpg")), args.num_images)
    result = run_measurement(
        args.model,
        images,
        threads=args.threads,
        mode=args.mode,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        openvino_precision=args.precision_hint,
        max_seconds=args.max_seconds,
        tiling=TilingConfig(args.tiles, args.tile_overlap) if args.tiles else None,
        openvino_cpu_pinning=False if args.no_cpu_pinning else None,
    )
    result["precision_hint"] = args.precision_hint
    result["run"] = args.run
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    e2e = result["e2e"]
    print(
        f"{result['backend']} t={args.threads} b={args.batch_size}: p50 {e2e['p50_ms']:.2f} ms,"
        f" p95 {e2e['p95_ms']:.2f}, p99 {e2e['p99_ms']:.2f},"
        f" {result['throughput_images_per_s']:.1f} img/s,"
        f" other load {result['contamination']['other_processes_cores']:.2f} cores"
    )


if __name__ == "__main__":
    main()
