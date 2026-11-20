"""Latency and throughput measurement for one model configuration.

``bench/run_benchmarks.py`` runs every configuration in a fresh process (the thread pools of
PyTorch, ONNX Runtime, OpenVINO and OpenCV are sized once per process) wrapped in
``flock /home/user/work/bench.lock``, and this module does the measuring inside that process.

What is timed:

* **end-to-end** (``e2e``): letterbox + tensor conversion + inference + NMS + rescaling of the
  boxes, on images already decoded in memory (JPEG decoding is excluded so that the numbers
  compare the inference stacks, not the image codec);
* **inference only** (``infer``): the backend call alone, from the same calls.

Each measurement starts with untimed warm-up calls (the first one, which includes lazy
initialization, is reported separately as ``first_call_ms``). Percentiles use NumPy's default
linear interpolation.

The machine is shared, and the lock only serializes *timing* runs. So every measurement also
records **contamination**: the CPU time that *other* processes used during the timed loop
(machine-wide busy time from ``/proc/stat`` minus this process's own CPU time), in cores, and
the hypervisor's steal time. ``bench/run_benchmarks.py`` repeats a measurement whose
contamination exceeds :func:`allowed_other_load` and keeps every attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tsd.infer import StageTimes, load_detector
from tsd.infer.tiling import TiledDetector, TilingConfig
from tsd.io import Image, load_image
from tsd.provenance import provenance

Mode = Literal["single", "batch"]
INTERESTING_CPU_FLAGS = (
    "avx2",
    "avx512f",
    "avx512_vnni",
    "avx_vnni",
    "avx512_bf16",
    "amx_tile",
    "amx_int8",
    "amx_bf16",
)


@dataclass(frozen=True)
class LatencyStats:
    n: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def of(cls, samples_ms: Sequence[float]) -> LatencyStats:
        a = np.asarray(samples_ms, dtype=np.float64)
        if a.size == 0:
            raise ValueError("no samples")
        p50, p95, p99 = (float(v) for v in np.percentile(a, [50, 95, 99]))
        return cls(
            n=int(a.size),
            mean_ms=float(a.mean()),
            std_ms=float(a.std(ddof=1)) if a.size > 1 else 0.0,
            min_ms=float(a.min()),
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            max_ms=float(a.max()),
        )


@dataclass(frozen=True)
class TimedCalls:
    wall_ms: list[float]
    infer_ms: list[float]
    first_call_ms: float
    contamination: dict[str, float]


def time_calls(
    fn: Callable[[], StageTimes], warmup: int, iterations: int, max_seconds: float = 600.0
) -> TimedCalls:
    """Run ``fn`` ``warmup`` times untimed, then up to ``iterations`` times (stopping early after
    ``max_seconds``), recording per-call wall-clock and inference-only milliseconds, the first
    warm-up call's time, and the other processes' CPU use during the timed calls."""
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be >= 1")
    first_ms = 0.0
    for i in range(warmup):
        t0 = time.perf_counter()
        fn()
        if i == 0:
            first_ms = (time.perf_counter() - t0) * 1e3
    wall: list[float] = []
    infer: list[float] = []
    before = cpu_sample()
    deadline = time.monotonic() + max_seconds
    for _ in range(iterations):
        t0 = time.perf_counter()
        stages = fn()
        wall.append((time.perf_counter() - t0) * 1e3)
        infer.append(stages.infer_ms)
        if time.monotonic() > deadline:
            break
    return TimedCalls(wall, infer, first_ms, other_load(before, cpu_sample()))


@dataclass(frozen=True)
class CpuSample:
    wall_s: float
    busy_s: float  # non-idle CPU time of the whole machine (all CPUs)
    steal_s: float
    own_s: float  # this process's user + system time (all threads)


def cpu_sample() -> CpuSample:
    with Path("/proc/stat").open() as f:
        values = [int(v) for v in f.readline().split()[1:9]]
    user, nice, system, _idle, _iowait, irq, softirq, steal = values
    tick = os.sysconf("SC_CLK_TCK")
    own = os.times()
    return CpuSample(
        wall_s=time.monotonic(),
        busy_s=(user + nice + system + irq + softirq + steal) / tick,
        steal_s=steal / tick,
        own_s=own.user + own.system,
    )


def other_load(before: CpuSample, after: CpuSample) -> dict[str, float]:
    """Average number of cores used by other processes (and stolen) between two samples."""
    wall = max(after.wall_s - before.wall_s, 1e-9)
    other = (after.busy_s - before.busy_s) - (after.own_s - before.own_s)
    return {
        "other_processes_cores": round(max(other, 0.0) / wall, 3),
        "steal_cores": round((after.steal_s - before.steal_s) / wall, 3),
        "own_cores": round((after.own_s - before.own_s) / wall, 3),
    }


def allowed_other_load(threads: int, cpus: int) -> float:
    """Contamination accepted for a measurement with ``threads`` threads on ``cpus`` CPUs: half
    of the CPUs the measurement leaves free, but at least 0.3 cores."""
    return max(0.3, (cpus - threads) / 2)


def quiet_check(window_s: float = 3.0) -> float:
    """Cores used by other processes over a short idle window."""
    before = cpu_sample()
    time.sleep(window_s)
    return other_load(before, cpu_sample())["other_processes_cores"]


def _model_files(model: Path) -> list[Path]:
    return [model, model.with_suffix(".bin")] if model.suffix == ".xml" else [model]


def model_size_bytes(model: Path) -> int:
    return sum(f.stat().st_size for f in _model_files(model))


def model_sha256(model: Path) -> dict[str, str]:
    """SHA-256 of each file of ``model`` (an OpenVINO IR is an .xml plus a .bin)."""
    out = {}
    for f in _model_files(model):
        digest = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        out[f.name] = digest.hexdigest()
    return out


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("numpy", "opencv-python-headless", "torch", "onnxruntime", "openvino", "tsd"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def cpu_info(lscpu_text: str | None = None) -> dict[str, Any]:
    """Model name, logical CPU count and the SIMD/matrix flags that matter for INT8/BF16."""
    if lscpu_text is None:
        lscpu_text = subprocess.run(
            ["lscpu"], capture_output=True, text=True, check=True, env={"LC_ALL": "C"}
        ).stdout
    pairs = (line.split(":", 1) for line in lscpu_text.splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in pairs}
    flags = set(fields.get("Flags", "").split())
    return {
        "model_name": fields.get("Model name", ""),
        "cpus": int(fields.get("CPU(s)", "0") or 0),
        "threads_per_core": int(fields.get("Thread(s) per core", "0") or 0),
        "hypervisor": fields.get("Hypervisor vendor", ""),
        "flags": {f: f in flags for f in INTERESTING_CPU_FLAGS},
    }


def pick_images(paths: Sequence[Path], count: int) -> list[Path]:
    """A deterministic, evenly spaced subset of ``paths`` (sorted by name)."""
    ordered = sorted(paths, key=lambda p: p.name)
    if count >= len(ordered):
        return ordered
    step = len(ordered) / count
    return [ordered[int(i * step)] for i in range(count)]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def run_measurement(
    model: Path,
    image_paths: Sequence[Path],
    threads: int,
    mode: Mode = "single",
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 200,
    openvino_precision: Literal["f32", "bf16"] = "f32",
    max_seconds: float = 600.0,
    tiling: TilingConfig | None = None,
    openvino_cpu_pinning: bool | None = None,
) -> dict[str, Any]:
    """Measure one configuration in the current process (see the module docstring). With
    ``tiling``, one call is the whole sliced pipeline for one image (tiles + full image, merged),
    and the ``infer`` statistics equal the end-to-end ones."""
    import cv2

    cv2.setNumThreads(threads)
    images: list[Image] = [load_image(p) for p in image_paths]
    detector = load_detector(
        model,
        threads=threads,
        openvino_precision=openvino_precision,
        openvino_cpu_pinning=openvino_cpu_pinning,
    )
    if mode == "single" and batch_size != 1:
        raise ValueError("single-image mode uses batch size 1")
    if tiling is not None and mode != "single":
        raise ValueError("tiled inference is measured one image at a time")
    cursor = 0
    tiled = TiledDetector(detector, tiling) if tiling is not None else None

    def call() -> StageTimes:
        nonlocal cursor
        batch = [images[(cursor + k) % len(images)] for k in range(batch_size)]
        cursor += batch_size
        if tiled is not None:
            t0 = time.perf_counter()
            tiled.predict_one(batch[0])
            return StageTimes(0.0, (time.perf_counter() - t0) * 1e3, 0.0)
        return detector.predict_timed(batch)[1]

    load_before = os.getloadavg()
    started = time.time()
    timed = time_calls(call, warmup, iterations, max_seconds)
    load_after = os.getloadavg()
    e2e = LatencyStats.of(timed.wall_ms)
    result: dict[str, Any] = {
        "model": str(model),
        "backend": detector.name if tiled is None else tiled.name,
        "threads": threads,
        "mode": mode,
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": len(timed.wall_ms),
        "images": [p.name for p in image_paths],
        "model_size_bytes": model_size_bytes(model),
        "model_sha256": model_sha256(model),
        "first_call_ms": timed.first_call_ms,
        "e2e": asdict(e2e),
        "infer": asdict(LatencyStats.of(timed.infer_ms)),
        # every timed call, so the per-run percentiles can be recomputed (rounded to 1 us)
        "samples_ms": {
            "e2e": [round(v, 3) for v in timed.wall_ms],
            "infer": [round(v, 3) for v in timed.infer_ms],
        },
        "throughput_images_per_s": batch_size * 1e3 / e2e.p50_ms,
        "started_unix": round(started, 1),
        "seconds": round(time.time() - started, 1),
        "loadavg_before": [round(v, 2) for v in load_before],
        "loadavg_after": [round(v, 2) for v in load_after],
        "contamination": timed.contamination,
        "cpus": os.cpu_count(),
        "versions": package_versions(),
        "nms": {"conf_threshold": detector.nms.conf_threshold, "iou": detector.nms.iou_threshold},
        "input_size": list(detector.input_size),
        "letterbox": "fixed square, auto=False (identical for every runtime)",
        "opencv_threads": cv2.getNumThreads(),
        "provenance": provenance(detector.runtime_settings()),
    }
    if tiled is not None:
        result["tiling"] = asdict(tiled.config)
        result["mean_inferences_per_image"] = float(
            np.mean([len(tiled.tiles(im)) + int(tiled.config.full_image) for im in images])
        )
    return result


def config_id(
    model: Path, precision_hint: str, mode: str, threads: int, batch_size: int, tiles: int = 0
) -> str:
    stem = model.parent.name if model.suffix == ".xml" else model.name
    hint = "_bf16" if precision_hint == "bf16" else ""
    tiled = f"_tiles{tiles}" if tiles else ""
    return _slug(f"{stem}{hint}{tiled}_{mode}_t{threads}_b{batch_size}")


SPEEDUP_CAVEAT = (
    "Speedups were measured on an Intel Xeon with AVX-512 VNNI and AMX (INT8/BF16 matrix units); "
    "INT8 and BF16 gains on a typical laptop CPU without AMX are smaller."
)


BACKEND_LEGEND = (
    "Backends: `openvino-fp32` = FP32 IR, f32 precision hint; `openvino-bf16` = the same FP32 IR"
    " with the bf16 hint (OpenVINO's default on this CPU); `openvino-int8` = INT8 IR (NNCF), its"
    " non-quantized ops in FP32 (f32 hint); `openvino-int8-bf16` = the same INT8 IR, non-quantized"
    " ops in BF16 (bf16 hint, OpenVINO's default on this CPU); `-nopin` = OpenVINO thread pinning"
    " off (PyTorch and ONNX Runtime threads are never pinned)."
)

EX_TEMPFAIL = 75


def next_free(path: Path, tag: str) -> Path:
    """``<stem>.<tag><k>.json`` with the smallest ``k >= 1`` that does not exist yet."""
    k = 1
    while (candidate := path.with_name(f"{path.stem}.{tag}{k}{path.suffix}")).exists():
        k += 1
    return candidate


def measure_with_retries(
    command_for: Callable[[list[Path]], list[str]],
    outputs: Sequence[Path],
    load_of: Callable[[dict[str, Any]], float],
    limit: float,
    attempts: int,
    wait: float,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Run ``command_for(pending)`` (a measurement under ``flock`` that writes the ``pending``
    result files) until every output has a result whose other-process load is at most
    ``limit``. Exit status 75 means "machine busy, nothing measured": wait and retry. A
    contaminated result is kept as ``<stem>.attempt<k>.json`` (next free ``k``, never
    overwritten) and re-measured; after ``attempts`` tries (busy ones included) the least
    contaminated kept attempt is used with ``"contaminated": true``. Each result records
    ``attempt`` (the how-manyth session of this call that measured it, so ``attempt`` k means
    k - 1 contaminated attempt files from this call) and ``busy_skips`` (sessions of this call
    that exited busy before it). Raises ``SystemExit`` if an output ends without any result.
    Returns the outputs that had to fall back to a contaminated attempt."""
    pending = list(outputs)
    measured = busy = 0
    for _ in range(attempts):
        if not pending:
            break
        code = subprocess.run(command_for(pending), env=env, cwd=cwd, check=False).returncode
        if code == EX_TEMPFAIL:
            busy += 1
            sleep(wait)
            continue
        if code != 0:
            raise SystemExit(f"measurement failed with exit status {code}")
        measured += 1
        still = []
        for out in pending:
            if not out.exists():
                still.append(out)
                continue
            result = json.loads(out.read_text())
            result["attempt"] = measured
            result["busy_skips"] = busy
            if load_of(result) <= limit:
                result["contaminated"] = False
                out.write_text(json.dumps(result, indent=2) + "\n")
            else:
                result["contaminated"] = True
                next_free(out, "attempt").write_text(json.dumps(result, indent=2) + "\n")
                out.unlink()
                still.append(out)
        pending = still
        if pending:
            sleep(wait)
    fallback, missing = [], []
    for out in pending:
        kept = sorted(out.parent.glob(f"{out.stem}.attempt*{out.suffix}"))
        if not kept:
            missing.append(out)
            continue
        best = min(kept, key=lambda q: load_of(json.loads(q.read_text())))
        result = json.loads(best.read_text())
        result["contaminated"] = True
        result["fallback_from"] = best.name
        out.write_text(json.dumps(result, indent=2) + "\n")
        fallback.append(out)
    if missing:
        raise SystemExit("no result for: " + ", ".join(m.name for m in missing))
    return fallback


def family_of(model: str) -> str:
    """``models/yolo11n_openvino_int8/yolo11n.xml`` -> ``yolo11n``."""
    return Path(model).stem


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _summarize_runs(runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Median over repeated runs of one configuration, with the run-to-run range."""
    ordered = [runs[k] for k in sorted(runs)]

    def med(part: str) -> dict[str, float]:
        keys = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms")
        return {k: _median([r[part][k] for r in ordered]) for k in keys}

    p50 = [r["e2e"]["p50_ms"] for r in ordered]
    thr = [r["throughput_images_per_s"] for r in ordered]
    loads = [r.get("contamination", {}).get("other_processes_cores") for r in ordered]
    return {
        "runs": len(ordered),
        "e2e": med("e2e"),
        "infer": med("infer"),
        "e2e_p50_range_ms": [min(p50), max(p50)],
        "throughput_images_per_s": _median(thr),  # batch size / p50 call time
        "throughput_range": [min(thr), max(thr)],
        # batch size / mean call time: counts the slow tail, so usually below the p50 figure
        "throughput_mean_images_per_s": _median(
            [r["batch_size"] * 1e3 / r["e2e"]["mean_ms"] for r in ordered]
        ),
        "other_load_cores": max((v for v in loads if v is not None), default=None),
        "contaminated": any(bool(r.get("contaminated", False)) for r in ordered),
        "per_run": [
            {
                "run": k,
                "p50_ms": runs[k]["e2e"]["p50_ms"],
                "p95_ms": runs[k]["e2e"]["p95_ms"],
                "p99_ms": runs[k]["e2e"]["p99_ms"],
                "throughput_images_per_s": runs[k]["throughput_images_per_s"],
                "other_load_cores": runs[k].get("contamination", {}).get("other_processes_cores"),
                "contaminated": bool(runs[k].get("contaminated", False)),
                "attempt": runs[k].get("attempt", 1),
                "started_unix": runs[k].get("started_unix"),
            }
            for k in sorted(runs)
        ],
    }


def paired_ratio(
    row: dict[str, Any], base: dict[str, Any], slot: str, invert: bool = False
) -> dict[str, Any] | None:
    """Per-run ratio of ``row`` to ``base`` throughput in ``slot`` (``invert``: of p50 latency),
    paired by run index (the runs of one round are measured close together in time); the median,
    the range and, for comparison, the ratio of the two medians."""
    a, b = row["_runs"].get(slot, {}), base["_runs"].get(slot, {})
    common = sorted(set(a) & set(b))
    if not common:
        return None

    def ratio(x: dict[str, Any], y: dict[str, Any]) -> float:
        if invert:
            return float(x["e2e"]["p50_ms"] / y["e2e"]["p50_ms"])
        return float(x["throughput_images_per_s"] / y["throughput_images_per_s"])

    per_run = [ratio(a[k], b[k]) for k in common]
    medians = ratio(row[slot], base[slot])
    return {
        "median": _median(per_run),
        "range": [min(per_run), max(per_run)],
        "runs": len(per_run),
        "per_run": per_run,
        "ratio_of_medians": medians,
    }


def aggregate(
    raw: Sequence[dict[str, Any]], accuracy: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """One row per (model family, backend) with 1-thread, 4-thread and batch-8 numbers (each the
    median over repeated runs, with the range), the paired speedup against ``pytorch-fp32`` of
    the same family, and accuracy from a ``tsd eval`` report (``runs`` keyed
    ``<family>/<backend>``). Two results for the same family, backend, setting and run index are
    an error (they would silently replace each other)."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for r in raw:
        key = (family_of(r["model"]), r["backend"])
        row = rows.setdefault(
            key,
            {
                "family": key[0],
                "backend": key[1],
                "model_size_bytes": r["model_size_bytes"],
                "precision_hint": r.get("precision_hint"),
                "_runs": {},
            },
        )
        if r.get("tiling") is not None:  # sliced inference: compared with its untiled backend
            row["tiling"] = r["tiling"]
            row["untiled_backend"] = r["backend"].split("+", 1)[0]
            row["mean_inferences_per_image"] = r.get("mean_inferences_per_image")
        slot = f"{r['mode']}_t{r['threads']}_b{r['batch_size']}"
        run = int(r.get("run", 1))
        runs = row["_runs"].setdefault(slot, {})
        if run in runs:
            raise ValueError(f"duplicate result: {key[0]} {key[1]} {slot} run {run}")
        runs[run] = r
    for row in rows.values():
        for slot, runs in row["_runs"].items():
            row[slot] = _summarize_runs(runs)
    accuracy_runs = (accuracy or {}).get("runs", {})
    for (family, backend), row in rows.items():
        if "tiling" in row:
            untiled = rows.get((family, row["untiled_backend"]))
            if untiled is not None:
                slow = {
                    slot: paired_ratio(row, untiled, slot, invert=True)
                    for slot in row["_runs"]
                    if slot.startswith("single_") and slot in untiled
                }
                row["slowdown_vs_untiled"] = {k: v for k, v in slow.items() if v is not None}
            continue
        base = rows.get((family, "pytorch-fp32"))
        if base is not None:
            speed = {
                slot: paired_ratio(row, base, slot)
                for slot in row["_runs"]
                if slot.startswith(("single_", "batch_")) and slot in base
            }
            row["speedup_vs_pytorch"] = {k: v for k, v in speed.items() if v is not None}
        acc = accuracy_runs.get(f"{family}/{backend}")
        if acc is not None:
            row["accuracy"] = {
                "map50": acc["map50"],
                "map50_95": acc["map50_95"],
                "ci95": acc["ci95"],
                "delta_vs_pytorch": acc.get("delta_vs_baseline"),
            }
    out = []
    for row in rows.values():
        row.pop("_runs")
        out.append(row)
    return sorted(out, key=lambda r: (r["family"], r["backend"] != "pytorch-fp32", r["backend"]))


def run_to_run(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How much repeated runs of one configuration differ, and which runtime won each round.

    The spread of a configuration (row and setting with at least two runs) is (max - min) /
    median of its per-run p50. ``fastest_per_round`` compares, per family, setting and run index,
    the pinned untiled backends (no ``-nopin``, no ``+tiles``) by p50."""
    spreads = []
    rounds: dict[tuple[str, str, int], list[tuple[float, str]]] = {}
    for row in rows:
        for slot, summary in row.items():
            if not isinstance(summary, dict) or "per_run" not in summary:
                continue
            p50 = [r["p50_ms"] for r in summary["per_run"]]
            if len(p50) >= 2:
                spreads.append(
                    {
                        "family": row["family"],
                        "backend": row["backend"],
                        "setting": slot,
                        "p50_ms": [round(v, 1) for v in p50],
                        "spread": (max(p50) - min(p50)) / _median(p50),
                    }
                )
            if "-nopin" in row["backend"] or "+" in row["backend"]:
                continue
            for r in summary["per_run"]:
                key = (row["family"], slot, int(r["run"]))
                rounds.setdefault(key, []).append((r["p50_ms"], row["backend"]))
    values = [s["spread"] for s in spreads]
    winners = [
        {"family": f, "setting": slot, "run": run, "fastest": min(c)[1]}
        for (f, slot, run), c in sorted(rounds.items())
    ]
    wins: dict[str, int] = {}
    for (_f, _slot, _run), c in sorted(rounds.items()):
        wins[min(c)[1]] = wins.get(min(c)[1], 0) + 1
    return {
        "configurations": len(spreads),
        "spread_median": _median(values) if values else None,
        "spread_above_10pct": sum(v > 0.10 for v in values),
        "largest": sorted(spreads, key=lambda s: -s["spread"])[:5],
        "fastest_per_round": winners,
        "fastest_counts": dict(sorted(wins.items(), key=lambda kv: -kv[1])),
    }


def headline(rows: Sequence[dict[str, Any]], slot: str = "single_t4_b1") -> list[dict[str, Any]]:
    """Per family: PyTorch FP32 against the fastest other backend in ``slot`` (by median
    throughput); the speedup is the median of the per-run paired speedups, with their range."""
    out = []
    for family in sorted({r["family"] for r in rows}):
        candidates = [
            r
            for r in rows
            if r["family"] == family
            and r["backend"] != "pytorch-fp32"
            and slot in r
            and slot in r.get("speedup_vs_pytorch", {})
            and "tiling" not in r
        ]
        base = next(
            (r for r in rows if r["family"] == family and r["backend"] == "pytorch-fp32"), None
        )
        if base is None or slot not in base or not candidates:
            continue
        best = max(candidates, key=lambda r: r[slot]["throughput_images_per_s"])
        acc = best.get("accuracy") or {}
        speed = best["speedup_vs_pytorch"][slot]
        out.append(
            {
                "family": family,
                "slot": slot,
                "runs": speed["runs"],
                "baseline_p50_ms": base[slot]["e2e"]["p50_ms"],
                "best_backend": best["backend"],
                "best_p50_ms": best[slot]["e2e"]["p50_ms"],
                "speedup": speed["median"],
                "speedup_range": speed["range"],
                "speedup_ratio_of_medians": speed["ratio_of_medians"],
                "delta_map50_95": (acc.get("delta_vs_pytorch") or {}).get("map50_95"),
                "delta_map50_95_ci95": (acc.get("delta_vs_pytorch") or {})
                .get("ci95", {})
                .get("map50_95"),
            }
        )
    return out


def markdown_report(rows: Sequence[dict[str, Any]], cpu: dict[str, Any], note: str = "") -> str:
    def lat(row: dict[str, Any], slot: str) -> str:
        if slot not in row:
            return "-"
        e = row[slot]["e2e"]
        flag = "*" if row[slot].get("contaminated") else ""
        lo, hi = row[slot]["e2e_p50_range_ms"]
        spread = f" ({lo:.1f}-{hi:.1f})" if row[slot]["runs"] > 1 else " (1 run)"
        return f"{e['p50_ms']:.1f} / {e['p95_ms']:.1f} / {e['p99_ms']:.1f}{spread}{flag}"

    def speed(row: dict[str, Any], slot: str) -> str:
        value = row.get("speedup_vs_pytorch", {}).get(slot)
        if value is None:
            return "-"
        lo, hi = value["range"]
        return f"{value['median']:.2f}x ({lo:.2f}-{hi:.2f})"

    def acc(row: dict[str, Any]) -> str:
        a = row.get("accuracy")
        if not a:
            return "-"
        delta = a.get("delta_vs_pytorch")
        text = f"{a['map50_95']:.3f}"
        if delta:
            lo, hi = delta["ci95"]["map50_95"]
            text += f" ({delta['map50_95']:+.4f} [{lo:+.4f}, {hi:+.4f}])"
        return text

    flags = ", ".join(f for f, on in cpu.get("flags", {}).items() if on)
    lines = [
        "# CPU inference benchmarks",
        "",
        f"CPU: {cpu.get('model_name', '?')}, {cpu.get('cpus', '?')} vCPUs"
        f" ({cpu.get('hypervisor') or 'bare metal'}); flags: {flags}.",
        "",
        f"> {SPEEDUP_CAVEAT}",
        "",
    ]
    if note:
        lines += [note, ""]
    lines += [
        "Latency = end-to-end per image (fixed 640x640 letterbox for every runtime + inference +"
        " NMS, decoded image in memory), p50 / p95 / p99 in ms: the median over the repeated runs,"
        " in brackets the range of the per-run p50. Batch-8 = images/s for batches of 8 at 4"
        " threads (from the median batch time). Speedups are against PyTorch FP32 at the same"
        " setting: the median of the per-run paired ratios (run k of a configuration against"
        " run k of PyTorch, measured in the same round), in brackets their range."
        " mAP50-95: Open Images stop sign + traffic light; in brackets the paired difference to"
        " PyTorch FP32 with its 95% bootstrap interval. Every measurement records the CPU use of"
        " other processes on this shared machine (benchmarks.json, `other_load_cores`) and was"
        " repeated until it stayed below the limit; `*` marks one that never did.",
        "",
        "| model | backend | size (MB) | 1 thread | 4 threads | batch-8 img/s"
        " | speedup 1t / 4t / b8 | mAP50-95 (Δ vs PyTorch [95% CI]) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    tiled = [r for r in rows if "tiling" in r]
    for r in rows:
        if "tiling" in r:
            continue
        b8 = r.get("batch_t4_b8")
        b8_text = "-"
        if b8 is not None:
            lo, hi = b8["throughput_range"]
            b8_text = f"{b8['throughput_images_per_s']:.1f} ({lo:.1f}-{hi:.1f})"
            b8_text += "*" if b8.get("contaminated") else ""
        speeds = " / ".join(speed(r, s) for s in ("single_t1_b1", "single_t4_b1", "batch_t4_b8"))
        lines.append(
            f"| {r['family']} | {r['backend']} | {r['model_size_bytes'] / 1e6:.1f}"
            f" | {lat(r, 'single_t1_b1')} | {lat(r, 'single_t4_b1')} | {b8_text}"
            f" | {speeds} | {acc(r)} |"
        )
    lines += ["", "## Headline (single image, 4 threads)", ""]
    for h in headline(rows):
        delta = h["delta_map50_95"]
        ci = h["delta_map50_95_ci95"]
        acc_text = (
            "accuracy not evaluated"
            if delta is None
            else f"mAP50-95 {delta:+.4f} (95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}])"
        )
        lo, hi = h["speedup_range"]
        lines.append(
            f"- {h['family']}: {h['best_backend']} {h['best_p50_ms']:.1f} ms vs PyTorch FP32"
            f" {h['baseline_p50_ms']:.1f} ms median p50 (ratio of the medians"
            f" {h['speedup_ratio_of_medians']:.2f}x); speedup **{h['speedup']:.2f}x**, the median"
            f" of {h['runs']} paired per-round ratios (range {lo:.2f}-{hi:.2f}x; {acc_text})."
            " AMX/VNNI CPU: INT8/BF16 speedups on a typical laptop are smaller."
        )
    if tiled:
        lines += [
            "",
            "## Sliced inference: latency cost (single image, 4 threads)",
            "",
            "One call = all tiles of one image plus the full-image pass, merged; the inference"
            " count is the mean over the benchmark images (it grows with the image size).",
            "",
            "| model | backend | tile (px) / overlap | inferences per image | p50 / p95 / p99 ms"
            " | untiled p50 ms | slowdown |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in tiled:
            untiled = next(
                (
                    u
                    for u in rows
                    if u["family"] == r["family"] and u["backend"] == r["untiled_backend"]
                ),
                {},
            )
            base = untiled.get("single_t4_b1", {}).get("e2e", {}).get("p50_ms")
            slow = r.get("slowdown_vs_untiled", {}).get("single_t4_b1", {}).get("median")
            n = r.get("mean_inferences_per_image")
            lines.append(
                f"| {r['family']} | {r['untiled_backend']} | {r['tiling']['tile_size']} /"
                f" {r['tiling']['overlap']:.0%} | {'-' if n is None else f'{n:.1f}'}"
                f" | {lat(r, 'single_t4_b1')} | {'-' if base is None else f'{base:.1f}'}"
                f" | {'-' if slow is None else f'{slow:.1f}x'} |"
            )
    return "\n".join(lines) + "\n"
