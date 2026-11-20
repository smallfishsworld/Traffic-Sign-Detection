"""Benchmark statistics, timing loop, CPU info parsing, aggregation and reporting."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tsd.benchmark import (
    CpuSample,
    LatencyStats,
    aggregate,
    allowed_other_load,
    config_id,
    cpu_info,
    headline,
    markdown_report,
    other_load,
    pick_images,
    quiet_check,
    run_measurement,
    run_to_run,
    time_calls,
)
from tsd.infer import StageTimes

LSCPU = """Architecture:            x86_64
CPU(s):                  4
Model name:              Intel(R) Xeon(R) Processor @ 2.10GHz
Thread(s) per core:      1
Hypervisor vendor:       KVM
Flags:                   fpu sse avx2 avx512f avx512_vnni amx_tile amx_int8
"""


def test_latency_percentiles() -> None:
    s = LatencyStats.of(list(range(1, 101)))
    assert (s.n, s.min_ms, s.max_ms) == (100, 1.0, 100.0)
    assert s.p50_ms == pytest.approx(50.5)
    assert s.p95_ms == pytest.approx(95.05)
    assert s.p99_ms == pytest.approx(99.01)
    with pytest.raises(ValueError, match="no samples"):
        LatencyStats.of([])


def test_time_calls_warms_up_and_respects_the_time_cap() -> None:
    calls = []

    def fn() -> StageTimes:
        calls.append(1)
        return StageTimes(0.0, 1.5, 0.0)

    timed = time_calls(fn, warmup=3, iterations=5)
    assert len(calls) == 8
    assert len(timed.wall_ms) == 5
    assert timed.infer_ms == [1.5] * 5
    assert timed.first_call_ms >= 0
    assert set(timed.contamination) == {"other_processes_cores", "steal_cores", "own_cores"}
    assert len(time_calls(fn, warmup=1, iterations=50, max_seconds=0.0).wall_ms) == 1
    with pytest.raises(ValueError, match=">= 1"):
        time_calls(fn, warmup=0, iterations=1)


def test_contamination_accounting() -> None:
    a = CpuSample(wall_s=0.0, busy_s=10.0, steal_s=1.0, own_s=5.0)
    b = CpuSample(wall_s=2.0, busy_s=16.0, steal_s=1.2, own_s=9.0)  # 6 s busy, 4 s ours
    assert other_load(a, b) == {"other_processes_cores": 1.0, "steal_cores": 0.1, "own_cores": 2.0}
    assert allowed_other_load(4, 4) == 0.3
    assert allowed_other_load(1, 4) == 1.5
    assert quiet_check(0.05) >= 0.0


def test_cpu_info_and_helpers() -> None:
    info = cpu_info(LSCPU)
    assert info["cpus"] == 4
    assert info["hypervisor"] == "KVM"
    assert info["flags"]["amx_int8"]
    assert not info["flags"]["amx_bf16"]
    paths = [Path(f"{i:02d}.jpg") for i in range(10)]
    assert [p.name for p in pick_images(paths[::-1], 4)] == ["00.jpg", "02.jpg", "05.jpg", "07.jpg"]
    assert len(pick_images(paths, 50)) == 10
    xml = Path("models/yolo11n_openvino_int8/yolo11n.xml")
    assert config_id(xml, "f32", "single", 4, 1) == "yolo11n_openvino_int8_single_t4_b1"
    assert config_id(Path("models/yolo11n.pt"), "f32", "batch", 4, 8) == "yolo11n.pt_batch_t4_b8"
    assert config_id(xml, "bf16", "single", 1, 1) == "yolo11n_openvino_int8_bf16_single_t1_b1"


def _raw(
    model: str, backend: str, mode: str, threads: int, batch: int, p50: float, run: int = 1
) -> dict:
    stats = {
        "p50_ms": p50, "p95_ms": p50 * 1.1, "p99_ms": p50 * 1.2, "mean_ms": p50,
        "min_ms": p50 * 0.9, "max_ms": p50 * 1.5,
    }  # fmt: skip
    return {
        "run": run,
        "model": model,
        "backend": backend,
        "mode": mode,
        "threads": threads,
        "batch_size": batch,
        "model_size_bytes": 10_000_000,
        "e2e": stats,
        "infer": stats,
        "throughput_images_per_s": batch * 1e3 / p50,
        "contamination": {"other_processes_cores": 0.05, "steal_cores": 0.0},
        "contaminated": p50 == 20.0,
    }


def test_aggregate_speedups_accuracy_and_markdown() -> None:
    raw = [
        _raw("models/yolo11n.pt", "pytorch-fp32", "single", 4, 1, 40.0),
        _raw("models/yolo11n.pt", "pytorch-fp32", "batch", 4, 8, 240.0),
        _raw("models/yolo11n_openvino_int8/yolo11n.xml", "openvino-int8", "single", 4, 1, 10.0),
        _raw("models/yolo11n_openvino_int8/yolo11n.xml", "openvino-int8", "batch", 4, 8, 60.0),
        _raw("models/yolo11n.onnx", "onnxruntime-fp32", "single", 4, 1, 20.0),
    ]
    accuracy = {
        "runs": {
            "yolo11n/openvino-int8": {
                "map50": 0.5,
                "map50_95": 0.3,
                "ci95": {"map50": [0.4, 0.6], "map50_95": [0.2, 0.4]},
                "delta_vs_baseline": {
                    "map50": -0.01,
                    "map50_95": -0.005,
                    "ci95": {"map50": [-0.02, 0.0], "map50_95": [-0.01, 0.001]},
                },
            }
        }
    }
    rows = aggregate(raw, accuracy)
    assert rows[0]["backend"] == "pytorch-fp32"
    int8 = next(r for r in rows if r["backend"] == "openvino-int8")
    speed = {k: v["median"] for k, v in int8["speedup_vs_pytorch"].items()}
    assert speed == {"single_t4_b1": 4.0, "batch_t4_b8": 4.0}
    assert int8["accuracy"]["delta_vs_pytorch"]["map50_95"] == -0.005
    (h,) = headline(rows)
    assert (h["best_backend"], h["speedup"]) == ("openvino-int8", 4.0)
    text = markdown_report(rows, cpu_info(LSCPU))
    assert "**4.00x**" in text
    assert "AMX" in text
    assert "-0.0050 [-0.0100, +0.0010]" in text
    assert "20.0 / 22.0 / 24.0 (1 run)*" in text  # the contaminated measurement is flagged


def test_tiled_measurements_get_their_own_table() -> None:
    xml = "models/yolo11n_openvino_fp32/yolo11n.xml"
    tiled = _raw(xml, "openvino-fp32+tiles512", "single", 4, 1, 300.0)
    tiled["tiling"] = {"tile_size": 512, "overlap": 0.2}
    tiled["mean_inferences_per_image"] = 7.5
    raw = [
        _raw("models/yolo11n.pt", "pytorch-fp32", "single", 4, 1, 40.0),
        _raw(xml, "openvino-fp32", "single", 4, 1, 25.0),
        tiled,
    ]
    rows = aggregate(raw, None)
    row = next(r for r in rows if "tiling" in r)
    assert row["slowdown_vs_untiled"]["single_t4_b1"]["median"] == 12.0
    assert "speedup_vs_pytorch" not in row
    (h,) = headline(rows)
    assert h["best_backend"] == "openvino-fp32"  # a tiled run is never the headline
    text = markdown_report(rows, cpu_info(LSCPU))
    assert (
        "| yolo11n | openvino-fp32 | 512 / 20% | 7.5 | 300.0 / 330.0 / 360.0 (1 run) | 25.0"
        " | 12.0x |" in text
    )
    assert text.count("openvino-fp32+tiles512") == 0


def test_repeated_runs_give_medians_and_paired_speedups() -> None:
    pt, ov = "models/yolo11n.pt", "models/yolo11n_openvino_int8/yolo11n.xml"
    raw = [
        _raw(pt, "pytorch-fp32", "single", 4, 1, p50, run=k)
        for k, p50 in enumerate([40.0, 60.0, 44.0], 1)
    ] + [
        _raw(ov, "openvino-int8", "single", 4, 1, p50, run=k)
        for k, p50 in enumerate([10.0, 20.0, 22.0], 1)
    ]
    rows = aggregate(raw, None)
    int8 = next(r for r in rows if r["backend"] == "openvino-int8")
    slot = int8["single_t4_b1"]
    assert slot["runs"] == 3
    assert slot["e2e"]["p50_ms"] == 20.0
    assert slot["e2e_p50_range_ms"] == [10.0, 22.0]
    speed = int8["speedup_vs_pytorch"]["single_t4_b1"]
    assert speed["per_run"] == pytest.approx([4.0, 3.0, 2.0])  # run k against PyTorch run k
    assert speed["median"] == pytest.approx(3.0)
    assert speed["range"] == pytest.approx([2.0, 4.0])
    assert speed["ratio_of_medians"] == pytest.approx(44.0 / 20.0)
    (h,) = headline(rows)
    assert (h["speedup"], h["runs"]) == (pytest.approx(3.0), 3)
    assert "3.00x (2.00-4.00)" in markdown_report(rows, cpu_info(LSCPU))
    with pytest.raises(ValueError, match="duplicate result"):
        aggregate([*raw, _raw(ov, "openvino-int8", "single", 4, 1, 9.0, run=2)], None)


@pytest.mark.weights
@pytest.mark.slow
def test_run_measurement_smoke(
    weights_path: Path, sample_images: list[np.ndarray], tmp_path: Path
) -> None:
    paths = []
    for i, image in enumerate(sample_images):
        paths.append(tmp_path / f"{i}.jpg")
        cv2.imwrite(str(paths[-1]), image)
    result = run_measurement(weights_path, paths, threads=1, warmup=1, iterations=3)
    assert result["backend"] == "pytorch-fp32"
    assert result["iterations"] == 3
    assert result["e2e"]["p50_ms"] >= result["infer"]["p50_ms"] > 0
    assert result["model_size_bytes"] == weights_path.stat().st_size
    settings = result["provenance"]["runtime_settings"]
    assert settings["torch_num_threads"] == 1
    assert result["provenance"]["git"]["commit"]
    batch = run_measurement(
        weights_path, paths, threads=1, mode="batch", batch_size=2, warmup=1, iterations=2
    )
    assert batch["throughput_images_per_s"] > 0


def test_run_to_run_spread_and_round_winners() -> None:
    xml = "models/yolo11n_openvino_int8/yolo11n.xml"
    torch_p50 = enumerate((40.0, 44.0, 50.0), 1)
    int8_p50 = enumerate((10.0, 10.0, 60.0), 1)
    raw = [
        *(_raw("models/yolo11n.pt", "pytorch-fp32", "single", 4, 1, p, k) for k, p in torch_p50),
        *(_raw(xml, "openvino-int8", "single", 4, 1, p, k) for k, p in int8_p50),
        *(_raw(xml, "openvino-int8-nopin", "single", 4, 1, 1.0, k) for k in (1, 2, 3)),
    ]
    summary = run_to_run(aggregate(raw, None))
    assert summary["configurations"] == 3
    assert summary["spread_above_10pct"] == 2
    assert summary["largest"][0]["spread"] == pytest.approx(5.0)  # (60 - 10) / 10
    assert summary["spread_median"] == pytest.approx(10 / 44)
    # -nopin is not a candidate; run 3 is won by PyTorch (50 < 60)
    assert [w["fastest"] for w in summary["fastest_per_round"]] == [
        "openvino-int8", "openvino-int8", "pytorch-fp32",
    ]  # fmt: skip
    assert summary["fastest_counts"] == {"openvino-int8": 2, "pytorch-fp32": 1}
