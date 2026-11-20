"""End-to-end smoke test: export yolo11n to ONNX and OpenVINO, run both on two real photos and
compare with PyTorch. Needs only the cached weights file (skipped with a reason otherwise)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsd.infer import ModelMeta, load_detector
from tsd.infer.backends import OnnxRuntimeBackend, OpenVINOBackend, TorchBackend
from tsd.infer.export import compare_raw_outputs, export_onnx, export_openvino

pytestmark = [pytest.mark.weights, pytest.mark.slow]


@pytest.fixture(scope="module")
def exported(weights_path: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("export")
    onnx_path = export_onnx(weights_path, out / "yolo11n.onnx")
    return onnx_path, export_openvino(onnx_path, out / "openvino")


def test_exported_models_match_pytorch(
    weights_path: Path, exported: tuple[Path, Path], sample_images: list[np.ndarray]
) -> None:
    onnx_path, xml = exported
    reference = TorchBackend(weights_path, threads=2)
    for candidate in (OnnxRuntimeBackend(onnx_path, threads=2), OpenVINOBackend(xml, threads=2)):
        report = compare_raw_outputs(reference, candidate, sample_images)
        assert report.reference_detections >= 5  # people and a bus
        assert report.matched_detections == report.reference_detections
        assert report.max_abs_box_px < 0.1
        assert report.max_abs_score < 1e-4
        assert report.unmatched_candidates == 0
        assert report.candidate_detections == report.reference_detections
        assert report.failures() == []


def test_dynamic_batch_and_metadata(
    weights_path: Path, exported: tuple[Path, Path], sample_images: list[np.ndarray]
) -> None:
    onnx_path, xml = exported
    meta = ModelMeta.load(onnx_path)
    assert meta.names[0] == "person"
    assert meta.names[11] == "stop sign"
    assert meta.source == weights_path.name
    assert ModelMeta.load(xml).extra["format"] == "openvino"
    detector = load_detector(onnx_path, threads=2)
    single = [detector.predict([im])[0] for im in sample_images]
    batched = detector.predict(sample_images)  # batch of 2 through the dynamic batch axis
    for a, b in zip(single, batched, strict=True):
        np.testing.assert_allclose(a.boxes, b.boxes, atol=1e-2)
        np.testing.assert_array_equal(a.class_ids, b.class_ids)
    people = int((batched[0].class_ids == 0).sum())
    assert people >= 3  # bus.jpg shows four people
