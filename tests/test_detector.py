"""Detector composition with a fake backend, model metadata and backend dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsd.infer import Detections, Detector, ModelMeta, NMSConfig, backend_for


class FakeBackend:
    """Returns one confident 'class 1' box centred at (320, 320), 64 px wide, in input pixels."""

    name = "fake"

    def __init__(self) -> None:
        self.batches: list[tuple[int, ...]] = []

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self.batches.append(batch.shape)
        raw = np.zeros((batch.shape[0], 4 + 3, 10), np.float32)
        raw[:, :4, 0] = [320, 320, 64, 32]
        raw[:, 4 + 1, 0] = 0.9
        return raw


def test_boxes_are_mapped_back_to_original_pixels() -> None:
    backend = FakeBackend()
    det = Detector(backend, ["a", "b", "c"])
    image = np.zeros((960, 1280, 3), np.uint8)  # scale 0.5, pad_y 80
    (out,) = det.predict([image])
    assert backend.batches == [(1, 3, 640, 640)]
    np.testing.assert_array_equal(out.class_ids, [1])
    np.testing.assert_allclose(out.boxes[0], [576, 448, 704, 512])
    assert out.to_dicts(det.names)[0]["class_name"] == "b"


def test_batch_prediction_and_timings() -> None:
    det = Detector(FakeBackend(), ["a", "b", "c"], nms=NMSConfig(conf_threshold=0.95))
    images = [np.zeros((480, 640, 3), np.uint8), np.zeros((100, 300, 3), np.uint8)]
    dets, times = det.predict_timed(images)
    assert [len(d) for d in dets] == [0, 0]  # the per-detector NMS config applies
    assert times.total_ms >= times.infer_ms >= 0
    dets = det.predict(images, nms=NMSConfig(conf_threshold=0.5))  # per-call override
    assert [len(d) for d in dets] == [1, 1]


def test_detections_validation_and_helpers() -> None:
    with pytest.raises(ValueError, match="same detections"):
        Detections(np.zeros((2, 4), np.float32), np.zeros(1, np.float32), np.zeros(1, np.int64))
    a = Detections(np.ones((1, 4), np.float32), np.array([0.2], np.float32), np.array([0]))
    b = Detections(np.ones((2, 4), np.float32), np.array([0.9, 0.5], np.float32), np.array([1, 2]))
    both = Detections.concatenate([a, b])
    assert len(both) == 3
    assert [d["score"] for d in both.to_dicts(["x", "y", "z"])] == [0.9, 0.5, 0.2]
    assert len(Detections.concatenate([])) == 0


def test_model_meta_round_trip(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    meta = ModelMeta(["a", "b"], (640, 640), "int8", "m.pt", "abc", {"opset": 18})
    assert meta.save(model) == tmp_path / "m.onnx.json"
    assert ModelMeta.load(model) == meta


def test_backend_dispatch() -> None:
    assert backend_for(Path("x.pt")) == "pytorch"
    assert backend_for(Path("x.onnx")) == "onnxruntime"
    assert backend_for(Path("dir/x.xml")) == "openvino"
    with pytest.raises(ValueError, match="cannot infer"):
        backend_for(Path("x.tflite"))
