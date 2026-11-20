"""End-to-end evaluation smoke test on two real photos: the model's own confident detections are
the ground truth, so the pipeline (prediction, class mapping, matching, report) must score ~1."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tsd.eval.protocol import EvalSet
from tsd.eval.runner import RunDetections, report, run_model
from tsd.infer import load_detector

pytestmark = [pytest.mark.weights, pytest.mark.slow]


def test_self_consistent_evaluation(
    weights_path: Path, sample_images: list[np.ndarray], tmp_path: Path
) -> None:
    detector = load_detector(weights_path, threads=2)
    classes = ["Person", "Bus", "Traffic sign"]  # Traffic sign is not a COCO class
    records = []
    for i, image in enumerate(sample_images):
        path = tmp_path / "images" / f"{i}.jpg"
        path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        det = detector.predict([cv2.imread(str(path))])[0]
        keep = (det.scores > 0.5) & np.isin(det.class_ids, [0, 5])
        labels = [0 if c == 0 else 1 for c in det.class_ids[keep]]
        records.append(
            {
                "image_id": str(i),
                "file": f"images/{i}.jpg",
                "width": image.shape[1],
                "height": image.shape[0],
                "boxes": det.boxes[keep].tolist(),
                "labels": labels,
                "group_of": [False] * len(labels),
                "verified_positive": sorted(set(labels)),
                "verified_negative": [],
            }
        )
    assert sum(len(r["labels"]) for r in records) >= 5
    (tmp_path / "annotations.json").write_text(json.dumps({"classes": classes, "images": records}))
    evalset = EvalSet.load(tmp_path)
    run = run_model(weights_path, evalset, label="yolo11n/pytorch-fp32", threads=2)
    assert run.backend == "pytorch-fp32"
    assert run.source == weights_path.name
    run.save(tmp_path / "run.json")
    doc = report(evalset, [RunDetections.load(tmp_path / "run.json")], replicates=10)
    result = doc["runs"]["yolo11n/pytorch-fp32"]
    assert result["classes"] == ["Person", "Bus"]
    assert result["excluded_classes"] == ["Traffic sign"]
    assert result["map50"] > 0.99
    assert result["pycocotools"]["max_abs_diff"] < 1e-9
