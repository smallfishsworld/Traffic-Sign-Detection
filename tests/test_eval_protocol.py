"""Open Images protocol: class mapping, unverified-class filtering, crowd mapping, reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import has_module
from tsd.eval.protocol import EvalSet, ground_truth, map_classes, predictions
from tsd.eval.runner import RunDetections, dataset_note, markdown_table, report
from tsd.infer import Detections

COCO_LIKE = [
    "person",
    "bicycle",
    "car",
    "x",
    "x",
    "x",
    "x",
    "x",
    "x",
    "traffic light",
    "x",
    "stop sign",
]


def _write_set(root: Path) -> EvalSet:
    images = [
        {  # stop sign present; traffic light verified absent
            "image_id": "a",
            "split": "test",
            "file": "images/a.jpg",
            "width": 640,
            "height": 480,
            "boxes": [[10, 10, 110, 110]],
            "labels": [1],
            "group_of": [False],
            "depiction": [False],
            "occluded": [False],
            "truncated": [False],
            "verified_positive": [0, 1],
            "verified_negative": [2],
        },
        {  # traffic light group-of box; stop sign never verified
            "image_id": "b",
            "split": "test",
            "file": "images/b.jpg",
            "width": 640,
            "height": 480,
            "boxes": [[200, 50, 300, 250], [0, 0, 20, 20]],
            "labels": [2, 0],
            "group_of": [True, False],
            "depiction": [False, False],
            "occluded": [False, False],
            "truncated": [False, False],
            "verified_positive": [0, 2],
            "verified_negative": [],
        },
    ]
    doc = {"info": {}, "classes": ["Traffic sign", "Stop sign", "Traffic light"], "images": images}
    (root / "annotations.json").write_text(json.dumps(doc))
    return EvalSet.load(root)


def _dets(rows: list[tuple[list[float], float, int]]) -> Detections:
    if not rows:
        return Detections.empty()
    return Detections(
        np.array([r[0] for r in rows], np.float32),
        np.array([r[1] for r in rows], np.float32),
        np.array([r[2] for r in rows], np.int64),
    )


def test_class_mapping_by_name() -> None:
    cmap = map_classes(["Traffic sign", "Stop sign", "Traffic light"], COCO_LIKE)
    assert cmap.names == ["Stop sign", "Traffic light"]
    assert cmap.dataset_ids == [1, 2]
    assert cmap.model_ids == [11, 9]
    assert cmap.excluded == ["Traffic sign"]
    np.testing.assert_array_equal(cmap.to_eval_ids(np.array([11, 9, 0, 2])), [0, 1, -1, -1])
    with pytest.raises(ValueError, match="none of"):
        map_classes(["Traffic sign"], COCO_LIKE)


def test_ground_truth_keeps_evaluated_classes_and_marks_group_of_as_crowd(tmp_path: Path) -> None:
    evalset = _write_set(tmp_path)
    cmap = map_classes(evalset.classes, COCO_LIKE)
    a, b = ground_truth(evalset, cmap)
    np.testing.assert_array_equal(a.class_ids, [0])
    np.testing.assert_array_equal(b.class_ids, [1])  # the Traffic sign box is dropped
    np.testing.assert_array_equal(b.crowd, [True])
    assert evalset.paths()[0] == tmp_path / "images/a.jpg"


def test_unverified_class_detections_are_dropped(tmp_path: Path) -> None:
    evalset = _write_set(tmp_path)
    cmap = map_classes(evalset.classes, COCO_LIKE)
    dets = [
        _dets(
            [([10, 10, 110, 110], 0.9, 11), ([400, 10, 420, 60], 0.5, 9), ([0, 0, 5, 5], 0.4, 0)]
        ),
        _dets([([210, 60, 240, 120], 0.8, 9), ([500, 300, 600, 400], 0.7, 11)]),
    ]
    preds, stats = predictions(evalset, cmap, dets)
    # image a: the traffic light is verified absent -> kept (a false positive); person dropped
    np.testing.assert_array_equal(preds[0].class_ids, [0, 1])
    # image b: stop sign was never verified there -> dropped
    np.testing.assert_array_equal(preds[1].class_ids, [1])
    assert stats.dropped_detections == {"Stop sign": 1, "Traffic light": 0}
    assert stats.unevaluated_images == {"Stop sign": 1, "Traffic light": 0}
    strict, strict_stats = predictions(evalset, cmap, dets, drop_unverified=False)
    np.testing.assert_array_equal(strict[1].class_ids, [1, 0])
    assert strict_stats.dropped_detections == {"Stop sign": 0, "Traffic light": 0}
    with pytest.raises(ValueError, match="one Detections"):
        predictions(evalset, cmap, dets[:1])


def _run(label: str, backend: str, dets: list[Detections]) -> RunDetections:
    return RunDetections(
        label=label,
        model=f"{label}.bin",
        backend=backend,
        source="m.pt",
        model_names=COCO_LIKE,
        nms={},
        files_sha256={},
        image_ids=["a", "b"],
        detections=dets,
    )


@pytest.mark.skipif(not has_module("pycocotools"), reason="pycocotools not installed")
def test_report_pairs_runs_with_their_pytorch_baseline(tmp_path: Path) -> None:
    evalset = _write_set(tmp_path)
    good = [_dets([([10, 10, 110, 110], 0.9, 11)]), _dets([([210, 60, 240, 120], 0.8, 9)])]
    worse = [_dets([([30, 30, 130, 130], 0.9, 11)]), _dets([])]
    base, same, other = (
        _run("m/pytorch-fp32", "pytorch-fp32", good),
        _run("m/onnxruntime-fp32", "onnxruntime-fp32", good),
        _run("m/openvino-int8", "openvino-int8", worse),
    )
    path = tmp_path / "run.json"
    other.save(path)
    loaded = RunDetections.load(path)
    assert loaded.label == other.label
    np.testing.assert_allclose(loaded.detections[0].boxes, worse[0].boxes)
    doc = report(evalset, [base, same, loaded], replicates=20)
    runs = doc["runs"]
    assert runs["m/pytorch-fp32"]["map50"] == pytest.approx(1.0)
    assert "delta_vs_baseline" not in runs["m/pytorch-fp32"]
    assert runs["m/onnxruntime-fp32"]["delta_vs_baseline"]["map50_95"] == 0.0
    assert runs["m/onnxruntime-fp32"]["delta_vs_baseline"]["ci95"]["map50"] == [0.0, 0.0]
    assert runs["m/openvino-int8"]["delta_vs_baseline"]["map50"] < 0
    assert all(r["pycocotools"]["max_abs_diff"] < 1e-9 for r in runs.values())
    assert runs["m/pytorch-fp32"]["excluded_classes"] == ["Traffic sign"]
    table = markdown_table(doc)
    assert "(baseline)" in table
    assert "m/openvino-int8" in table
    bad = _run("x", "pytorch-fp32", good)
    bad.image_ids = ["b", "a"]
    with pytest.raises(ValueError, match="different image list"):
        report(evalset, [bad], replicates=2)


@pytest.mark.skipif(not has_module("pycocotools"), reason="pycocotools not installed")
def test_tiled_run_is_compared_with_the_untiled_run_of_its_backend(tmp_path: Path) -> None:
    evalset = _write_set(tmp_path)
    untiled = [_dets([]), _dets([])]  # misses the (large) stop sign of image a
    tiled = [_dets([([10, 10, 110, 110], 0.9, 11)]), _dets([])]
    pt = _run("m/pytorch-fp32", "pytorch-fp32", untiled)
    ov = _run("m/openvino-fp32", "openvino-fp32", untiled)
    ov_tiled = _run("m/openvino-fp32+tiles512", "openvino-fp32", tiled)
    ov_tiled.extra = {"tiling": {"tile_size": 512}}
    with pytest.warns(RuntimeWarning, match="All-NaN"):  # no small or medium ground truth here
        doc = report(evalset, [pt, ov, ov_tiled], replicates=20, run_cross_check=False)
    runs = doc["runs"]
    assert runs["m/openvino-fp32"]["baseline"] == "m/pytorch-fp32"
    assert runs["m/openvino-fp32+tiles512"]["baseline"] == "m/openvino-fp32"
    assert runs["m/openvino-fp32+tiles512"]["delta_vs_baseline"]["map50"] == 1.0
    by_size = runs["m/openvino-fp32+tiles512"]["by_size_delta_vs_baseline"]
    assert set(by_size) == {"small", "medium", "large"}
    assert by_size["large"]["map50_95"] == 1.0
    assert by_size["large"]["ar100"] == 1.0
    assert by_size["small"]["map50_95"] is None
    assert "by_size_delta_vs_baseline" in runs["m/openvino-fp32"]  # once any run is tiled


def test_boxes_count_for_their_ancestor_classes(tmp_path: Path) -> None:
    """A model with both Traffic sign and Stop sign: a Traffic-sign detection on a Stop-sign box
    is a true positive (Open Images box expansion), and the COCO-style map is unchanged."""
    doc = json.loads((_write_set(tmp_path) and (tmp_path / "annotations.json")).read_text())
    doc["info"] = {"class_parents": {"Stop sign": ["Traffic sign"]}}
    doc["images"] = doc["images"][:1]  # one Stop-sign box, Traffic sign verified present
    (tmp_path / "annotations.json").write_text(json.dumps(doc))
    evalset = EvalSet.load(tmp_path)
    assert evalset.images[0].labels.tolist() == [1, 0]  # the copy for the parent class
    names = ["traffic sign", "stop sign"]
    cmap = map_classes(evalset.classes, names)
    (truth,) = ground_truth(evalset, cmap)
    assert sorted(truth.class_ids.tolist()) == [0, 1]
    run = RunDetections(
        label="m/x", model="m.pt", backend="x", source="m.pt", model_names=names, nms={},
        files_sha256={}, image_ids=["a"], detections=[_dets([([10, 10, 110, 110], 0.9, 0)])],
    )  # fmt: skip
    entry = report(evalset, [run], replicates=5, run_cross_check=False)["runs"]["m/x"]
    assert entry["per_class"]["Traffic sign"]["ap50"] == pytest.approx(1.0)
    # the COCO-like class map evaluates no Traffic sign: nothing changes for those models
    assert "Traffic sign" in map_classes(evalset.classes, COCO_LIKE).excluded


def test_cross_check_flags_one_sided_nan() -> None:
    from tsd.eval.runner import cross_check

    same = cross_check(
        {"map50": 0.5, "map_large": float("nan")}, {"map50": 0.5001, "map_large": float("nan")}
    )
    assert same["max_abs_diff"] == pytest.approx(1e-4)
    assert same["nan_mismatch"] == []
    bad = cross_check({"map50": 0.5, "map_large": 0.2}, {"map50": 0.5, "map_large": float("nan")})
    assert bad["nan_mismatch"] == ["map_large"]
    assert bad["max_abs_diff"] == float("inf")


@pytest.mark.skipif(not has_module("pycocotools"), reason="pycocotools not installed")
def test_pycocotools_without_detections_is_not_all_zero(tmp_path: Path) -> None:
    from tsd.eval.runner import pycocotools_stats

    evalset = _write_set(tmp_path)
    cmap = map_classes(evalset.classes, COCO_LIKE)
    truths = ground_truth(evalset, cmap)
    preds, _ = predictions(evalset, cmap, [Detections.empty(), Detections.empty()])
    stats = pycocotools_stats(truths, preds, cmap.names)
    assert stats["map50"] == 0.0
    assert np.isnan(stats["map_small"])  # no small non-crowd ground truth at all
    assert stats["map_large"] == 0.0


def test_detection_files_round_trip_gzip(tmp_path: Path) -> None:
    run = RunDetections(
        label="m/x", model="m.pt", backend="x", source="m.pt", model_names=["a"], nms={},
        files_sha256={}, image_ids=["a"], detections=[_dets([([1, 2, 3, 4], 0.5, 0)])],
    )  # fmt: skip
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    run.save(a)
    run.save(b)
    assert a.read_bytes() == b.read_bytes()  # deterministic (no gzip timestamp)
    assert a.read_bytes()[:2] == b"\x1f\x8b"
    assert RunDetections.load(a).detections[0].boxes.tolist() == [[1, 2, 3, 4]]


def test_saved_detections_are_reused_only_when_nothing_changed(tmp_path: Path) -> None:
    """The settings are saved as JSON (tuples become lists); the first version compared them
    with the live tuples and never reused anything."""
    from dataclasses import asdict

    from tsd.eval.runner import can_reuse
    from tsd.infer import NMSConfig

    nms = asdict(NMSConfig(conf_threshold=0.001, iou_threshold=0.7, classes=(11, 9)))
    tiling = {"tile_size": 512, "overlap": 0.2, "merge_iou": None}
    run = RunDetections(
        label="m/x", model="m.pt", backend="x", source="m.pt", model_names=["a"], nms=nms,
        files_sha256={"m.pt": "ab"}, image_ids=["a", "b"], extra={"tiling": tiling},
        detections=[_dets([([1, 2, 3, 4], 0.5, 0)]), _dets([])],
    )  # fmt: skip
    run.save(tmp_path / "r.json.gz")
    saved = RunDetections.load(tmp_path / "r.json.gz")
    assert can_reuse(saved, {"m.pt": "ab"}, tiling, nms, ["a", "b"])
    assert not can_reuse(saved, {"m.pt": "cd"}, tiling, nms, ["a", "b"])
    assert not can_reuse(saved, {"m.pt": "ab"}, None, nms, ["a", "b"])
    assert not can_reuse(saved, {"m.pt": "ab"}, tiling, {**nms, "iou_threshold": 0.5}, ["a", "b"])
    assert not can_reuse(saved, {"m.pt": "ab"}, tiling, nms, ["a"])


def test_dataset_note_counts_verified_negative_images(tmp_path: Path) -> None:
    evalset = _write_set(tmp_path)
    assert dataset_note(evalset).startswith("2 Open Images V7")
    assert "are not included" in dataset_note(evalset)
    doc = json.loads((tmp_path / "annotations.json").read_text())
    negative = {**doc["images"][0], "image_id": "c", "boxes": [], "labels": [], "group_of": []}
    negative.update(
        depiction=[], occluded=[], truncated=[], verified_positive=[], verified_negative=[1]
    )
    doc["images"].append(negative)
    (tmp_path / "annotations.json").write_text(json.dumps(doc))
    note = dataset_note(EvalSet.load(tmp_path))
    assert note.startswith("3 Open Images V7")
    assert "1 verified-negative images" in note
    assert "not included" not in note
