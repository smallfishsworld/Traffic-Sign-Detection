"""NMS and candidate selection: hand-built cases, properties and parity with reference code."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import has_module
from tsd.boxes import box_iou
from tsd.infer.postprocess import NMSConfig, batched_nms, nms, select_detections


def _raw(boxes_cxcywh: list[list[float]], scores: list[list[float]]) -> np.ndarray:
    """Raw head output (4 + nc, anchors) from per-anchor boxes and class scores."""
    return np.concatenate([np.array(boxes_cxcywh).T, np.array(scores).T]).astype(np.float32)


def _random_raw(rng: np.random.Generator, anchors: int = 600, nc: int = 5) -> np.ndarray:
    cxcy = rng.uniform(0, 640, (anchors, 2))
    wh = rng.uniform(5, 120, (anchors, 2))
    scores = rng.uniform(0, 1, (anchors, nc)) ** 4
    return _raw(np.concatenate([cxcy, wh], 1).tolist(), scores.tolist())


def test_nms_suppresses_overlaps_strictly_above_threshold() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 20, 30, 30], [0, 0, 10, 5]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    # IoU(0, 1) = 90/110 = 0.818; IoU(0, 3) = 0.5 exactly.
    np.testing.assert_array_equal(nms(boxes, scores, 0.5), [0, 2, 3])
    np.testing.assert_array_equal(nms(boxes, scores, 0.49), [0, 2])
    np.testing.assert_array_equal(nms(boxes, scores, 0.9), [0, 1, 2, 3])
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).shape == (0,)


def test_batched_nms_keeps_overlapping_boxes_of_other_classes() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], float)
    scores = np.array([0.9, 0.8, 0.7])
    np.testing.assert_array_equal(batched_nms(boxes, scores, np.array([0, 1, 0]), 0.5), [0, 1])


@pytest.mark.skipif(not has_module("torchvision"), reason="torchvision not installed")
@pytest.mark.parametrize("seed", range(5))
def test_nms_matches_torchvision(seed: int) -> None:
    import torch
    import torchvision

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 500, (400, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(5, 100, (400, 2))], 1).astype(np.float32)
    scores = rng.permutation(400).astype(np.float32) / 400  # distinct scores: order is unique
    ours = nms(boxes, scores, 0.45)
    ref = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.45).numpy()
    np.testing.assert_array_equal(ours, ref)


def test_select_single_and_multi_label() -> None:
    raw = _raw(
        [[100, 100, 20, 20], [300, 300, 40, 40], [500, 500, 10, 10]],
        [[0.9, 0.6, 0.0], [0.2, 0.25, 0.1], [0.0, 0.0, 0.3]],
    )
    boxes, scores, cls = select_detections(raw, NMSConfig(conf_threshold=0.25))
    # Anchor 1's best score is exactly 0.25: not strictly above the threshold.
    np.testing.assert_array_equal(cls, [0, 2])
    np.testing.assert_allclose(boxes[0], [90, 90, 110, 110])
    _, scores, cls = select_detections(raw, NMSConfig(conf_threshold=0.25, multi_label=True))
    np.testing.assert_array_equal(cls, [0, 1, 2])  # anchor 0 contributes two classes
    np.testing.assert_allclose(scores, [0.9, 0.6, 0.3])


def test_class_filter_and_max_det() -> None:
    raw = _raw(
        [[100, 100, 20, 20], [300, 300, 40, 40]],
        [[0.9, 0.6, 0.0], [0.2, 0.5, 0.1]],
    )
    _, _, cls = select_detections(raw, NMSConfig(conf_threshold=0.25, classes=(1,)))
    np.testing.assert_array_equal(cls, [1, 1])  # best *allowed* class per anchor
    _, scores, _ = select_detections(raw, NMSConfig(conf_threshold=0.25, max_det=1))
    np.testing.assert_allclose(scores, [0.9])
    empty = select_detections(raw, NMSConfig(conf_threshold=0.95))
    assert [len(x) for x in empty] == [0, 0, 0]


@pytest.mark.parametrize("seed", range(5))
def test_no_same_class_pair_overlaps_above_threshold(seed: int) -> None:
    config = NMSConfig(conf_threshold=0.05, iou_threshold=0.5, multi_label=True)
    boxes, scores, cls = select_detections(_random_raw(np.random.default_rng(seed)), config)
    assert len(boxes) > 10
    assert (np.diff(scores) <= 0).all()
    for c in np.unique(cls):
        iou = box_iou(boxes[cls == c], boxes[cls == c])
        np.fill_diagonal(iou, 0)
        assert iou.max() <= 0.5 + 1e-6


@pytest.mark.skipif(not has_module("ultralytics"), reason="ultralytics not installed")
@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_matches_ultralytics_nms(seed: int, multi_label: bool) -> None:
    import torch

    from tsd.infer.backends import import_ultralytics

    import_ultralytics()
    from ultralytics.utils.nms import non_max_suppression

    raw = _random_raw(np.random.default_rng(10 + seed), anchors=800, nc=6)
    ref = non_max_suppression(
        torch.from_numpy(raw[None]), 0.05, 0.6, multi_label=multi_label, max_det=300
    )[0].numpy()
    config = NMSConfig(conf_threshold=0.05, iou_threshold=0.6, multi_label=multi_label)
    boxes, scores, cls = select_detections(raw, config)
    order = np.lexsort((cls, -scores))
    ref_order = np.lexsort((ref[:, 5], -ref[:, 4]))
    np.testing.assert_allclose(boxes[order], ref[ref_order, :4], atol=1e-3)
    np.testing.assert_allclose(scores[order], ref[ref_order, 4], atol=1e-6)
    np.testing.assert_array_equal(cls[order], ref[ref_order, 5].astype(np.int64))
