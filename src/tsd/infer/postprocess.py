"""Post-processing of the raw YOLO head output: candidate selection and NMS in NumPy.

The YOLO11 detect head outputs, per image, a ``(4 + num_classes, num_anchors)`` array: box centre
and size in network-input pixels followed by per-class sigmoid scores. This module turns it into
final detections the same way Ultralytics' ``non_max_suppression`` does (strict ``> conf``
threshold, optional multi-label candidates, class-aware NMS via per-class coordinate offsets,
``max_det`` cap), so every backend shares one tested implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tsd.boxes import cxcywh_to_xyxy

_CLASS_OFFSET = 7680.0  # larger than any input coordinate, as in Ultralytics (max_wh)


@dataclass(frozen=True)
class NMSConfig:
    """Selection settings. The defaults are Ultralytics' ``predict`` defaults; evaluation uses
    :data:`EVAL_NMS` (Ultralytics' ``val`` settings).

    ``classes`` restricts candidates *before* the per-anchor arg-max: the result is the detector
    restricted to those classes (Ultralytics instead drops an anchor whose best class is not in the
    list). With ``multi_label=True`` the two are identical.
    """

    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    max_det: int = 300
    classes: tuple[int, ...] | None = None
    multi_label: bool = False
    agnostic: bool = False
    max_candidates: int = 30000


EVAL_NMS = NMSConfig(conf_threshold=0.001, iou_threshold=0.7, max_det=300, multi_label=True)


def nms(
    boxes: NDArray[np.floating], scores: NDArray[np.floating], iou_threshold: float
) -> NDArray[np.int64]:
    """Greedy non-maximum suppression; returns kept indices in descending score order.

    A box is suppressed when its IoU with a higher-scoring kept box is strictly greater than
    ``iou_threshold`` (the ``torchvision.ops.nms`` convention). Each step computes the IoU of the
    current best box against all remaining boxes in one vectorised operation.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    b = boxes.astype(np.float64, copy=False)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating],
    class_ids: NDArray[np.int64],
    iou_threshold: float,
    class_offset: float | None = None,
) -> NDArray[np.int64]:
    """Class-aware NMS: boxes of different classes never suppress each other.

    Classes are separated by shifting each box by ``class_id * class_offset``, which must exceed
    every coordinate. The default is Ultralytics' 7680 (model-input coordinates), raised to the
    largest coordinate + 1 for boxes in full-resolution image coordinates (merged tiles of an
    image larger than 7680 px in both dimensions would otherwise let classes suppress each other).
    """
    if class_offset is None:
        class_offset = max(_CLASS_OFFSET, float(np.abs(boxes).max(initial=0.0)) + 1.0)
    offsets = class_ids.astype(np.float64)[:, None] * class_offset
    return nms(boxes.astype(np.float64) + offsets, scores, iou_threshold)


def select_detections(
    pred: NDArray[np.float32], config: NMSConfig
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """Final ``(boxes_xyxy, scores, class_ids)`` for one image from its raw head output."""
    scores_all = pred[4:].T  # (anchors, classes)
    class_map = np.arange(scores_all.shape[1])
    if config.classes is not None:
        class_map = np.asarray(config.classes, dtype=np.int64)
        scores_all = scores_all[:, class_map]
    if config.multi_label:
        anchor_idx, col = np.nonzero(scores_all > config.conf_threshold)
        scores = scores_all[anchor_idx, col]
    else:
        col = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(col)), col]
        anchor_idx = np.nonzero(scores > config.conf_threshold)[0]
        col, scores = col[anchor_idx], scores[anchor_idx]
    if len(scores) > config.max_candidates:
        top = np.argsort(-scores, kind="stable")[: config.max_candidates]
        anchor_idx, col, scores = anchor_idx[top], col[top], scores[top]
    boxes = cxcywh_to_xyxy(pred[:4, anchor_idx].T)
    class_ids = class_map[col].astype(np.int64)
    if config.agnostic:
        keep = nms(boxes, scores, config.iou_threshold)
    else:
        keep = batched_nms(boxes, scores, class_ids, config.iou_threshold)
    keep = keep[: config.max_det]
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32), class_ids[keep]


def select_batch(
    raw: NDArray[np.float32], config: NMSConfig
) -> Sequence[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]]:
    return [select_detections(raw[i], config) for i in range(raw.shape[0])]
