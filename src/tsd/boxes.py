"""Axis-aligned box utilities shared by data tooling, augmentation, inference and evaluation.

Conventions used throughout the package:

* ``xyxy``: absolute pixel coordinates ``(x1, y1, x2, y2)`` with ``x1 <= x2`` and ``y1 <= y2``.
  Coordinates are continuous (a box covering pixel columns 0..9 is ``x1=0, x2=10``).
* ``cxcywh``: centre and size in absolute pixels (the raw YOLO head output format).
* ``xywhn``: centre and size normalised by image width/height (the YOLO label-file format).

All functions take and return ``float64`` arrays of shape ``(N, 4)`` unless stated otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_boxes(boxes: ArrayLike) -> FloatArray:
    """Return ``boxes`` as a ``(N, 4)`` float64 array (an empty input gives shape ``(0, 4)``)."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected boxes of shape (N, 4), got {arr.shape}")
    return arr


def cxcywh_to_xyxy(boxes: ArrayLike) -> FloatArray:
    b = as_boxes(boxes)
    half_w, half_h = b[:, 2] / 2.0, b[:, 3] / 2.0
    return np.stack(
        [b[:, 0] - half_w, b[:, 1] - half_h, b[:, 0] + half_w, b[:, 1] + half_h], axis=1
    )


def xyxy_to_cxcywh(boxes: ArrayLike) -> FloatArray:
    b = as_boxes(boxes)
    return np.stack(
        [
            (b[:, 0] + b[:, 2]) / 2.0,
            (b[:, 1] + b[:, 3]) / 2.0,
            b[:, 2] - b[:, 0],
            b[:, 3] - b[:, 1],
        ],
        axis=1,
    )


def xyxy_to_xywhn(boxes: ArrayLike, width: float, height: float) -> FloatArray:
    """Absolute ``xyxy`` to normalised centre/size (YOLO label format)."""
    c = xyxy_to_cxcywh(boxes)
    return c / np.array([width, height, width, height], dtype=np.float64)


def xywhn_to_xyxy(boxes: ArrayLike, width: float, height: float) -> FloatArray:
    """Normalised centre/size (YOLO label format) to absolute ``xyxy``."""
    b = as_boxes(boxes) * np.array([width, height, width, height], dtype=np.float64)
    return cxcywh_to_xyxy(b)


def normalize_corners(boxes: ArrayLike) -> FloatArray:
    """Sort each box's coordinates so that ``x1 <= x2`` and ``y1 <= y2``."""
    b = as_boxes(boxes)
    return np.stack(
        [
            np.minimum(b[:, 0], b[:, 2]),
            np.minimum(b[:, 1], b[:, 3]),
            np.maximum(b[:, 0], b[:, 2]),
            np.maximum(b[:, 1], b[:, 3]),
        ],
        axis=1,
    )


def clip_boxes(boxes: ArrayLike, width: float, height: float) -> FloatArray:
    """Clip ``xyxy`` boxes to the image rectangle ``[0, width] x [0, height]``."""
    b = as_boxes(boxes).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0.0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0.0, height)
    return b


def box_area(boxes: ArrayLike) -> FloatArray:
    b = as_boxes(boxes)
    return np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)


def box_iou(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Pairwise IoU matrix of shape ``(len(a), len(b))``."""
    a_, b_ = as_boxes(a), as_boxes(b)
    inter = _intersection(a_, b_)
    union = box_area(a_)[:, None] + box_area(b_)[None, :] - inter
    out: FloatArray = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return out


def box_ioa(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Pairwise intersection over the area of ``a`` (shape ``(len(a), len(b))``).

    This is the overlap measure COCO uses for "crowd" ground truth and the one used to decide
    how much of an existing box a pasted template would cover.
    """
    a_, b_ = as_boxes(a), as_boxes(b)
    inter = _intersection(a_, b_)
    area_a = np.broadcast_to(box_area(a_)[:, None], inter.shape)
    out: FloatArray = np.divide(inter, area_a, out=np.zeros_like(inter), where=area_a > 0)
    return out


def _intersection(a: FloatArray, b: FloatArray) -> FloatArray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh: FloatArray = np.clip(rb - lt, 0.0, None)
    inter: FloatArray = wh[..., 0] * wh[..., 1]
    return inter
