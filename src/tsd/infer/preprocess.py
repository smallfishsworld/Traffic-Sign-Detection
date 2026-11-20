"""Letterbox pre-processing shared by every backend.

Mirrors Ultralytics' ``LetterBox(auto=False, center=True)`` (same rounding and pad colour), so the
PyTorch, ONNX Runtime and OpenVINO backends all receive bit-identical input tensors and differ only
in how they execute the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from tsd.boxes import FloatArray, as_boxes, clip_boxes

Image = NDArray[np.uint8]
PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    """How an original image was mapped into the network input."""

    scale: float
    pad_x: int
    pad_y: int
    orig_width: int
    orig_height: int
    # Actual per-axis resize gains (resized size / original size). They differ slightly from
    # ``scale`` because the resized size is rounded to whole pixels; None means ``scale``.
    gain_x: float | None = None
    gain_y: float | None = None

    def to_original(self, boxes: ArrayLike) -> FloatArray:
        """Map ``xyxy`` boxes from network-input pixels back to original-image pixels (clipped),
        with the per-axis gains like Ultralytics' ``scale_boxes``."""
        b = as_boxes(boxes).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / (self.gain_x or self.scale)
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / (self.gain_y or self.scale)
        return clip_boxes(b, self.orig_width, self.orig_height)


def letterbox(
    image: Image, size: tuple[int, int] = (640, 640), pad_value: int = PAD_VALUE
) -> tuple[Image, LetterboxInfo]:
    """Resize keeping the aspect ratio and pad to ``size`` = ``(height, width)``."""
    h, w = image.shape[:2]
    out_h, out_w = size
    r = min(out_h / h, out_w / w)
    new_w, new_h = max(1, round(w * r)), max(1, round(h * r))  # thin images: keep 1 px
    dw, dh = (out_w - new_w) / 2, (out_h - new_h) / 2
    resized = image
    if (new_w, new_h) != (w, h):
        resized = np.asarray(
            cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR), dtype=np.uint8
        )
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    info = LetterboxInfo(r, left, top, w, h, gain_x=new_w / w, gain_y=new_h / h)
    return np.asarray(padded, dtype=np.uint8), info


def to_input_tensor(images: Sequence[Image]) -> NDArray[np.float32]:
    """Stack letterboxed BGR ``uint8`` images into a contiguous ``float32`` NCHW RGB batch in
    ``[0, 1]``."""
    batch = np.stack(images)[..., ::-1].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(batch, dtype=np.float32) / np.float32(255.0)


def preprocess(
    images: Sequence[Image], size: tuple[int, int] = (640, 640)
) -> tuple[NDArray[np.float32], list[LetterboxInfo]]:
    boxed = [letterbox(im, size) for im in images]
    return to_input_tensor([b for b, _ in boxed]), [info for _, info in boxed]
