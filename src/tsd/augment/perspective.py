"""Perspective (homography) augmentation that keeps bounding boxes consistent with the image.

A homography ``H`` maps a source point ``(x, y)`` to ``(x'/w', y'/w')`` where
``(x', y', w')^T = H (x, y, 1)^T`` (report section 4.2.3). An axis-aligned box does not stay
axis-aligned under ``H``: its four corners map to a convex quadrilateral (all corners in front of
the projection plane). That quadrilateral is clipped to the output image (Sutherland-Hodgman) and
the label is the axis-aligned box of the clipped polygon, i.e. of the part of the warped box that
is still in the image. Boxes are dropped when:

* any corner maps behind the projection plane (``w' <= 0``), which happens only for extreme ``H``;
* the box is narrower or shorter than ``min_size`` pixels;
* less than ``min_visibility`` of the warped quadrilateral's area is inside the image ("mostly
  out of frame");
* its aspect ratio exceeds ``max_aspect`` (a sliver left at the border).

The label encloses the warped *box*, which encloses the object, so labels stay complete; under
strong rotation the box of a rotated rectangle is looser than the object inside it
(ENGINEERING_NOTES section 1.6). Clipping the polygon rather than the enclosing box keeps boxes
cut by the image border tight (the enclosing box clipped to the image can extend along the
border far beyond the visible part).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from tsd.boxes import FloatArray, as_boxes

Image = NDArray[np.uint8]
_EPS = 1e-9


def corners(boxes: FloatArray) -> FloatArray:
    """The four corners of each ``xyxy`` box, shape ``(N, 4, 2)`` (clockwise from top-left)."""
    b = as_boxes(boxes)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            np.stack([x1, y1], 1),
            np.stack([x2, y1], 1),
            np.stack([x2, y2], 1),
            np.stack([x1, y2], 1),
        ],
        axis=1,
    )


def transform_points(points: FloatArray, H: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Apply ``H`` to ``(N, 2)`` points; returns the projected points and their ``w'``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homog = pts @ H[:, :2].T + H[:, 2]
    w = homog[:, 2]
    safe_w = np.where(np.abs(w) < _EPS, _EPS, w)
    return homog[:, :2] / safe_w[:, None], w


def corner_homography(
    width: int, height: int, max_shift: float, rng: np.random.Generator
) -> FloatArray:
    """Random homography that moves each image corner by up to ``max_shift`` of the image size.

    This "four-point" parameterisation produces plausible viewpoint changes and is bounded by
    construction (unlike sampling the 8 matrix entries directly).
    """
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    shift = rng.uniform(-max_shift, max_shift, size=(4, 2)) * np.array([width, height])
    dst = src + shift
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return np.asarray(H, dtype=np.float64)


@dataclass(frozen=True)
class BoxFilter:
    min_size: float = 2.0
    min_visibility: float = 0.25
    max_aspect: float = 20.0


def polygon_area(poly: FloatArray) -> float:
    """Area of a simple polygon given as ``(K, 2)`` vertices in order (shoelace formula)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def clip_polygon(poly: FloatArray, width: float, height: float) -> FloatArray:
    """Clip a convex polygon ``(K, 2)`` to ``[0, width] x [0, height]`` (Sutherland-Hodgman)."""
    out = np.asarray(poly, dtype=np.float64)
    for axis, limit, keep_below in ((0, 0.0, False), (0, width, True), (1, 0.0, False),
                                    (1, height, True)):  # fmt: skip
        if len(out) == 0:
            break
        inside = out[:, axis] <= limit if keep_below else out[:, axis] >= limit
        kept: list[FloatArray] = []
        for j in range(len(out)):
            a, b = out[j - 1], out[j]
            a_in, b_in = inside[j - 1], inside[j]
            if a_in != b_in:  # the edge crosses the clip line: add the crossing point
                t = (limit - a[axis]) / (b[axis] - a[axis])
                crossing = a + t * (b - a)
                crossing[axis] = limit  # exactly on the line, no rounding outside the image
                kept.append(crossing)
            if b_in:
                kept.append(b)
        out = np.array(kept).reshape(-1, 2)
    return out


def warp_boxes(
    boxes: FloatArray,
    H: FloatArray,
    out_width: int,
    out_height: int,
    box_filter: BoxFilter | None = None,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Map ``xyxy`` boxes through ``H`` into an ``out_width x out_height`` image.

    Returns the kept boxes (clipped to the image) and the boolean keep-mask over the input boxes.
    """
    f = box_filter or BoxFilter()
    b = as_boxes(boxes)
    n = len(b)
    if n == 0:
        return b, np.zeros(0, dtype=bool)
    pts, w = transform_points(corners(b).reshape(-1, 2), H)
    pts = pts.reshape(n, 4, 2)
    in_front = (w.reshape(n, 4) > _EPS).all(axis=1)
    clipped = np.zeros((n, 4))
    visible = np.zeros(n)
    for i in np.flatnonzero(in_front):
        full = polygon_area(pts[i])
        part = clip_polygon(pts[i], out_width, out_height)
        if full > 0 and len(part) >= 3:
            clipped[i] = [*part.min(axis=0), *part.max(axis=0)]
            visible[i] = polygon_area(part) / full
    wh = clipped[:, 2:] - clipped[:, :2]
    aspect = np.maximum(wh[:, 0], wh[:, 1]) / np.maximum(np.minimum(wh[:, 0], wh[:, 1]), _EPS)
    keep = (
        in_front
        & (wh >= f.min_size).all(axis=1)
        & (visible >= f.min_visibility)
        & (aspect <= f.max_aspect)
    )
    return clipped[keep], keep


def warp_image(
    image: Image, H: FloatArray, out_width: int, out_height: int, border_value: int = 114
) -> Image:
    warped = cv2.warpPerspective(
        image,
        H,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )
    return np.asarray(warped, dtype=np.uint8)


def apply_homography(
    image: Image,
    boxes: FloatArray,
    class_ids: NDArray[np.int64],
    H: FloatArray,
    box_filter: BoxFilter | None = None,
    border_value: int = 114,
) -> tuple[Image, FloatArray, NDArray[np.int64]]:
    """Warp an image and its labels with the same homography (output size = input size)."""
    height, width = image.shape[:2]
    out_boxes, keep = warp_boxes(boxes, H, width, height, box_filter)
    return warp_image(image, H, width, height, border_value), out_boxes, class_ids[keep]


@dataclass(frozen=True)
class RandomPerspective:
    """Apply a random corner homography with probability ``p`` (report: "perspective transform")."""

    max_shift: float = 0.08
    p: float = 0.5
    box_filter: BoxFilter = BoxFilter()
    border_value: int = 114

    def __call__(
        self,
        image: Image,
        boxes: FloatArray,
        class_ids: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> tuple[Image, FloatArray, NDArray[np.int64]]:
        if rng.random() >= self.p:
            return image, as_boxes(boxes), class_ids
        height, width = image.shape[:2]
        H = corner_homography(width, height, self.max_shift, rng)
        return apply_homography(image, boxes, class_ids, H, self.box_filter, self.border_value)
