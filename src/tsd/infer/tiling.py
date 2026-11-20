"""Sliced (SAHI-style) inference for small objects.

The detector sees a 640 x 640 letterbox of the whole image, so a 20 px traffic light in a
1024 px photo shrinks to 12 px. Tiled inference also runs the detector on overlapping crops
(``tile_size`` pixels of the *original* image, each letterboxed to the model input, i.e.
enlarged when ``tile_size`` < 640), maps their boxes back to image coordinates, adds the usual
full-image pass (so large objects that no tile contains whole are still found), and merges
everything with class-wise NMS.

A tile cuts through every object that crosses its border, and the detector happily reports the
visible part with a high score. Merged naively, such truncated boxes either suppress the
full-image box of a large object (when they overlap it enough) or become extra false positives.
A tile detection that reaches a tile border lying *inside* the image (within ``border_margin`` x
``tile_size`` px) is therefore dropped when a detection of the same class that touches no inner
border (from another tile or from the full-image pass) covers at least ``rescue_ioa`` of it: an
object smaller than the overlap minus twice the margin always lies inside some tile and clear
of its inner borders, and larger ones are left to the full-image pass. Without such a
detection the border box is kept, so an object that only tiles see (a very wide image whose
full-image pass is too coarse, or ``full_image=False``) is never filtered away.

The merge NMS uses the detector's own NMS IoU unless ``merge_iou`` is set, so a tiled run
differs from the plain one only by the tiles (the evaluation's 0.7, the service's 0.5).

Cost: one inference per tile plus one for the full image (7 for a 1024 x 768 image with
512 px tiles and 20% overlap), batched.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tsd.boxes import box_ioa
from tsd.infer.detector import Detector
from tsd.infer.postprocess import NMSConfig, batched_nms
from tsd.infer.types import Detections
from tsd.io import Image


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int = 512
    overlap: float = 0.2  # fraction of tile_size shared by neighbouring tiles
    full_image: bool = True
    merge_iou: float | None = None  # None: the detector's NMS IoU
    batch_size: int = 8
    # Drop tile detections within border_margin * tile_size px of an inner tile border
    # (truncated objects) if a non-border detection covers rescue_ioa of them; None keeps them
    # (plain SAHI-style merging).
    border_margin: float | None = 0.02
    rescue_ioa: float = 0.5

    def __post_init__(self) -> None:
        if self.tile_size < 32 or not 0.0 <= self.overlap < 1.0 or self.batch_size < 1:
            raise ValueError("invalid tiling configuration")
        if self.border_margin is not None and not 0.0 <= self.border_margin < 0.5:
            raise ValueError("invalid tiling configuration: border_margin")
        if not 0.0 < self.rescue_ioa <= 1.0:
            raise ValueError("invalid tiling configuration: rescue_ioa")
        if self.merge_iou is not None and not 0.0 < self.merge_iou <= 1.0:
            raise ValueError("invalid tiling configuration: merge_iou")


def inner_border_mask(
    boxes: np.ndarray, crop: tuple[int, int, int, int], width: int, height: int, margin: float
) -> np.ndarray:
    """``True`` for boxes (in crop coordinates) that stay ``margin`` px away from every crop
    border lying inside the ``width`` x ``height`` image; borders on the image edge don't count."""
    x0, y0, x1, y1 = crop
    keep = np.ones(len(boxes), dtype=bool)
    if x0 > 0:
        keep &= boxes[:, 0] > margin
    if y0 > 0:
        keep &= boxes[:, 1] > margin
    if x1 < width:
        keep &= boxes[:, 2] < (x1 - x0) - margin
    if y1 < height:
        keep &= boxes[:, 3] < (y1 - y0) - margin
    return keep


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    """Evenly spaced tile offsets along one axis: the first tile starts at 0, the last ends at
    ``length``, neighbours overlap by at least ``overlap * tile`` pixels."""
    if length <= tile:
        return [0]
    min_overlap = overlap * tile
    count = math.ceil((length - min_overlap) / (tile - min_overlap))
    count = max(count, 2)
    step = (length - tile) / (count - 1)
    return [round(i * step) for i in range(count)]


def tile_grid(
    width: int, height: int, tile: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """``(x0, y0, x1, y1)`` crops that cover the image."""
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in tile_starts(height, tile, overlap)
        for x in tile_starts(width, tile, overlap)
    ]


class TiledDetector:
    """Wraps a :class:`Detector`; same ``predict`` signature for one image at a time."""

    def __init__(self, detector: Detector, config: TilingConfig | None = None) -> None:
        self.detector = detector
        self.config = config or TilingConfig()

    @property
    def name(self) -> str:
        return f"{self.detector.name}+tiles{self.config.tile_size}"

    def tiles(self, image: Image) -> list[tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        if max(h, w) <= self.config.tile_size:
            return []  # the full-image pass already sees it at (at least) tile resolution
        return tile_grid(w, h, self.config.tile_size, self.config.overlap)

    def predict_one(self, image: Image, nms: NMSConfig | None = None) -> Detections:
        h, w = image.shape[:2]
        crops = self.tiles(image)
        views = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in crops]
        regions: list[tuple[int, int, int, int] | None] = list(crops)
        if self.config.full_image or not views:
            views.append(image)
            regions.append(None)  # the full-image pass: nothing is truncated
        margin = self.config.border_margin
        parts: list[Detections] = []
        touching: list[np.ndarray] = []  # per part: reaches an inner tile border
        size = self.config.batch_size
        for start in range(0, len(views), size):
            batch = self.detector.predict(views[start : start + size], nms)
            for det, crop in zip(batch, regions[start : start + size], strict=True):
                if crop is None:
                    parts.append(det)
                    touching.append(np.zeros(len(det), dtype=bool))
                    continue
                if margin is None:
                    touching.append(np.zeros(len(det), dtype=bool))
                else:
                    keep = inner_border_mask(det.boxes, crop, w, h, margin * self.config.tile_size)
                    touching.append(~keep)
                dx, dy = crop[0], crop[1]
                shift = np.array([dx, dy, dx, dy], dtype=np.float32)
                parts.append(Detections(det.boxes + shift, det.scores, det.class_ids))
        merged = Detections.concatenate(parts)
        if len(merged) == 0:
            return merged
        at_border = np.concatenate(touching)
        if at_border.any():
            merged = merged.select(~self._covered(merged, at_border))
        cfg = nms or self.detector.nms
        iou = self.config.merge_iou if self.config.merge_iou is not None else cfg.iou_threshold
        keep_idx = batched_nms(merged.boxes, merged.scores, merged.class_ids, iou)
        return merged.select(keep_idx[: cfg.max_det])

    def _covered(self, dets: Detections, at_border: np.ndarray) -> np.ndarray:
        """Border detections that a same-class non-border detection covers (IoA >= rescue)."""
        clear = np.flatnonzero(~at_border)
        out = np.zeros(len(dets), dtype=bool)
        if len(clear) == 0:
            return out
        border = np.flatnonzero(at_border)
        ioa = box_ioa(dets.boxes[border], dets.boxes[clear])
        same = dets.class_ids[border][:, None] == dets.class_ids[clear][None, :]
        out[border] = ((ioa >= self.config.rescue_ioa) & same).any(axis=1)
        return out

    def predict(self, images: Sequence[Image], nms: NMSConfig | None = None) -> list[Detections]:
        return [self.predict_one(image, nms) for image in images]
