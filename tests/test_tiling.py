"""Tiled inference: grid coverage, coordinate mapping, merging, and the class-offset fix."""

from __future__ import annotations

import numpy as np
import pytest

from tsd.infer import Detector
from tsd.infer.postprocess import batched_nms
from tsd.infer.tiling import (
    TiledDetector,
    TilingConfig,
    inner_border_mask,
    tile_grid,
    tile_starts,
)


@pytest.mark.parametrize(
    ("length", "tile", "overlap"),
    [(1024, 512, 0.2), (768, 512, 0.2), (8100, 640, 0.25), (513, 512, 0.0)],
)
def test_tiles_cover_the_axis_with_the_requested_overlap(
    length: int, tile: int, overlap: float
) -> None:
    starts = tile_starts(length, tile, overlap)
    assert starts[0] == 0
    assert starts[-1] + tile == length
    gaps = np.diff(starts)
    assert (tile - gaps >= overlap * tile - 1).all()  # neighbours share >= overlap * tile px
    assert tile_starts(300, 512, 0.2) == [0]


def test_grid_is_row_major_and_clipped() -> None:
    grid = tile_grid(1024, 768, 512, 0.2)
    assert len(grid) == 3 * 2
    assert grid[0] == (0, 0, 512, 512)
    assert grid[-1] == (512, 256, 1024, 768)
    assert tile_grid(400, 300, 512, 0.2) == [(0, 0, 400, 300)]
    with pytest.raises(ValueError, match="invalid"):
        TilingConfig(overlap=1.0)


class SpotBackend:
    """Detects a bright square anywhere in the (letterboxed) input, if it is at least 12 px."""

    name = "spot"

    def __init__(self) -> None:
        self.batches: list[int] = []

    def infer(self, batch: np.ndarray) -> np.ndarray:
        self.batches.append(batch.shape[0])
        raw = np.zeros((batch.shape[0], 4 + 2, 4), np.float32)
        for b, img in enumerate(batch):
            mask = img[0] > 0.99
            ys, xs = np.nonzero(mask)
            if len(xs) == 0 or xs.max() - xs.min() + 1 < 12:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            raw[b, :4, 0] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
            raw[b, 4 + 1, 0] = 0.9
        return raw


def test_small_object_is_found_only_with_tiles_and_mapped_back() -> None:
    image = np.zeros((1536, 2048, 3), np.uint8)
    image[1000:1016, 1500:1516] = 255  # 16 px: 5 px in the full-image letterbox, 16+ in a tile
    backend = SpotBackend()
    detector = Detector(backend, ["a", "b"])
    assert len(detector.predict([image])[0]) == 0
    tiled = TiledDetector(detector, TilingConfig(tile_size=512, overlap=0.2, batch_size=8))
    (det,) = tiled.predict([image])
    assert len(det) == 1  # duplicates from overlapping tiles are merged
    np.testing.assert_allclose(det.boxes[0], [1500, 1000, 1516, 1016], atol=1.0)
    assert det.class_ids[0] == 1
    n_tiles = len(tile_grid(2048, 1536, 512, 0.2))
    assert sum(backend.batches) == 1 + n_tiles + 1  # untiled call, tiles, full-image pass
    assert tiled.name == "spot+tiles512"


def test_small_images_skip_tiling() -> None:
    backend = SpotBackend()
    tiled = TiledDetector(Detector(backend, ["a", "b"]), TilingConfig(tile_size=512))
    tiled.predict([np.zeros((400, 500, 3), np.uint8)])
    assert backend.batches == [1]


def test_class_offset_grows_with_coordinates() -> None:
    # Different classes in a > 7680 px square image: with a fixed 7680 offset the class-1 box
    # near the origin lands exactly on the class-0 box near (7700, 7700) and is suppressed.
    boxes = np.array([[7700.0, 7700, 8000, 8000], [20, 20, 320, 320]])
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 1])
    assert sorted(batched_nms(boxes, scores, classes, 0.5).tolist()) == [0, 1]
    assert batched_nms(boxes, scores, classes, 0.5, class_offset=7680.0).tolist() == [0]


def test_truncated_boxes_at_inner_tile_borders_are_dropped() -> None:
    # A 200 px object crosses the border of the first tile, which reports the visible part
    # with the same score. Plain merging keeps that truncated box (it comes first) and
    # suppresses the complete ones (IoU 0.66): the failure seen on large stop signs.
    image = np.zeros((1024, 1536, 3), np.uint8)
    image[300:500, 380:580] = 255
    detector = Detector(SpotBackend(), ["a", "b"])
    plain = TiledDetector(detector, TilingConfig(tile_size=512, border_margin=None))
    (naive,) = plain.predict([image])
    assert len(naive) == 1
    assert naive.boxes[0][2] == pytest.approx(512, abs=1.0)  # truncated at the tile border
    filtered = TiledDetector(detector, TilingConfig(tile_size=512))
    (det,) = filtered.predict([image])
    assert len(det) == 1
    np.testing.assert_allclose(det.boxes[0], [380, 300, 580, 500], atol=3.0)


def test_inner_border_mask_ignores_image_edges() -> None:
    boxes = np.array([[0.0, 0, 50, 50], [470, 100, 512, 150], [100, 100, 200, 200]])
    # crop at the image's top-left corner: its left/top borders are image edges
    corner = inner_border_mask(boxes, (0, 0, 512, 512), 1536, 1024, 10.0)
    assert corner.tolist() == [True, False, True]
    # the same crop covering the whole image width: the right border is an image edge too
    assert inner_border_mask(boxes, (0, 0, 512, 512), 512, 1024, 10.0).tolist() == [True] * 3
    inner = inner_border_mask(boxes, (400, 200, 912, 712), 1536, 1024, 10.0)
    assert inner.tolist() == [False, False, True]
    with pytest.raises(ValueError, match="border_margin"):
        TilingConfig(border_margin=0.5)


class FixedBackend:
    """Returns the same two class-0 boxes (in input pixels) for every image of a batch."""

    name = "fixed"

    def infer(self, batch: np.ndarray) -> np.ndarray:
        raw = np.zeros((batch.shape[0], 4 + 1, 2), np.float32)
        # (cx, cy, w, h): [100, 100, 200, 200] and [130, 100, 230, 200], IoU 0.54
        raw[:, :4, 0] = [150, 150, 100, 100]
        raw[:, :4, 1] = [180, 150, 100, 100]
        raw[:, 4, :] = [0.9, 0.8]
        return raw


def test_merge_nms_uses_the_detectors_iou_unless_set() -> None:
    from tsd.infer import NMSConfig

    image = np.zeros((512, 512, 3), np.uint8)  # no tiles: only the full-image pass
    loose = NMSConfig(conf_threshold=0.1, iou_threshold=0.7)
    detector = Detector(FixedBackend(), ["a"], nms=loose)
    assert len(detector.predict([image])[0]) == 2  # IoU 0.54 < 0.7: both survive
    (same,) = TiledDetector(detector, TilingConfig(tile_size=512)).predict([image])
    assert len(same) == 2  # the merge adds no extra suppression
    (strict,) = TiledDetector(detector, TilingConfig(tile_size=512, merge_iou=0.5)).predict([image])
    assert len(strict) == 1
    with pytest.raises(ValueError, match="merge_iou"):
        TilingConfig(merge_iou=0.0)


def test_object_seen_only_by_tiles_survives_the_border_filter() -> None:
    # 8192 px wide: the full-image pass shrinks the 140 px object below the detector's 12 px.
    # It lies in the overlap band of tiles 0 and 1 and touches an inner border of both, so an
    # unconditional border filter would drop every detection of it.
    image = np.zeros((512, 8192, 3), np.uint8)
    image[200:340, 390:530] = 255
    detector = Detector(SpotBackend(), ["a", "b"])
    assert len(detector.predict([image])[0]) == 0
    (det,) = TiledDetector(detector, TilingConfig(tile_size=512)).predict([image])
    assert len(det) == 1
    x1, _, x2, _ = det.boxes[0]
    assert min(x2, 530) - max(x1, 390) >= 0.75 * 140
