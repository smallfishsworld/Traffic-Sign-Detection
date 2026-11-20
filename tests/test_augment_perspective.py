"""Geometry tests for the perspective augmentation (property-style, seeded)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tsd.augment.perspective import (
    BoxFilter,
    RandomPerspective,
    apply_homography,
    clip_polygon,
    corner_homography,
    corners,
    polygon_area,
    transform_points,
    warp_boxes,
)

W, H = 320, 240


def _random_boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    xy = rng.uniform([0, 0], [W - 40, H - 40], size=(n, 2))
    wh = rng.uniform(8, 40, size=(n, 2))
    return np.concatenate([xy, xy + wh], axis=1)


def _scene(boxes: np.ndarray) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes.round().astype(int):
        img[y1:y2, x1:x2] = 255
    return img


def test_identity_is_a_no_op() -> None:
    rng = np.random.default_rng(0)
    boxes = _random_boxes(rng, 10)
    image = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    out_img, out_boxes, out_cls = apply_homography(image, boxes, np.arange(10), np.eye(3))
    np.testing.assert_array_equal(out_img, image)
    np.testing.assert_allclose(out_boxes, boxes, atol=1e-9)
    np.testing.assert_array_equal(out_cls, np.arange(10))


def test_known_affine_homography_maps_corners_exactly() -> None:
    s, tx, ty = 1.5, 12.0, -7.0
    Hm = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]], dtype=np.float64)
    boxes = np.array([[10.0, 20.0, 50.0, 60.0], [100.0, 80.0, 120.0, 140.0]])
    out, keep = warp_boxes(boxes, Hm, 10_000, 10_000)
    assert keep.all()
    np.testing.assert_allclose(out, boxes * s + [tx, ty, tx, ty], atol=1e-9)


def test_projective_homography_uses_enclosing_box_of_projected_corners() -> None:
    Hm = np.array([[1.1, 0.05, 3.0], [-0.02, 0.95, 4.0], [4e-4, -2e-4, 1.0]])
    box = np.array([[30.0, 40.0, 90.0, 100.0]])
    expected = []
    for x, y in [(30, 40), (90, 40), (90, 100), (30, 100)]:
        u, v, w = Hm @ np.array([x, y, 1.0])
        expected.append((u / w, v / w))
    exp = np.array(expected)
    out, keep = warp_boxes(box, Hm, 10_000, 10_000)
    assert keep.all()
    np.testing.assert_allclose(
        out[0], [exp[:, 0].min(), exp[:, 1].min(), exp[:, 0].max(), exp[:, 1].max()], atol=1e-9
    )


def test_affine_round_trip_recovers_boxes() -> None:
    Hm = np.array([[0.8, 0, 20], [0, 1.25, -5], [0, 0, 1]], dtype=np.float64)
    boxes = np.array([[40.0, 30.0, 80.0, 70.0]])
    fwd, _ = warp_boxes(boxes, Hm, 10_000, 10_000)
    back, _ = warp_boxes(fwd, np.linalg.inv(Hm), 10_000, 10_000)
    np.testing.assert_allclose(back, boxes, atol=1e-9)


@pytest.mark.parametrize("seed", range(25))
def test_random_homography_properties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    boxes = _random_boxes(rng, 12)
    Hm = corner_homography(W, H, 0.15, rng)
    out, keep = warp_boxes(boxes, Hm, W, H, BoxFilter(min_size=2, min_visibility=0.25))
    assert out.shape == (int(keep.sum()), 4)
    # Kept boxes are valid and inside the image.
    assert (out[:, 0] >= 0).all()
    assert (out[:, 1] >= 0).all()
    assert (out[:, 2] <= W).all()
    assert (out[:, 3] <= H).all()
    assert (out[:, 2] - out[:, 0] >= 2).all()
    assert (out[:, 3] - out[:, 1] >= 2).all()
    # Every projected corner of a kept box that is inside the image lies inside its output box.
    pts, _ = transform_points(corners(boxes[keep]).reshape(-1, 2), Hm)
    pts = pts.reshape(-1, 4, 2)
    in_image = (pts >= 0).all(axis=2) & (pts <= [W, H]).all(axis=2)
    lo = np.broadcast_to(out[:, None, :2], pts.shape)
    hi = np.broadcast_to(out[:, None, 2:], pts.shape)
    assert (pts[in_image] >= lo[in_image] - 1e-6).all()
    assert (pts[in_image] <= hi[in_image] + 1e-6).all()


@pytest.mark.parametrize("seed", range(10))
def test_warped_pixels_match_warped_boxes(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    box = _random_boxes(rng, 1) + np.full(4, 40.0)
    box = np.clip(box, 0, [W, H, W, H])
    Hm = corner_homography(W, H, 0.1, rng)
    warped = cv2.warpPerspective(_scene(box.round()), Hm, (W, H), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(warped[..., 0] > 127)
    out, keep = warp_boxes(box.round(), Hm, W, H, BoxFilter(min_visibility=0.0))
    if not keep.any():
        assert len(xs) == 0
        return
    pixel_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
    np.testing.assert_allclose(out[0], pixel_box, atol=1.5)


@pytest.mark.parametrize("seed", range(20))
def test_boxes_cut_by_the_border_stay_tight(seed: int) -> None:
    """A rotated box that crosses the image border: the label must match the warped pixels that
    are still in the image, not the enclosing box clipped to the image (which can run along the
    border far beyond them)."""
    rng = np.random.default_rng(300 + seed)
    angle = np.deg2rad(rng.uniform(15, 40))
    c, s_ = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    move = np.array([[1, 0, W - 10.0], [0, 1, rng.uniform(60, 150)], [0, 0, 1]])
    centre = np.array([[1, 0, -20.0], [0, 1, -20.0], [0, 0, 1]])
    Hm = move @ rot @ centre  # a 40 x 40 box rotated about its centre, centred 10 px from x = W
    box = np.array([[0.0, 0.0, 40.0, 40.0]])
    src = np.zeros((40, 40, 3), dtype=np.uint8) + 255
    warped = cv2.warpPerspective(src, Hm, (W, H), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(warped[..., 0] > 127)
    out, keep = warp_boxes(box, Hm, W, H, BoxFilter(min_visibility=0.0, max_aspect=1e9))
    assert keep.all()
    pixel_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
    np.testing.assert_allclose(out[0], pixel_box, atol=1.5)


def test_clip_polygon_and_area() -> None:
    square = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    assert polygon_area(square) == pytest.approx(400.0)
    part = clip_polygon(square, 100, 100)
    assert polygon_area(part) == pytest.approx(100.0)
    np.testing.assert_allclose(part.min(axis=0), [0, 0])
    np.testing.assert_allclose(part.max(axis=0), [10, 10])
    assert len(clip_polygon(square - 50, 100, 100)) == 0


def test_out_of_frame_boxes_are_dropped() -> None:
    shift = np.array([[1, 0, W * 0.9], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    boxes = np.array(
        [[0.0, 10.0, 20.0, 30.0], [10.0, 10.0, 50.0, 30.0], [200.0, 10.0, 210.0, 30.0]]
    )
    out, keep = warp_boxes(boxes, shift, W, H, BoxFilter(min_visibility=0.25))
    # Box 0 lands at x in [288, 308]: 100% visible. Box 1 at [298, 338]: 22/40 = 55% visible.
    # Box 2 at [488, 498]: entirely outside.
    np.testing.assert_array_equal(keep, [True, True, False])
    np.testing.assert_allclose(out[1], [298, 10, 320, 30])
    _, strict = warp_boxes(boxes, shift, W, H, BoxFilter(min_visibility=0.6))
    np.testing.assert_array_equal(strict, [True, False, False])


def test_points_behind_camera_are_dropped() -> None:
    Hm = np.array([[1, 0, 0], [0, 1, 0], [-0.02, 0, 1]], dtype=np.float64)  # w' <= 0 for x >= 50
    boxes = np.array([[10.0, 10.0, 20.0, 20.0], [40.0, 10.0, 60.0, 20.0]])
    _, keep = warp_boxes(boxes, Hm, 10_000, 10_000)
    np.testing.assert_array_equal(keep, [True, False])


def test_random_perspective_probability_and_determinism() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    boxes = _random_boxes(rng, 5)
    cls = np.arange(5)
    off = RandomPerspective(p=0.0)
    same_img, same_boxes, _ = off(img, boxes, cls, np.random.default_rng(1))
    assert same_img is img
    np.testing.assert_array_equal(same_boxes, boxes)
    on = RandomPerspective(p=1.0, max_shift=0.1)
    a = on(img, boxes, cls, np.random.default_rng(7))
    b = on(img, boxes, cls, np.random.default_rng(7))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], img)


def test_empty_boxes() -> None:
    out, keep = warp_boxes(np.zeros((0, 4)), np.eye(3), W, H)
    assert out.shape == (0, 4)
    assert keep.shape == (0,)
