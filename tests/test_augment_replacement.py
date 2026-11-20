"""Tests for the replacement (copy-paste) augmentation and its photometric jitter."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from tsd.augment.photometric import PhotometricJitter, illumination_gain, jitter_bgr
from tsd.augment.replacement import (
    ReplacementAugment,
    alpha_bbox,
    alpha_blend,
    class_sampling_weights,
)
from tsd.augment.templates import (
    SignTemplate,
    load_templates,
    render_speed_limit,
    render_warning,
    synthetic_templates,
)
from tsd.boxes import box_ioa, box_iou

W, H = 256, 192


def _image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(30, 220, (H, W, 3), dtype=np.uint8)


def _square_template(class_id: int, value: int = 200, size: int = 32) -> SignTemplate:
    bgra = np.full((size, size, 4), value, dtype=np.uint8)
    bgra[..., 3] = 255
    return SignTemplate(class_id, bgra, f"c{class_id}")


def test_sampling_weights_are_inverse_frequency() -> None:
    w = class_sampling_weights({0: 1000, 1: 10, 2: 100}, [0, 1, 2])
    assert pytest.approx(sum(w.values())) == 1.0
    assert w[1] > w[2] > w[0]
    uniform = class_sampling_weights({0: 1000, 1: 10}, [0, 1], power=0.0)
    assert uniform[0] == pytest.approx(uniform[1])
    only = class_sampling_weights({0: 5}, [3])  # a class with templates but no instances yet
    assert only == {3: 1.0}
    with pytest.raises(ValueError, match="no classes"):
        class_sampling_weights({0: 1}, [])


def test_alpha_blend_transparent_opaque_and_border_clipping() -> None:
    img = _image()
    before = img.copy()
    patch = np.full((20, 20, 3), 7.0, dtype=np.float32)
    alpha_blend(img, patch, np.zeros((20, 20), np.float32), 10, 10)
    np.testing.assert_array_equal(img, before)
    alpha_blend(img, patch, np.ones((20, 20), np.float32), 10, 10)
    assert (img[10:30, 10:30] == 7).all()
    np.testing.assert_array_equal(img[30:, :], before[30:, :])
    alpha_blend(img, patch, np.ones((20, 20), np.float32), W - 5, H - 5)  # mostly outside
    assert (img[H - 5 :, W - 5 :] == 7).all()
    snapshot = img.copy()
    alpha_blend(img, patch, np.ones((20, 20), np.float32), W + 5, 0)  # fully outside: no-op
    np.testing.assert_array_equal(img, snapshot)


def test_alpha_bbox_is_tight_and_clipped() -> None:
    alpha = np.zeros((10, 10), np.float32)
    alpha[2:5, 3:9] = 1.0
    assert alpha_bbox(alpha, 100, 50, W, H) == (103.0, 52.0, 109.0, 55.0)
    assert alpha_bbox(alpha, W - 5, 0, W, H) == (W - 2.0, 2.0, float(W), 5.0)
    assert alpha_bbox(np.zeros((4, 4), np.float32), 0, 0, W, H) is None


def test_illumination_gain_is_a_mean_preserving_ramp() -> None:
    g = illumination_gain(40, 60, strength=0.6, angle=0.0)
    assert g.mean() == pytest.approx(1.0, abs=1e-4)
    assert g[:, 0].mean() == pytest.approx(0.7, abs=0.01)
    assert g[:, -1].mean() == pytest.approx(1.3, abs=0.01)
    assert (np.diff(g, axis=1) > 0).all()  # monotone along the ramp direction
    assert np.allclose(np.diff(g, axis=0), 0)  # constant across it
    lit = illumination_gain(40, 40, strength=0.0, angle=0.0, highlight=0.3)
    assert lit[20, 20] > lit[0, 0] + 0.2


def test_jitter_none_is_identity_and_jitter_is_seeded() -> None:
    rgb = _image()[:32, :32].astype(np.float32)
    np.testing.assert_array_equal(
        jitter_bgr(rgb, PhotometricJitter.none(), np.random.default_rng(0)), rgb
    )
    a = jitter_bgr(rgb, PhotometricJitter(), np.random.default_rng(3))
    b = jitter_bgr(rgb, PhotometricJitter(), np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0
    assert a.max() <= 255
    assert not np.array_equal(a, rgb)


def test_replace_updates_class_and_box_and_only_touches_the_box() -> None:
    img = _image()
    boxes = np.array([[40.0, 40.0, 72.0, 72.0], [150.0, 60.0, 182.0, 92.0]])
    cls = np.array([0, 0])
    aug = ReplacementAugment(
        templates={1: [_square_template(1)], 2: [_square_template(2)]},
        class_counts={0: 500, 1: 5, 2: 50},
        p_replace=1.0,
        p_add=0.0,
        cover_scale=(1.05, 1.05),
        jitter=PhotometricJitter.none(),
        match_brightness=0.0,
        feather=False,
    )
    res = aug(img, boxes, cls, np.random.default_rng(0))
    assert len(res.boxes) == 2
    assert set(res.class_ids.tolist()) <= {1, 2}
    assert [op.kind for op in res.ops] == ["replace", "replace"]
    assert (np.diag(box_iou(res.boxes, boxes)) > 0.85).all()
    changed = np.any(res.image != img, axis=2)
    allowed = np.zeros_like(changed)
    for x1, y1, x2, y2 in res.boxes.astype(int):
        allowed[y1:y2, x1:x2] = True
    assert not (changed & ~allowed).any()
    for x1, y1, x2, y2 in res.boxes.astype(int):  # opaque template fully replaced the pixels
        assert (res.image[y1:y2, x1:x2] == 200).all()
    np.testing.assert_array_equal(img, _image())  # input not modified


def test_replace_skips_incompatible_aspect_ratio() -> None:
    aug = ReplacementAugment(
        templates={1: [_square_template(1)]}, class_counts={0: 10, 1: 1}, p_replace=1.0, p_add=0.0
    )
    tall = np.array([[40.0, 20.0, 50.0, 80.0]])
    res = aug(_image(), tall, np.array([0]), np.random.default_rng(0))
    assert res.ops == ()
    np.testing.assert_array_equal(res.class_ids, [0])


def test_add_pastes_non_overlapping_signs_inside_the_image() -> None:
    boxes = np.array([[100.0, 80.0, 130.0, 110.0]])
    aug = ReplacementAugment(
        templates=synthetic_templates({1: "pl15", 2: "w57"}),
        class_counts={0: 100, 1: 1, 2: 10},
        p_replace=0.0,
        p_add=1.0,
        max_add=3,
        max_overlap=0.0,
    )
    for seed in range(20):
        res = aug(_image(seed), boxes, np.array([0]), np.random.default_rng(seed))
        assert 1 <= len(res.boxes) <= 4
        np.testing.assert_array_equal(res.boxes[0], boxes[0])
        new = res.boxes[1:]
        assert (new[:, :2] >= 0).all()
        assert (new[:, 2] <= W).all()
        assert (new[:, 3] <= H).all()
        assert box_ioa(new, boxes).max(initial=0.0) == 0.0
        assert set(res.class_ids[1:].tolist()) <= {1, 2}
        assert all(op.kind == "add" for op in res.ops)


def test_same_seed_same_result() -> None:
    aug = ReplacementAugment(
        templates=synthetic_templates({1: "pl15", 2: "w57"}), class_counts={0: 9, 1: 1, 2: 3}
    )
    boxes = np.array([[40.0, 40.0, 80.0, 80.0]])
    a = aug(_image(), boxes, np.array([0]), np.random.default_rng(11))
    b = aug(_image(), boxes, np.array([0]), np.random.default_rng(11))
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_array_equal(a.boxes, b.boxes)
    assert a.ops == b.ops


def test_rare_classes_are_pasted_most_often() -> None:
    aug = ReplacementAugment(
        templates={c: [_square_template(c)] for c in (0, 1, 2)},
        class_counts={0: 1000, 1: 10, 2: 100},
    )
    rng = np.random.default_rng(0)
    counts = Counter(aug.sample_class(rng) for _ in range(3000))
    assert counts[1] > counts[2] > counts[0]
    assert aug.sample_class(rng, exclude=1) in {0, 2}


def test_brightness_matching_moves_template_towards_scene() -> None:
    dark = np.full((H, W, 3), 40, dtype=np.uint8)
    box = np.array([[60.0, 60.0, 92.0, 92.0]])
    common = {
        "templates": {1: [_square_template(1, value=220)]},
        "class_counts": {0: 10, 1: 1},
        "p_replace": 1.0,
        "p_add": 0.0,
        "jitter": PhotometricJitter.none(),
        "feather": False,
    }
    plain = ReplacementAugment(match_brightness=0.0, **common)  # type: ignore[arg-type]
    matched = ReplacementAugment(match_brightness=0.5, **common)  # type: ignore[arg-type]
    a = plain(dark, box, np.array([0]), np.random.default_rng(0)).image[65:85, 65:85].mean()
    b = matched(dark, box, np.array([0]), np.random.default_rng(0)).image[65:85, 65:85].mean()
    assert a == pytest.approx(220, abs=1)
    assert 40 < b < 220


def test_synthetic_templates_have_transparent_corners() -> None:
    disc = render_speed_limit("15")
    tri = render_warning()
    for t in (disc, tri):
        assert t.shape == (128, 128, 4)
        assert t[0, 0, 3] == 0
        assert t[64, 64, 3] == 255
    with pytest.raises(ValueError, match="BGRA"):
        SignTemplate(0, disc[..., :3])


def test_load_templates_matches_class_names(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "pl15.png"), render_speed_limit("15"))
    cv2.imwrite(str(tmp_path / "pl15_2.png"), render_speed_limit("15", 64))
    cv2.imwrite(str(tmp_path / "w57.png"), render_warning()[..., :3])  # no alpha channel
    cv2.imwrite(str(tmp_path / "unknown.png"), render_warning())
    out = load_templates(tmp_path, ["pl15", "pl80", "w57"])
    assert sorted(out) == [0, 2]
    assert len(out[0]) == 2
    assert (out[2][0].bgra[..., 3] == 255).all()


def _disc_scene() -> tuple[np.ndarray, np.ndarray]:
    img = np.full((H, W, 3), 90, dtype=np.uint8)
    cv2.circle(img, (80, 80), 24, (0, 255, 0), -1)  # an old disc sign in a unique colour
    return img, np.array([[56.0, 56.0, 105.0, 105.0]])


def _shape_aug(p_replace: float = 1.0, **kwargs: object) -> ReplacementAugment:
    templates = {
        0: [SignTemplate(0, render_speed_limit("15"), "pl15", "circle")],
        1: [SignTemplate(1, render_warning(), "w57", "triangle")],
        2: [SignTemplate(2, render_speed_limit("30"), "pl30", "circle")],
    }
    params: dict[str, object] = {
        "templates": templates, "class_counts": {0: 500, 1: 3, 2: 20}, "p_replace": p_replace,
        "p_add": 0.0, "jitter": PhotometricJitter.none(), "match_brightness": 0.0,
    }  # fmt: skip
    params.update(kwargs)
    return ReplacementAugment(**params)  # type: ignore[arg-type]


def test_a_disc_is_only_replaced_by_a_disc_that_covers_it() -> None:
    img, box = _disc_scene()
    old = np.all(img == (0, 255, 0), axis=2)
    for seed in range(30):
        res = _shape_aug()(img, box, np.array([0]), np.random.default_rng(seed))
        if not res.ops:
            continue
        assert res.class_ids.tolist() == [2]  # never the triangle class 1
        still_green = np.all(res.image == (0, 255, 0), axis=2) & old
        assert still_green.sum() <= 0.01 * old.sum()


def test_replacement_is_skipped_when_the_silhouette_would_not_cover() -> None:
    img, box = _disc_scene()
    # only a triangle is offered for the disc's class: no same-shape target -> no replacement
    aug = _shape_aug(
        templates={0: [SignTemplate(0, render_speed_limit("15"), "pl15", "circle")],
                   1: [SignTemplate(1, render_warning(), "w57", "triangle")]},
        class_counts={0: 500, 1: 3},
    )  # fmt: skip
    for seed in range(20):
        assert aug(img, box, np.array([0]), np.random.default_rng(seed)).ops == ()
    from tsd.augment.replacement import coverage
    from tsd.augment.templates import silhouette_mask

    disc = silhouette_mask("circle", 49, 49)
    tri = cv2.resize(render_warning(), (49, 49))[..., 3].astype(np.float32) / 255
    assert coverage(tri, 0, 0, disc, 0, 0) < 0.8  # what a disc-to-triangle swap would leave


def test_templates_with_transparent_margins_are_cropped_before_stretching() -> None:
    """render_warning's triangle leaves transparent rows on its 128 px canvas (8 above, 7 below);
    stretched uncropped to a box it does not cover a triangle filling that box, and the coverage
    check refused most triangle replacements."""
    tri = render_warning()
    assert (tri[:8, :, 3] == 0).all()
    img = np.full((H, W, 3), 90, dtype=np.uint8)
    box = np.array([[56.0, 56.0, 105.0, 105.0]])
    from tsd.augment.templates import silhouette_mask

    old = np.zeros((H, W), dtype=bool)
    old[56:105, 56:105] = silhouette_mask("triangle", 49, 49) > 0
    img[old] = (0, 255, 0)
    aug = _shape_aug(
        templates={0: [SignTemplate(0, tri, "w1", "triangle")],
                   1: [SignTemplate(1, render_warning(), "w2", "triangle")]},
        class_counts={0: 500, 1: 3},
    )  # fmt: skip
    replaced = 0
    for seed in range(20):
        res = aug(img, box, np.array([0]), np.random.default_rng(seed))
        if res.ops:
            replaced += 1
            still_green = np.all(res.image == (0, 255, 0), axis=2) & old
            assert still_green.sum() <= 0.01 * old.sum()
    assert replaced == 20


def test_rare_classes_are_not_overwritten() -> None:
    img, box = _disc_scene()
    aug = _shape_aug(p_replace=0.5)
    rare = sum(
        bool(aug(img, box, np.array([2]), np.random.default_rng(s)).ops) for s in range(1000)
    )
    common = sum(
        bool(aug(img, box, np.array([0]), np.random.default_rng(s)).ops) for s in range(1000)
    )
    assert rare < 50  # class 2 (20 instances) is never replaced by the more common class 0
    assert 400 < common < 600  # class 0 (500) is replaced at about p_replace


def test_opaque_canvas_edges_are_feathered() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    aug = ReplacementAugment(
        templates={1: [_square_template(1)]}, class_counts={0: 10, 1: 1}, p_replace=1.0,
        p_add=0.0, jitter=PhotometricJitter.none(), match_brightness=0.0, cover_scale=(1.0, 1.0),
    )  # fmt: skip
    res = aug(img, np.array([[16.0, 16.0, 48.0, 48.0]]), np.array([0]), np.random.default_rng(0))
    assert res.ops
    assert 0 < int(res.image[32, 16, 0]) < 200  # the edge column is blended, not a hard cut
