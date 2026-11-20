"""COCO-style AP: hand-computed cases, randomized agreement with pycocotools, bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import has_module
from tsd.eval.metrics import (
    AREA_RANGES,
    ImagePredictions,
    ImageTruth,
    bootstrap,
    build_cache,
    match_image,
    summarize,
)


def _truth(
    boxes: list[list[float]], classes: list[int], crowd: list[bool] | None = None
) -> ImageTruth:
    return ImageTruth(
        np.asarray(boxes, np.float64).reshape(-1, 4),
        np.asarray(classes, np.int64),
        np.asarray(crowd or [False] * len(classes), bool),
    )


def _preds(boxes: list[list[float]], scores: list[float], classes: list[int]) -> ImagePredictions:
    return ImagePredictions(
        np.asarray(boxes, np.float64).reshape(-1, 4),
        np.asarray(scores, np.float64),
        np.asarray(classes, np.int64),
    )


def test_perfect_detection_scores_one() -> None:
    cache = build_cache(
        [_truth([[10, 10, 60, 60]], [0])], [_preds([[10, 10, 60, 60]], [0.9], [0])], ["a"]
    )
    s = summarize(cache)
    assert s.map50 == pytest.approx(1.0)
    assert s.map50_95 == pytest.approx(1.0)
    assert s.ar100 == pytest.approx(1.0)


def test_higher_scored_false_positive_halves_ap() -> None:
    truth = _truth([[10, 10, 60, 60]], [0])
    preds = _preds([[200, 200, 250, 250], [10, 10, 60, 60]], [0.9, 0.8], [0, 0])
    s = summarize(build_cache([truth], [preds], ["a"]))
    assert s.map50 == pytest.approx(0.5)  # precision 0.5 at every recall level
    assert s.per_class["a"]["ar100"] == pytest.approx(1.0)


def test_crowd_region_absorbs_detections_and_is_never_missed() -> None:
    truth = _truth([[10, 10, 60, 60], [100, 100, 300, 300]], [0, 0], [False, True])
    preds = _preds(
        [[10, 10, 60, 60], [120, 120, 160, 160], [200, 200, 280, 280]], [0.9, 0.95, 0.97], [0, 0, 0]
    )
    s = summarize(build_cache([truth], [preds], ["a"]))
    assert s.map50_95 == pytest.approx(1.0)  # both detections inside the crowd box are ignored


def test_classes_without_ground_truth_are_skipped_and_max_dets_applies() -> None:
    truth = _truth([[0, 0, 50, 50]], [0])
    preds = _preds([[0, 0, 50, 50], [5, 5, 40, 40]], [0.5, 0.9], [0, 1])  # class 1 has no GT
    s = summarize(build_cache([truth], [preds], ["a", "b"]))
    assert list(s.per_class) == ["a"]
    assert s.map50 == pytest.approx(1.0)
    many = _preds([[300, 300, 310, 310]] * 3 + [[0, 0, 50, 50]], [0.9, 0.8, 0.7, 0.1], [0] * 4)
    capped = summarize(build_cache([truth], [many], ["a"], max_dets=3))
    assert capped.map50 == 0.0  # the true positive is the 4th detection
    rec = match_image(truth.boxes, truth.crowd, many.boxes, many.scores, AREA_RANGES["all"], 3)
    assert rec is not None
    assert rec.matched.shape == (10, 3)


def _scenario(
    seed: int, n_images: int = 30, n_classes: int = 3
) -> tuple[list[ImageTruth], list[ImagePredictions]]:
    rng = np.random.default_rng(seed)
    truths, preds = [], []
    for _ in range(n_images):
        n = int(rng.integers(0, 6))
        xy = rng.uniform(0, 500, (n, 2))
        wh = np.exp(rng.uniform(np.log(4), np.log(300), (n, 2)))
        gt = np.concatenate([xy, xy + wh], axis=1)
        cls = rng.integers(0, n_classes, n)
        crowd = rng.random(n) < 0.15
        det_boxes, det_scores, det_cls = [], [], []
        for b, c, w in zip(gt, cls, wh, strict=True):
            for _ in range(int(rng.choice([0, 1, 1, 1, 2]))):
                jitter = rng.normal(0, 0.08, 4) * np.concatenate([w, w])
                det_boxes.append(b + jitter)
                det_scores.append(rng.uniform(0.05, 1.0))
                det_cls.append(c if rng.random() < 0.9 else rng.integers(0, n_classes))
        for _ in range(int(rng.integers(0, 4))):  # false positives
            p = rng.uniform(0, 500, 2)
            det_boxes.append(
                np.concatenate([p, p + np.exp(rng.uniform(np.log(4), np.log(200), 2))])
            )
            det_scores.append(rng.uniform(0.0, 0.8))
            det_cls.append(rng.integers(0, n_classes))
        truths.append(ImageTruth(gt, cls.astype(np.int64), crowd))
        preds.append(
            ImagePredictions(
                np.asarray(det_boxes, np.float64).reshape(-1, 4),
                np.asarray(det_scores, np.float64),
                np.asarray(det_cls, np.int64),
            )
        )
    return truths, preds


def _pycocotools_differences(seed: int) -> dict[str, float]:
    """|ours - pycocotools| per metric on random scenario ``seed`` (0 if both are NaN, inf if
    only one is)."""
    from tsd.eval.runner import pycocotools_stats

    truths, preds = _scenario(seed)
    names = ["a", "b", "c"]
    cache = build_cache(truths, preds, names)
    ref = pycocotools_stats(truths, preds, names)
    s = summarize(cache)
    ours = {
        "map50_95": s.map50_95,
        "map50": s.map50,
        "map75": s.map75,
        "ar100": s.ar100,
        **{f"map_{a}": summarize(cache, a).map50_95 for a in ("small", "medium", "large")},
    }
    out = {}
    for key, value in ref.items():
        if np.isnan(value) or np.isnan(ours[key]):
            out[key] = 0.0 if np.isnan(value) and np.isnan(ours[key]) else float("inf")
        else:
            out[key] = abs(ours[key] - value)
    return out


@pytest.mark.skipif(not has_module("pycocotools"), reason="pycocotools not installed")
@pytest.mark.parametrize("seed", range(6))  # `python tests/test_eval_metrics.py 200` runs 200
def test_matches_pycocotools(seed: int) -> None:
    differences = _pycocotools_differences(seed)
    assert max(differences.values()) <= 1e-12, differences


def test_bootstrap_is_seeded_and_paired() -> None:
    truths, preds = _scenario(42, n_images=40)
    cache = build_cache(truths, preds, ["a", "b", "c"])
    a = bootstrap({"x": cache, "y": cache}, replicates=60, seed=1, baselines={"y": "x"})
    b = bootstrap({"x": cache, "y": cache}, replicates=60, seed=1, baselines={"y": "x"})
    assert a == b
    assert a["y"]["delta_map50_95"].low == a["y"]["delta_map50_95"].high == 0.0
    assert "delta_map50" not in a["x"]
    point = summarize(cache).map50
    assert 0.0 <= a["x"]["map50"].low <= point <= a["x"]["map50"].high <= 1.0


def test_bootstrap_rejects_mismatched_image_sets() -> None:
    t1, p1 = _scenario(1, n_images=5)
    t2, p2 = _scenario(2, n_images=6)
    with pytest.raises(ValueError, match="same images"):
        bootstrap(
            {"a": build_cache(t1, p1, ["a", "b", "c"]), "b": build_cache(t2, p2, ["a", "b", "c"])}
        )


if __name__ == "__main__":  # the largest difference from pycocotools over N random scenarios
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    worst = {seed: max(_pycocotools_differences(seed).values()) for seed in range(n)}
    seed = max(worst, key=lambda k: worst[k])
    print(
        f"{n} scenarios (seeds 0-{n - 1}): largest |ours - pycocotools| {worst[seed]:.3g}"
        f" (seed {seed}); scenarios with any difference > 1e-12: "
        f"{sum(v > 1e-12 for v in worst.values())}"
    )
