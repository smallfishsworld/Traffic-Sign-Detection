"""COCO-style box AP with per-image match caching, for bootstrap confidence intervals.

The matching and accumulation follow ``pycocotools.cocoeval.COCOeval`` exactly (greedy matching
of score-sorted detections, crowd ground truth matched by intersection-over-detection-area and
never counted as a miss, unmatched detections outside an area range ignored, 101-point
interpolated precision, ``maxDets`` per image and class); ``tests/test_eval_metrics.py`` checks
the results against pycocotools to 1e-12.

Because matching is per image, it is computed once. A bootstrap replicate then only resamples
images, concatenates their cached match arrays and recomputes precision/recall, which makes a
1000-replicate percentile interval take seconds.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tsd.boxes import as_boxes, box_area

# Exactly pycocotools' (unrounded) arrays: its 0.70 recall point is 0.7000000000000001, which
# decides on which side of the threshold a recall of exactly 7/10 falls.
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}


@dataclass(frozen=True)
class ImageTruth:
    """Ground truth of one image for the evaluated classes (``class_ids`` are 0..K-1)."""

    boxes: NDArray[np.float64]
    class_ids: NDArray[np.int64]
    crowd: NDArray[np.bool_]


@dataclass(frozen=True)
class ImagePredictions:
    boxes: NDArray[np.float64]
    scores: NDArray[np.float64]
    class_ids: NDArray[np.int64]


@dataclass(frozen=True)
class MatchRecord:
    """Cached matching of one (image, class, area range)."""

    scores: NDArray[np.float64]  # (D,) descending
    matched: NDArray[np.bool_]  # (T, D)
    ignored: NDArray[np.bool_]  # (T, D)
    num_gt: int  # non-ignored ground truth


def crowd_iou(
    dets: NDArray[np.float64], gts: NDArray[np.float64], crowd: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """IoU matrix ``(D, G)``; for crowd ground truth the denominator is the detection's area."""
    d, g = as_boxes(dets), as_boxes(gts)
    lt = np.maximum(d[:, None, :2], g[None, :, :2])
    rb = np.minimum(d[:, None, 2:], g[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_d, area_g = box_area(d)[:, None], box_area(g)[None, :]
    union = np.where(crowd[None, :], area_d, area_d + area_g - inter)
    out: NDArray[np.float64] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return out


def match_image(
    truth_boxes: NDArray[np.float64],
    truth_crowd: NDArray[np.bool_],
    det_boxes: NDArray[np.float64],
    det_scores: NDArray[np.float64],
    area_range: tuple[float, float],
    max_dets: int,
    iou_thresholds: NDArray[np.float64] = IOU_THRESHOLDS,
) -> MatchRecord | None:
    """COCOeval.evaluateImg for one image and class. ``None`` when there is nothing to evaluate."""
    if len(truth_boxes) == 0 and len(det_boxes) == 0:
        return None
    gt_area = box_area(truth_boxes)
    gt_ignore = truth_crowd | (gt_area < area_range[0]) | (gt_area > area_range[1])
    gt_order = np.argsort(gt_ignore, kind="mergesort")
    gt_boxes, gt_ignore, crowd = truth_boxes[gt_order], gt_ignore[gt_order], truth_crowd[gt_order]
    dt_order = np.argsort(-det_scores, kind="mergesort")[:max_dets]
    dt_boxes, dt_scores = det_boxes[dt_order], det_scores[dt_order]
    num_t, num_d, num_g = len(iou_thresholds), len(dt_boxes), len(gt_boxes)
    matched = np.zeros((num_t, num_d), dtype=bool)
    ignored = np.zeros((num_t, num_d), dtype=bool)
    if num_d and num_g:
        ious = crowd_iou(dt_boxes, gt_boxes, crowd)
        for t, threshold in enumerate(iou_thresholds):
            gt_taken = np.zeros(num_g, dtype=bool)
            for d in range(num_d):
                best_iou, best = min(threshold, 1 - 1e-10), -1
                for g in range(num_g):
                    if gt_taken[g] and not crowd[g]:
                        continue
                    if best > -1 and not gt_ignore[best] and gt_ignore[g]:
                        break
                    if ious[d, g] < best_iou:
                        continue
                    best_iou, best = ious[d, g], g
                if best == -1:
                    continue
                matched[t, d] = True
                ignored[t, d] = gt_ignore[best]
                gt_taken[best] = True
    dt_area = box_area(dt_boxes)
    outside = (dt_area < area_range[0]) | (dt_area > area_range[1])
    ignored |= ~matched & outside[None, :]
    return MatchRecord(dt_scores, matched, ignored, int((~gt_ignore).sum()))


def precision_curve(
    records: Sequence[MatchRecord], recall_points: NDArray[np.float64] = RECALL_POINTS
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """COCOeval.accumulate for one class/area: interpolated precision ``(T, R)`` and the final
    recall ``(T,)``. ``None`` when there is no non-ignored ground truth."""
    num_gt = sum(r.num_gt for r in records)
    if num_gt == 0:
        return None
    scores = np.concatenate([r.scores for r in records])
    order = np.argsort(-scores, kind="mergesort")
    matched = np.concatenate([r.matched for r in records], axis=1)[:, order]
    ignored = np.concatenate([r.ignored for r in records], axis=1)[:, order]
    tp = np.cumsum(matched & ~ignored, axis=1, dtype=np.float64)
    fp = np.cumsum(~matched & ~ignored, axis=1, dtype=np.float64)
    num_t = matched.shape[0]
    precision = np.zeros((num_t, len(recall_points)))
    recall = np.zeros(num_t)
    if tp.shape[1] == 0:
        return precision, recall
    rc = tp / num_gt
    pr = tp / (tp + fp + np.spacing(1))
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]  # monotone envelope
    recall = rc[:, -1]
    for t in range(num_t):
        idx = np.searchsorted(rc[t], recall_points, side="left")
        valid = idx < rc.shape[1]
        precision[t, valid] = pr[t, idx[valid]]
    return precision, recall


@dataclass
class MatchCache:
    """Per-image match records for every (class, area range)."""

    num_images: int
    class_names: list[str]
    records: dict[tuple[int, str], list[MatchRecord | None]] = field(default_factory=dict)

    def curve(
        self,
        class_id: int,
        area: str,
        image_indices: Sequence[int] | NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        recs = self.records[(class_id, area)]
        idx = range(self.num_images) if image_indices is None else image_indices
        return precision_curve([r for r in (recs[i] for i in idx) if r is not None])


def build_cache(
    truths: Sequence[ImageTruth],
    predictions: Sequence[ImagePredictions],
    class_names: Sequence[str],
    areas: Sequence[str] = tuple(AREA_RANGES),
    max_dets: int = 100,
) -> MatchCache:
    if len(truths) != len(predictions):
        raise ValueError("one prediction entry per ground-truth image is required")
    cache = MatchCache(len(truths), list(class_names))
    for k in range(len(class_names)):
        for area in areas:
            cache.records[(k, area)] = [
                match_image(
                    t.boxes[t.class_ids == k],
                    t.crowd[t.class_ids == k],
                    p.boxes[p.class_ids == k],
                    p.scores[p.class_ids == k],
                    AREA_RANGES[area],
                    max_dets,
                )
                for t, p in zip(truths, predictions, strict=True)
            ]
    return cache


@dataclass(frozen=True)
class Summary:
    map50_95: float
    map50: float
    map75: float
    ar100: float
    per_class: dict[str, dict[str, float]]


def summarize(
    cache: MatchCache,
    area: str = "all",
    image_indices: Sequence[int] | NDArray[np.int64] | None = None,
) -> Summary:
    """mAP over classes with ground truth (COCO convention: classes without any are skipped)."""
    per_class: dict[str, dict[str, float]] = {}
    for k, name in enumerate(cache.class_names):
        curve = cache.curve(k, area, image_indices)
        if curve is None:
            continue
        precision, recall = curve
        per_class[name] = {
            "ap50_95": float(precision.mean()),
            "ap50": float(precision[0].mean()),
            "ap75": float(precision[5].mean()),
            "ar100": float(recall.mean()),
        }
    if not per_class:
        nan = float("nan")
        return Summary(nan, nan, nan, nan, {})

    def mean_of(key: str) -> float:
        return float(np.mean([v[key] for v in per_class.values()]))

    return Summary(
        mean_of("ap50_95"), mean_of("ap50"), mean_of("ap75"), mean_of("ar100"), per_class
    )


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def bootstrap(
    caches: Mapping[str, MatchCache],
    replicates: int = 1000,
    seed: int = 0,
    area: str = "all",
    baselines: Mapping[str, str] | None = None,
    confidence: float = 0.95,
) -> dict[str, dict[str, Interval]]:
    """Percentile intervals for mAP50 and mAP50-95 of each cache.

    Every replicate draws one resample of the images (with replacement) and scores *all* caches
    on it, so the ``delta_*`` intervals of a cache against its entry in ``baselines`` are paired
    (the image-to-image variation shared by both runs cancels out).
    """
    names = list(caches)
    sizes = {c.num_images for c in caches.values()}
    if len(sizes) != 1:
        raise ValueError("all caches must cover the same images")
    n = sizes.pop()
    pairs = {k: v for k, v in (baselines or {}).items() if k != v}
    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in names}
    for _ in range(replicates):
        idx = rng.integers(0, n, size=n)
        results = {name: summarize(caches[name], area, idx) for name in names}
        for name, s in results.items():
            samples[name]["map50"].append(s.map50)
            samples[name]["map50_95"].append(s.map50_95)
            if name in pairs:
                base = results[pairs[name]]
                samples[name]["delta_map50"].append(s.map50 - base.map50)
                samples[name]["delta_map50_95"].append(s.map50_95 - base.map50_95)
    tail = 100 * (1 - confidence) / 2
    return {
        name: {
            metric: Interval(*(float(v) for v in np.nanpercentile(values, [tail, 100 - tail])))
            for metric, values in metric_samples.items()
        }
        for name, metric_samples in samples.items()
    }
