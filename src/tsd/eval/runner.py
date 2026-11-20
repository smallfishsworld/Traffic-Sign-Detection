"""Run models over the evaluation set, save their detections, and score saved detections.

Scores are always computed from the saved detection files, so a reported number can be
recomputed from the file without running the model again (``tsd eval --reuse``).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from tsd.eval.metrics import (
    AREA_RANGES,
    IOU_THRESHOLDS,
    RECALL_POINTS,
    ImagePredictions,
    ImageTruth,
    MatchCache,
    bootstrap,
    build_cache,
    summarize,
)
from tsd.eval.protocol import ClassMap, EvalSet, ground_truth, map_classes, predictions
from tsd.infer import EVAL_NMS, Detections, Detector, NMSConfig, load_detector
from tsd.io import Image, load_image
from tsd.net import file_digest

PredictFn = Callable[[Image], Detections]


def model_files(model: Path) -> list[Path]:
    """The files that make up a model (an OpenVINO IR is an .xml plus a .bin)."""
    return [model, model.with_suffix(".bin")] if model.suffix == ".xml" else [model]


@dataclass
class RunDetections:
    """One model's detections on every evaluation image (evaluated classes only)."""

    label: str
    model: str
    backend: str
    source: str
    model_names: list[str]
    nms: dict[str, Any]
    files_sha256: dict[str, str]
    image_ids: list[str]
    detections: list[Detections]
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        doc = {k: v for k, v in asdict(self).items() if k != "detections"}
        doc["detections"] = [
            {
                "boxes": [[round(float(v), 3) for v in b] for b in d.boxes],
                "scores": [round(float(s), 6) for s in d.scores],
                "class_ids": [int(c) for c in d.class_ids],
            }
            for d in self.detections
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(doc, separators=(",", ":")).encode()
        if path.suffix == ".gz":  # mtime=0: the same detections give byte-identical files
            data = gzip.compress(data, mtime=0)
        path.write_bytes(data)

    @classmethod
    def load(cls, path: Path) -> RunDetections:
        data = path.read_bytes()
        doc = json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)
        doc["detections"] = [
            Detections(
                np.asarray(d["boxes"], np.float32).reshape(-1, 4),
                np.asarray(d["scores"], np.float32),
                np.asarray(d["class_ids"], np.int64),
            )
            for d in doc["detections"]
        ]
        return cls(**doc)


def _as_json(value: Any) -> Any:
    """``value`` as it reads back from a JSON file (tuples become lists)."""
    return json.loads(json.dumps(value))


def can_reuse(
    saved: RunDetections,
    files_sha256: Mapping[str, str],
    tiling: Mapping[str, Any] | None,
    nms: Mapping[str, Any],
    image_ids: Sequence[str],
) -> bool:
    """Whether ``saved`` detections are what running the model now would give: same model files,
    tiling and NMS settings (compared as JSON, the form they were saved in) and images. The
    detection file's own SHA-256, which a report records, is for audit (it identifies the exact
    file that was scored) and is not checked here."""
    return (
        saved.files_sha256 == dict(files_sha256)
        and _as_json(saved.extra.get("tiling")) == _as_json(tiling)
        and _as_json(saved.nms) == _as_json(nms)
        and list(saved.image_ids) == list(image_ids)
    )


def config_key(nms: NMSConfig, tiling: Mapping[str, Any] | None, precision_hint: str) -> str:
    """Short hash of everything besides the model files that changes the detections (the
    NMS settings, the tiling configuration and the OpenVINO precision hint): part of the
    detection file's name, so runs with different settings never share a file."""
    doc = {"nms": asdict(nms), "tiling": dict(tiling) if tiling else None, "hint": precision_hint}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:10]


def eval_nms(cmap: ClassMap) -> NMSConfig:
    """Ultralytics ``val`` settings, restricted to the evaluated classes (so detections of other
    classes cannot use up the per-image detection cap)."""
    return replace(EVAL_NMS, classes=tuple(cmap.model_ids))


def detector_fn(detector: Detector, nms: NMSConfig) -> PredictFn:
    def predict(image: Image) -> Detections:
        return detector.predict([image], nms=nms)[0]

    return predict


def run_model(
    model: Path,
    evalset: EvalSet,
    label: str,
    threads: int | None = None,
    openvino_precision: str = "f32",
    wrap: Callable[[Detector, NMSConfig], PredictFn] = detector_fn,
    extra: Mapping[str, Any] | None = None,
) -> RunDetections:
    """Predict every evaluation image with ``model``; ``wrap`` builds the per-image predict
    function (plain letterboxed inference by default, tiled inference for ``tsd.tiling``)."""
    detector = load_detector(
        model,
        threads=threads,
        openvino_precision="bf16" if openvino_precision == "bf16" else "f32",
    )
    cmap = map_classes(evalset.classes, detector.names)
    nms = eval_nms(cmap)
    predict = wrap(detector, nms)
    t0 = time.perf_counter()
    dets = [predict(load_image(p)) for p in evalset.paths()]
    seconds = time.perf_counter() - t0
    source = model.name
    if model.suffix != ".pt":
        from tsd.infer import ModelMeta

        source = ModelMeta.load(model).source or model.name
    return RunDetections(
        label=label,
        model=str(model),
        backend=detector.name,
        source=source,
        model_names=detector.names,
        nms=asdict(nms),
        files_sha256={p.name: file_digest(p, "sha256") for p in model_files(model)},
        image_ids=[img.image_id for img in evalset.images],
        detections=dets,
        extra={
            **(extra or {}),
            "threads": threads,
            "runtime_settings": detector.runtime_settings(),
            "wall_seconds_not_a_benchmark": round(seconds, 1),
        },
    )


def pycocotools_stats(
    truths: Sequence[ImageTruth], preds: Sequence[ImagePredictions], class_names: Sequence[str]
) -> dict[str, float]:
    """The same evaluation through ``pycocotools.COCOeval`` (cross-check of ``tsd.eval.metrics``).
    Values pycocotools reports as -1 (no ground truth in that size range) become NaN."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    def xywh(b: Any) -> list[float]:
        x1, y1, x2, y2 = (float(v) for v in b)
        return [x1, y1, x2 - x1, y2 - y1]

    annotations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for i, (t, p) in enumerate(zip(truths, preds, strict=True)):
        for box, cls, crowd in zip(t.boxes, t.class_ids, t.crowd, strict=True):
            bb = xywh(box)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": i + 1,
                    "category_id": int(cls) + 1,
                    "bbox": bb,
                    "area": bb[2] * bb[3],
                    "iscrowd": int(crowd),
                }
            )
        for box, score, cls in zip(p.boxes, p.scores, p.class_ids, strict=True):
            results.append(
                {
                    "image_id": i + 1,
                    "category_id": int(cls) + 1,
                    "bbox": xywh(box),
                    "score": float(score),
                }
            )
    keys = ["map50_95", "map50", "map75", "map_small", "map_medium", "map_large"]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {
            "images": [{"id": i + 1} for i in range(len(truths))],
            "annotations": annotations,
            "categories": [{"id": k + 1, "name": n} for k, n in enumerate(class_names)],
        }
        gt.createIndex()
        if not results:  # loadRes refuses an empty list; no detections = AP 0 where GT exists
            sizes = {"small": 0, "medium": 0, "large": 0}
            for a in annotations:
                if not a["iscrowd"]:
                    for size, (lo, hi) in AREA_RANGES.items():
                        if size in sizes and lo <= a["area"] <= hi:
                            sizes[size] += 1
            empty = dict.fromkeys(["map50_95", "map50", "map75", "ar100"], 0.0)
            for size, n in sizes.items():
                empty[f"map_{size}"] = 0.0 if n else float("nan")
            return {k: empty[k] for k in [*keys, "ar100"]}
        evaluator = COCOeval(gt, gt.loadRes(results), "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = [float(v) if v > -1 else float("nan") for v in evaluator.stats]
    return {**dict(zip(keys, stats[:6], strict=True)), "ar100": stats[8]}


def cross_check(mine: Mapping[str, float], ref: Mapping[str, float]) -> dict[str, Any]:
    """Compare our metrics with pycocotools' key by key. A value that is NaN on one side only is
    a disagreement (``max_abs_diff`` becomes infinite); otherwise the largest difference over the
    keys that are finite on both sides (None if there are none)."""
    nan_mismatch = sorted(
        k for k, v in ref.items() if math.isnan(v) != math.isnan(float(mine.get(k, math.nan)))
    )
    diffs = [
        abs(float(mine[k]) - v)
        for k, v in ref.items()
        if k in mine and not math.isnan(v) and not math.isnan(float(mine[k]))
    ]
    worst: float | None = max(diffs) if diffs else None
    if nan_mismatch:
        worst = math.inf
    return {
        "stats": {k: _round(v, 6) for k, v in ref.items()},
        "compared_keys": len(diffs),
        "nan_mismatch": nan_mismatch,
        "max_abs_diff": worst,
    }


@dataclass
class ScoredRun:
    run: RunDetections
    cmap: ClassMap
    cache: MatchCache
    truths: list[ImageTruth]
    preds: list[ImagePredictions]
    filter_stats: dict[str, Any]


def score(evalset: EvalSet, run: RunDetections, drop_unverified: bool = True) -> ScoredRun:
    if run.image_ids != [img.image_id for img in evalset.images]:
        raise ValueError(f"{run.label}: detections are for a different image list")
    cmap = map_classes(evalset.classes, run.model_names)
    truths = ground_truth(evalset, cmap)
    preds, stats = predictions(evalset, cmap, run.detections, drop_unverified)
    cache = build_cache(truths, preds, cmap.names)
    return ScoredRun(run, cmap, cache, truths, preds, asdict(stats))


def _round(x: float, digits: int = 6) -> float | None:
    """Stored precision of every metric. Six decimals, so that the three-decimal tables (and
    anyone re-rounding the JSON) are not double-rounded: with four, a stored 0.7115 shown as
    0.712 could stand for 0.711498."""
    return None if np.isnan(x) else round(x, digits)


def report(
    evalset: EvalSet,
    runs: Sequence[RunDetections],
    replicates: int = 1000,
    seed: int = 0,
    baseline_backend: str = "pytorch-fp32",
    run_cross_check: bool = True,
    drop_unverified: bool = True,
    detection_files: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Point estimates, 95% bootstrap intervals and paired deltas against the run with
    ``baseline_backend`` that was exported from the same weights. ``detection_files`` (label ->
    path and sha256) records which committed detection file each run was scored from."""
    scored = {r.label: score(evalset, r, drop_unverified) for r in runs}
    # Untiled runs are compared with ``baseline_backend``; tiled runs with the untiled run of the
    # same backend (the effect of tiling alone). Always the same weights.
    baselines = {}
    for label, s in scored.items():
        tiled = s.run.extra.get("tiling") is not None
        wanted = s.run.backend if tiled else baseline_backend
        for other, o in scored.items():
            untiled = o.run.extra.get("tiling") is None
            if untiled and o.run.source == s.run.source and o.run.backend == wanted:
                baselines[label] = other
    caches = {k: v.cache for k, v in scored.items()}
    intervals = bootstrap(caches, replicates, seed, baselines=baselines)
    size_intervals = (
        {
            area: bootstrap(caches, replicates, seed, area=area, baselines=baselines)
            for area in ("small", "medium", "large")
        }
        if any(s.run.extra.get("tiling") for s in scored.values())
        else {}
    )
    # Every summary once per run and area; reused for by_size, the deltas and the cross-check.
    summaries = {
        label: {area: summarize(s.cache, area) for area in AREA_RANGES}
        for label, s in scored.items()
    }
    out: dict[str, Any] = {}
    for label, s in scored.items():
        overall = summaries[label]["all"]
        by_size = {}
        for area in AREA_RANGES:
            if area == "all":
                continue
            sub = summaries[label][area]
            gt_counts = {
                name: sum(r.num_gt for r in s.cache.records[(k, area)] if r is not None)
                for k, name in enumerate(s.cmap.names)
            }
            by_size[area] = {
                "map50_95": _round(sub.map50_95),
                "map50": _round(sub.map50),
                "ar100": _round(sub.ar100),
                "gt_boxes": gt_counts,
                "per_class": {
                    n: {k: _round(v) for k, v in m.items()} for n, m in sub.per_class.items()
                },
            }
        ci = intervals[label]
        entry: dict[str, Any] = {
            "model": s.run.model,
            "backend": s.run.backend,
            "source": s.run.source,
            "files_sha256": s.run.files_sha256,
            "classes": s.cmap.names,
            "excluded_classes": s.cmap.excluded,
            "detections_scored": int(sum(len(p.scores) for p in s.preds)),
            "unverified_filter": s.filter_stats,
            "map50": _round(overall.map50),
            "map50_95": _round(overall.map50_95),
            "map75": _round(overall.map75),
            "ar100": _round(overall.ar100),
            "ci95": {m: [_round(ci[m].low), _round(ci[m].high)] for m in ("map50", "map50_95")},
            "per_class": {
                n: {k: _round(v) for k, v in m.items()} for n, m in overall.per_class.items()
            },
            "by_size": by_size,
            "nms": s.run.nms,
            "extra": s.run.extra,
        }
        if detection_files and label in detection_files:
            entry["detections_file"] = detection_files[label]
        if label in baselines and baselines[label] != label:
            base = summaries[baselines[label]]["all"]
            entry["baseline"] = baselines[label]
            if size_intervals:
                entry["by_size_delta_vs_baseline"] = {}
                for area, area_ci in size_intervals.items():
                    a = summaries[label][area]
                    b = summaries[baselines[label]][area]
                    entry["by_size_delta_vs_baseline"][area] = {
                        "map50_95": _round(a.map50_95 - b.map50_95),
                        "map50": _round(a.map50 - b.map50),
                        "ar100": _round(a.ar100 - b.ar100),
                        "ci95": {
                            m: [
                                _round(area_ci[label][f"delta_{m}"].low),
                                _round(area_ci[label][f"delta_{m}"].high),
                            ]
                            for m in ("map50", "map50_95")
                        },
                    }
            entry["delta_vs_baseline"] = {
                "map50": _round(overall.map50 - base.map50),
                "map50_95": _round(overall.map50_95 - base.map50_95),
                "ci95": {
                    m: [_round(ci[f"delta_{m}"].low), _round(ci[f"delta_{m}"].high)]
                    for m in ("map50", "map50_95")
                },
            }
        if run_cross_check:
            ref = pycocotools_stats(s.truths, s.preds, s.cmap.names)
            mine = {
                "map50_95": overall.map50_95,
                "map50": overall.map50,
                "map75": overall.map75,
                "ar100": overall.ar100,
                **{f"map_{a}": summaries[label][a].map50_95 for a in ("small", "medium", "large")},
            }
            entry["pycocotools"] = cross_check(mine, ref)
        out[label] = entry
    return {"protocol": protocol_info(replicates, seed, drop_unverified, evalset), "runs": out}


def dataset_note(evalset: EvalSet) -> str:
    """What the evaluated images are, counted from the loaded set (not assumed)."""
    total = len(evalset.images)
    boxless = sum(len(im.boxes) == 0 for im in evalset.images)
    head = f"{total} Open Images V7 validation + test images"
    if not boxless:
        return (
            f"{head}, each with at least one box of a data-set class; verified-negative images"
            " without such boxes are not included"
        )
    return (
        f"{head}: {total - boxless} with at least one box of a data-set class and {boxless}"
        " verified-negative images without such boxes"
    )


def protocol_info(
    replicates: int, seed: int, drop_unverified: bool, evalset: EvalSet | None = None
) -> dict[str, Any]:
    """The evaluation protocol, written from the constants the code uses."""
    step = IOU_THRESHOLDS[1] - IOU_THRESHOLDS[0]
    ious = f"{IOU_THRESHOLDS[0]:.2f}:{step:.2f}:{IOU_THRESHOLDS[-1]:.2f}"
    small, medium = AREA_RANGES["small"][1] ** 0.5, AREA_RANGES["medium"][1] ** 0.5
    nms = EVAL_NMS
    return {
        **({"dataset_note": dataset_note(evalset)} if evalset is not None else {}),
        "matching": f"COCO (pycocotools-equivalent): IoU {ious}, {len(RECALL_POINTS)}-point"
        " interpolated AP, maxDets 100 per image and class, area ranges (pixels of the original"
        f" image, bounds inclusive as in pycocotools): small [0, {small:.0f}^2], medium"
        f" [{small:.0f}^2, {medium:.0f}^2], large [{medium:.0f}^2, inf)",
        "unverified_classes": (
            "detections of a class that is neither verified present nor verified absent in"
            " an image (and has no box there) are dropped"
            if drop_unverified
            else "strict variant: every detection counts (unverified ones as false positives)"
        ),
        "group_of": "Open Images group-of boxes are COCO crowd regions",
        "class_hierarchy": "boxes also count for every evaluated ancestor class (Open Images"
        " box expansion)",
        "nms": f"conf {nms.conf_threshold}, IoU {nms.iou_threshold},"
        f" {'multi-label' if nms.multi_label else 'single-label'}, max {nms.max_det} per image,"
        " evaluated classes only",
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "method": "percentile 95% interval; images resampled with replacement; one"
            " resample per replicate shared by all runs (paired deltas)",
        },
    }


def markdown_table(doc: Mapping[str, Any]) -> str:
    """The headline table of a :func:`report`."""

    def ci(pair: Sequence[float | None]) -> str:
        lo, hi = pair
        return "n/a" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"

    def num(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    lines = [
        "| run | mAP50 | 95% CI | mAP50-95 | 95% CI | Δ mAP50-95 vs baseline [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in doc["runs"].items():
        delta = r.get("delta_vs_baseline")
        delta_text = (
            f"{delta['map50_95']:+.4f} {ci(delta['ci95']['map50_95'])}"
            if delta
            else "(baseline)"
            if any(o.get("baseline") == label for o in doc["runs"].values())
            else "-"
        )
        lines.append(
            f"| {label} | {num(r['map50'])} | {ci(r['ci95']['map50'])} | {num(r['map50_95'])}"
            f" | {ci(r['ci95']['map50_95'])} | {delta_text} |"
        )
    return "\n".join(lines) + "\n"
