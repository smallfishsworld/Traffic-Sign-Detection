"""The Open Images evaluation protocol on top of COCO-style metrics.

What is evaluated (also in ``docs/EVALUATION.md``):

* **Images**: every image of the evaluation set built by ``tsd openimages`` (Open Images V7
  validation + test images with at least one Traffic sign, Stop sign or Traffic light box).
* **Classes**: the data-set classes the model also has, matched by case-insensitive name. For a
  COCO-pretrained model that is Stop sign (COCO ``stop sign``, id 11) and Traffic light
  (``traffic light``, id 9); Traffic sign has no COCO class and is excluded.
* **Unannotated classes**: Open Images is not exhaustively labelled. A class is *evaluated* in an
  image only if it was verified present or absent by a human (image-level labels, expanded over
  the class hierarchy) or has boxes there. Detections of a class that is not evaluated in an
  image are dropped before matching: they are neither true nor false positives.
* **Group-of boxes** (one box around several objects) become COCO "crowd" regions: detections
  inside them are ignored and the region is never counted as a miss.
* **Class hierarchy**: a box also counts for every ancestor class (a Stop-sign box is a
  Traffic-sign box too), the Open Images challenge's box expansion. It only matters for a model
  that has a parent class; COCO models have no Traffic sign class.
* Depicted, occluded and truncated objects are ordinary ground truth, as in Open Images.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tsd.eval.metrics import ImagePredictions, ImageTruth
from tsd.infer.types import Detections


@dataclass(frozen=True)
class EvalImage:
    image_id: str
    file: str  # relative to the data-set root
    width: int
    height: int
    boxes: NDArray[np.float64]  # (N, 4) xyxy, original pixels
    labels: NDArray[np.int64]  # data-set class indices
    group_of: NDArray[np.bool_]
    verified_positive: frozenset[int]
    verified_negative: frozenset[int]

    def is_evaluated(self, class_id: int) -> bool:
        return (
            class_id in self.verified_positive
            or class_id in self.verified_negative
            or bool((self.labels == class_id).any())
        )


def ancestors(name: str, parents: dict[str, list[str]]) -> list[str]:
    """Every ancestor of ``name`` in ``parents`` (child -> direct parents), nearest first."""
    out: list[str] = []
    todo = list(parents.get(name, []))
    while todo:
        parent = todo.pop(0)
        if parent not in out:
            out.append(parent)
            todo.extend(parents.get(parent, []))
    return out


def expand_boxes(
    boxes: NDArray[np.float64],
    labels: NDArray[np.int64],
    group_of: NDArray[np.bool_],
    classes: Sequence[str],
    parents: dict[str, list[str]],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_]]:
    """Add a copy of every box for each ancestor class (Open Images box expansion)."""
    index = {name: i for i, name in enumerate(classes)}
    extra = [
        (i, index[a])
        for i, c in enumerate(labels)
        for a in ancestors(classes[int(c)], parents)
        if a in index
    ]
    if not extra:
        return boxes, labels, group_of
    rows = np.array([i for i, _ in extra])
    return (
        np.concatenate([boxes, boxes[rows]]),
        np.concatenate([labels, np.array([a for _, a in extra], dtype=np.int64)]),
        np.concatenate([group_of, group_of[rows]]),
    )


@dataclass(frozen=True)
class EvalSet:
    root: Path
    classes: list[str]
    images: list[EvalImage]
    class_parents: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path, annotations: Path | None = None) -> EvalSet:
        doc = json.loads((annotations or root / "annotations.json").read_text())
        classes = list(doc["classes"])
        parents = {k: list(v) for k, v in doc.get("info", {}).get("class_parents", {}).items()}
        images = []
        for r in doc["images"]:
            boxes, labels, group_of = expand_boxes(
                np.asarray(r["boxes"], dtype=np.float64).reshape(-1, 4),
                np.asarray(r["labels"], dtype=np.int64),
                np.asarray(r["group_of"], dtype=bool),
                classes,
                parents,
            )
            images.append(
                EvalImage(
                    image_id=r["image_id"],
                    file=r["file"],
                    width=int(r["width"]),
                    height=int(r["height"]),
                    boxes=boxes,
                    labels=labels,
                    group_of=group_of,
                    verified_positive=frozenset(r["verified_positive"]),
                    verified_negative=frozenset(r["verified_negative"]),
                )
            )
        return cls(root, classes, images, parents)

    def paths(self) -> list[Path]:
        return [self.root / img.file for img in self.images]


@dataclass(frozen=True)
class ClassMap:
    """Evaluated classes: index ``k`` is data-set class ``dataset_ids[k]`` and model class
    ``model_ids[k]``."""

    names: list[str]
    dataset_ids: list[int]
    model_ids: list[int]
    excluded: list[str]  # data-set classes the model does not have

    def to_eval_ids(self, model_class_ids: NDArray[np.int64]) -> NDArray[np.int64]:
        """Model class ids → evaluated index ``k``; -1 for classes that are not evaluated."""
        lut = np.full(max([*self.model_ids, int(model_class_ids.max(initial=0))]) + 1, -1)
        lut[self.model_ids] = np.arange(len(self.model_ids))
        return lut[model_class_ids].astype(np.int64)


def map_classes(dataset_classes: Sequence[str], model_names: Sequence[str]) -> ClassMap:
    by_name = {name.casefold(): i for i, name in enumerate(model_names)}
    names, dataset_ids, model_ids, excluded = [], [], [], []
    for i, name in enumerate(dataset_classes):
        j = by_name.get(name.casefold())
        if j is None:
            excluded.append(name)
            continue
        names.append(name)
        dataset_ids.append(i)
        model_ids.append(j)
    if not names:
        raise ValueError(f"the model has none of the data-set classes {list(dataset_classes)}")
    return ClassMap(names, dataset_ids, model_ids, excluded)


def ground_truth(evalset: EvalSet, cmap: ClassMap) -> list[ImageTruth]:
    lut = {d: k for k, d in enumerate(cmap.dataset_ids)}
    out = []
    for img in evalset.images:
        keep = np.isin(img.labels, cmap.dataset_ids)
        out.append(
            ImageTruth(
                boxes=img.boxes[keep],
                class_ids=np.array([lut[int(c)] for c in img.labels[keep]], dtype=np.int64),
                crowd=img.group_of[keep],
            )
        )
    return out


@dataclass(frozen=True)
class FilterStats:
    """Detections dropped because their class was not verified in the image, per class."""

    dropped_detections: dict[str, int]
    unevaluated_images: dict[str, int]


def predictions(
    evalset: EvalSet,
    cmap: ClassMap,
    detections: Sequence[Detections],
    drop_unverified: bool = True,
) -> tuple[list[ImagePredictions], FilterStats]:
    """Model detections → evaluated classes, minus detections of unverified classes (unless
    ``drop_unverified`` is false: the strict variant where every detection counts)."""
    if len(detections) != len(evalset.images):
        raise ValueError("one Detections per evaluation image is required")
    dropped = dict.fromkeys(cmap.names, 0)
    unevaluated = dict.fromkeys(cmap.names, 0)
    out = []
    for img, det in zip(evalset.images, detections, strict=True):
        ids = cmap.to_eval_ids(det.class_ids.astype(np.int64))
        keep = ids >= 0
        for k, name in enumerate(cmap.names):
            if drop_unverified and not img.is_evaluated(cmap.dataset_ids[k]):
                unevaluated[name] += 1
                mask = ids == k
                dropped[name] += int(mask.sum())
                keep &= ~mask
        out.append(
            ImagePredictions(
                boxes=det.boxes[keep].astype(np.float64),
                scores=det.scores[keep].astype(np.float64),
                class_ids=ids[keep],
            )
        )
    return out, FilterStats(dropped, unevaluated)
