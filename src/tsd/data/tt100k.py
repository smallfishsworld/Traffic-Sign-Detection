"""TT100K (Tsinghua-Tencent 100K) to YOLO converter.

This is a rewrite of the 2024 course-project conversion script (report appendix A). Differences
from that script, each of which changes the produced dataset:

* Boxes are normalised (``xmin <= xmax``), clipped to the image and dropped when smaller than
  ``min_box_size`` pixels after clipping. Ultralytics rejects a whole image when any label has a
  coordinate outside ``[0, 1]``, so an unclipped box silently removes its image from training.
* Classes can be filtered (explicit list or the common ">= 100 instances" protocol) and the kept
  classes are re-indexed densely; the old script emitted every entry of ``types`` as a class
  id, including the many with few or no instances (the 2024 report, section 4.2.4, counts 201
  classes, 84 of them with fewer than 10 image instances).
* The validation split is carved deterministically out of TT100K ``train`` (hash of the image id)
  and TT100K ``test`` stays a held-out test split. The old script used ``test`` as the validation
  split, so checkpoint selection and the final score used the same images.
* The annotation JSON is streamed with ``ijson``: only the boxes of images in the requested splits
  are materialised (the file also carries polygons/ellipses for ~100k images).
* Image sizes are read from file headers in a process pool (no full decode).

The converter cannot be run on the real dataset in this repository's CI or build container (TT100K
is not downloadable there); it is tested on a synthetic dataset in the same format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import ijson
import numpy as np
from PIL import Image

from tsd.boxes import as_boxes, clip_boxes, normalize_corners
from tsd.data.yolo import write_data_yaml, write_labels

log = logging.getLogger(__name__)

SOURCE_SPLITS = ("train", "test")
LinkMode = Literal["symlink", "hardlink", "copy", "none"]
CountSplit = Literal["train", "train+test"]
ANNOTATION_CANDIDATES = ("annotations_all.json", "annotations.json")


@dataclass(frozen=True)
class SignBox:
    category: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    source_split: str
    rel_path: str
    objects: tuple[SignBox, ...]


@dataclass(frozen=True)
class ConvertConfig:
    root: Path
    out: Path
    annotations: Path | None = None
    classes: tuple[str, ...] | None = None
    min_instances: int = 0
    count_split: CountSplit = "train+test"
    val_fraction: float = 0.1
    split_salt: str = "tsd-tt100k-v1"
    min_box_size: float = 2.0
    link_mode: LinkMode = "symlink"
    workers: int = 2
    keep_empty: bool = True
    overwrite: bool = False  # replace an existing dataset in ``out`` (else refuse)

    def annotation_path(self) -> Path:
        if self.annotations is not None:
            return self.annotations
        for name in ANNOTATION_CANDIDATES:
            if (self.root / name).exists():
                return self.root / name
        raise FileNotFoundError(f"no {' or '.join(ANNOTATION_CANDIDATES)} under {self.root}")


@dataclass
class ConversionReport:
    classes: list[str]
    images: dict[str, int] = field(default_factory=dict)
    boxes: dict[str, int] = field(default_factory=dict)
    instances: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_boxes: dict[str, int] = field(default_factory=dict)
    skipped_images: dict[str, int] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


def read_split_ids(root: Path, split: str) -> list[str]:
    """Image ids listed in ``<root>/<split>/ids.txt`` (order preserved, duplicates removed)."""
    path = root / split / "ids.txt"
    ids = [line.strip() for line in path.read_text().splitlines()]
    return list(dict.fromkeys(i for i in ids if i))


def read_types(annotation_path: Path) -> list[str]:
    """The ``types`` list of the annotation JSON, read without parsing the ``imgs`` object."""
    types: list[str] = []
    with annotation_path.open("rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "types.item" and event == "string":
                types.append(value)
            elif prefix == "types" and event == "end_array":
                break
    return types


def iter_image_records(annotation_path: Path, wanted: Mapping[str, str]) -> Iterator[ImageRecord]:
    """Stream ``imgs`` and yield compact records for the ids in ``wanted`` (id -> source split)."""
    with annotation_path.open("rb") as f:
        for image_id, entry in ijson.kvitems(f, "imgs", use_float=True):
            split = wanted.get(str(image_id))
            if split is None:
                continue
            objects = tuple(
                SignBox(
                    category=str(obj["category"]),
                    xmin=float(obj["bbox"]["xmin"]),
                    ymin=float(obj["bbox"]["ymin"]),
                    xmax=float(obj["bbox"]["xmax"]),
                    ymax=float(obj["bbox"]["ymax"]),
                )
                for obj in entry.get("objects", [])
            )
            rel_path = str(entry.get("path") or f"{split}/{image_id}.jpg")
            yield ImageRecord(str(image_id), split, rel_path, objects)


def assign_split(image_id: str, source_split: str, val_fraction: float, salt: str) -> str:
    """Deterministic split: TT100K ``test`` -> ``test``; ``train`` -> ``val`` for a stable
    ``val_fraction`` of ids (by hash, so the assignment does not depend on file order or on
    which other images exist) and ``train`` otherwise."""
    if source_split == "test":
        return "test"
    digest = hashlib.sha256(f"{salt}:{image_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if u < val_fraction else "train"


def select_classes(
    types: Sequence[str],
    counts: Mapping[str, int],
    explicit: Sequence[str] | None,
    min_instances: int,
) -> list[str]:
    """Kept classes in ``types`` order: an explicit list, or classes with >= ``min_instances``."""
    if explicit is not None:
        unknown = sorted(set(explicit) - set(types))
        if unknown:
            raise ValueError(f"unknown TT100K classes: {unknown}")
        keep = set(explicit)
        return [t for t in types if t in keep]
    return [t for t in types if counts.get(t, 0) >= max(min_instances, 1)]


def read_image_size(path: Path) -> tuple[int, int] | None:
    """``(width, height)`` from the file header, or ``None`` if the file is missing/unreadable."""
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return None


def read_image_sizes(paths: Sequence[Path], workers: int) -> list[tuple[int, int] | None]:
    """Header-only size reads, fanned out over ``workers`` processes (1 = in-process)."""
    if workers <= 1:
        return [read_image_size(p) for p in paths]
    chunksize = max(1, min(64, len(paths) // (4 * workers)))
    # "spawn" rather than fork: forking a process that already runs library threads (OpenCV,
    # BLAS) can deadlock the child.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(read_image_size, paths, chunksize=chunksize))


def boxes_for_image(
    record: ImageRecord,
    width: int,
    height: int,
    class_index: Mapping[str, int],
    min_box_size: float,
    dropped: Counter[str],
) -> tuple[list[int], np.ndarray]:
    """Class ids and clipped ``xyxy`` boxes of one image; counts dropped boxes by reason."""
    cls_ids: list[int] = []
    raw: list[tuple[float, float, float, float]] = []
    for obj in record.objects:
        if obj.category not in class_index:
            dropped["filtered_class"] += 1
            continue
        cls_ids.append(class_index[obj.category])
        raw.append((obj.xmin, obj.ymin, obj.xmax, obj.ymax))
    boxes = clip_boxes(normalize_corners(as_boxes(raw)), width, height)
    if len(boxes) == 0:
        return [], boxes
    wh = boxes[:, 2:] - boxes[:, :2]
    ok = (wh >= min_box_size).all(axis=1)
    dropped["too_small_after_clip"] += int((~ok).sum())
    return [int(c) for c in np.asarray(cls_ids, dtype=np.int64)[ok]], boxes[ok]


def place_image(src: Path, dst: Path, mode: LinkMode) -> None:
    if mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


OUTPUTS = ("images", "labels", "data.yaml", "conversion_report.json")


def prepare_output(out: Path, overwrite: bool) -> None:
    """Refuse to write into a directory that already holds a converted dataset, unless
    ``overwrite``: then its images/ and labels/ (with any Ultralytics ``*.cache``) are removed
    first, so no file of an earlier run (another split or class list) survives."""
    existing = [name for name in OUTPUTS if (out / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{out} already holds a dataset ({', '.join(existing)}); pass overwrite=True"
            " (--overwrite) to replace it"
        )
    for name in ("images", "labels"):
        if (out / name).is_dir():
            shutil.rmtree(out / name)
    for name in ("data.yaml", "conversion_report.json"):
        (out / name).unlink(missing_ok=True)


def convert(config: ConvertConfig) -> ConversionReport:
    """Convert a TT100K directory into a YOLO dataset under ``config.out`` (which must not hold
    an earlier conversion unless ``config.overwrite``)."""
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1): Ultralytics needs a val split")
    annotation_path = config.annotation_path()
    prepare_output(config.out, config.overwrite)
    wanted: dict[str, str] = {}
    for split in SOURCE_SPLITS:
        for image_id in read_split_ids(config.root, split):
            wanted.setdefault(image_id, split)

    types = read_types(annotation_path)
    records = list(iter_image_records(annotation_path, wanted))
    skipped: Counter[str] = Counter()
    skipped["listed_but_not_annotated"] = len(wanted) - len(records)

    count_splits = {"train"} if config.count_split == "train" else {"train", "test"}
    counts = Counter(
        obj.category for r in records if r.source_split in count_splits for obj in r.objects
    )
    classes = select_classes(types, counts, config.classes, config.min_instances)
    class_index = {name: i for i, name in enumerate(classes)}

    sizes = read_image_sizes([config.root / r.rel_path for r in records], config.workers)
    dropped: Counter[str] = Counter()
    report = ConversionReport(classes=classes, config=asdict(config))
    per_split_images: Counter[str] = Counter()
    per_split_boxes: Counter[str] = Counter()
    instances: dict[str, Counter[str]] = {}

    for record, size in zip(records, sizes, strict=True):
        if size is None:
            skipped["missing_or_unreadable_image"] += 1
            continue
        width, height = size
        cls_ids, boxes = boxes_for_image(
            record, width, height, class_index, config.min_box_size, dropped
        )
        if not cls_ids and not config.keep_empty:
            skipped["no_kept_boxes"] += 1
            continue
        split = assign_split(
            record.image_id, record.source_split, config.val_fraction, config.split_salt
        )
        src = config.root / record.rel_path
        stem = Path(record.rel_path).stem
        write_labels(config.out / "labels" / split / f"{stem}.txt", cls_ids, boxes, width, height)
        place_image(src, config.out / "images" / split / src.name, config.link_mode)
        per_split_images[split] += 1
        per_split_boxes[split] += len(cls_ids)
        instances.setdefault(split, Counter()).update(classes[c] for c in cls_ids)

    if not per_split_images["val"]:
        raise ValueError(
            f"no image fell into the val split (val_fraction={config.val_fraction},"
            f" {per_split_images['train']} train images); Ultralytics needs a val split: convert"
            " more images or use a larger val_fraction or another split_salt"
        )
    splits = {s: f"images/{s}" for s in ("train", "val", "test") if per_split_images[s]}
    write_data_yaml(config.out / "data.yaml", classes, splits)
    report.images = dict(sorted(per_split_images.items()))
    report.boxes = dict(sorted(per_split_boxes.items()))
    report.instances = {s: dict(sorted(c.items())) for s, c in sorted(instances.items())}
    report.dropped_boxes = dict(sorted(dropped.items()))
    report.skipped_images = {k: v for k, v in sorted(skipped.items()) if v}
    (config.out / "conversion_report.json").write_text(report.to_json())
    log.info("converted %s images into %s", sum(per_split_images.values()), config.out)
    return report
