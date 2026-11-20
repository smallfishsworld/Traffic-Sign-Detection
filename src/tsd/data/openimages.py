"""Build a YOLO-format evaluation set from Open Images V7 validation + test images.

The set contains every validation/test image with at least one box of the target classes
(default: Traffic sign, Stop sign, Traffic light). Besides the YOLO labels it writes an
``annotations.json`` with what a correct Open Images evaluation needs and YOLO files cannot hold:

* box flags (``IsGroupOf``, ``IsDepiction``, ``IsOccluded``, ``IsTruncated``);
* the human-verified image-level labels, expanded over the class hierarchy the way the Open Images
  challenge does it: a positive label is also positive for its ancestors (a stop sign is a
  traffic sign) and a negative label is also negative for its descendants (no traffic sign means
  no stop sign). Open Images is not exhaustively labelled, so detections of a class that is
  neither verified present nor verified absent in an image must be ignored, not counted as false
  positives (see ``tsd.eval.protocol``).

Images are fetched from the public Open Images S3 bucket with :class:`tsd.net.Downloader`
(one connection and 2 MB/s by default, resumable, MD5-checked against the S3 ETag). The images
are not committed; ``manifest.csv`` (id, split, sha256, size, licence, author, source URLs) is.

Orientation: images are used exactly as stored. None of the selected images has an EXIF
orientation tag, and the two whose ``Rotation`` metadata is 270 are stored sideways on S3 with
boxes that match the stored pixels (checked by drawing the boxes), so rotating them would break
the labels.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from tsd.boxes import as_boxes
from tsd.data.yolo import write_data_yaml, write_labels
from tsd.io import load_image
from tsd.net import Downloader, DownloadItem, DownloadPolicy, file_digest

log = logging.getLogger(__name__)

SPLITS = ("validation", "test")
TARGET_CLASSES = ("Traffic sign", "Stop sign", "Traffic light")
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
_GCS = "https://storage.googleapis.com/openimages"
_EXIF_ORIENTATION = 0x0112


@dataclass(frozen=True)
class MetadataFile:
    url: str
    size: int
    md5: str  # hex


# Pinned sizes and MD5s (from the servers' x-goog-hash headers) of the files the builder reads.
METADATA: dict[str, MetadataFile] = {
    "validation-annotations-bbox.csv": MetadataFile(
        f"{_GCS}/v5/validation-annotations-bbox.csv", 25105048, "c5e8200df129ea6867e913e8b21fcab9"
    ),
    "test-annotations-bbox.csv": MetadataFile(
        f"{_GCS}/v5/test-annotations-bbox.csv", 77484237, "1cc058a7003b4e73d47642276e9b123b"
    ),
    "validation-annotations-human-imagelabels-boxable.csv": MetadataFile(
        f"{_GCS}/v5/validation-annotations-human-imagelabels-boxable.csv",
        10649275,
        "2e129f40a209e65dfe4342ed0e7bd5f2",
    ),
    "test-annotations-human-imagelabels-boxable.csv": MetadataFile(
        f"{_GCS}/v5/test-annotations-human-imagelabels-boxable.csv",
        32055857,
        "207ed0ce2755ad5a689c8eb6eeb039bf",
    ),
    "validation-images-with-rotation.csv": MetadataFile(
        f"{_GCS}/2018_04/validation/validation-images-with-rotation.csv",
        15245485,
        "643a54e43b0bab8acce8817b4a569780",
    ),
    "test-images-with-rotation.csv": MetadataFile(
        f"{_GCS}/2018_04/test/test-images-with-rotation.csv",
        45227339,
        "d832feb775b3cb78077bf8ae350adce5",
    ),
    "oidv7-class-descriptions-boxable.csv": MetadataFile(
        f"{_GCS}/v7/oidv7-class-descriptions-boxable.csv", 12064, "c5e7cb6b85d0539b2105db2d1974681f"
    ),
    "bbox_labels_600_hierarchy.json": MetadataFile(
        f"{_GCS}/2018_04/bbox_labels_600_hierarchy.json", 86291, "3fa04df9ea43b1274aaa9d06e1b38900"
    ),
}


@dataclass(frozen=True)
class BuildConfig:
    out: Path
    metadata_dir: Path
    reuse_dirs: tuple[Path, ...] = ()
    classes: tuple[str, ...] = TARGET_CLASSES
    splits: tuple[str, ...] = SPLITS
    policy: DownloadPolicy = field(default_factory=DownloadPolicy)
    max_images: int | None = None
    manifest_copy_dir: Path | None = None
    # Also keep images with no target box but a human-verified *negative* label for a target
    # class (Open Images challenge style: false positives there count against the model).
    include_verified_negatives: bool = False


@dataclass(frozen=True)
class OIBox:
    label: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    group_of: bool
    depiction: bool
    occluded: bool
    truncated: bool


@dataclass
class OIImage:
    image_id: str
    split: str
    boxes: list[OIBox] = field(default_factory=list)
    verified: dict[str, bool] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassHierarchy:
    """Ancestor/descendant relations restricted to a set of class names."""

    parents: Mapping[str, tuple[str, ...]]

    def ancestors(self, name: str) -> set[str]:
        out: set[str] = set()
        stack = list(self.parents.get(name, ()))
        while stack:
            p = stack.pop()
            if p not in out:
                out.add(p)
                stack.extend(self.parents.get(p, ()))
        return out

    def descendants(self, name: str) -> set[str]:
        return {c for c in self.parents if name in self.ancestors(c)}


def load_hierarchy(
    path: Path, mid_to_name: Mapping[str, str], keep: Iterable[str]
) -> ClassHierarchy:
    """Parse ``bbox_labels_600_hierarchy.json`` into parent links among the ``keep`` classes."""
    keep_set = set(keep)
    parents: dict[str, set[str]] = defaultdict(set)

    def walk(node: Mapping[str, Any], ancestors: list[str]) -> None:
        name = mid_to_name.get(node["LabelName"])
        if name in keep_set:
            nearest = next((a for a in reversed(ancestors) if a in keep_set), None)
            if nearest is not None:
                parents[name].add(nearest)
        chain = [*ancestors, name] if name is not None else ancestors
        for child in node.get("Subcategory", []):
            walk(child, chain)

    walk(json.loads(path.read_text()), [])
    return ClassHierarchy({k: tuple(sorted(v)) for k, v in parents.items()})


def expand_verified_labels(
    verified: Mapping[str, bool], hierarchy: ClassHierarchy
) -> tuple[set[str], set[str]]:
    """Open Images label expansion: positives go up the hierarchy, negatives go down."""
    positive = {n for n, v in verified.items() if v}
    negative = {n for n, v in verified.items() if not v}
    for name in list(positive):
        positive |= hierarchy.ancestors(name)
    for name in list(negative):
        negative |= hierarchy.descendants(name)
    if positive & negative:
        raise ValueError(f"contradictory verified labels after expansion: {positive & negative}")
    return positive, negative


def ensure_metadata(
    names: Sequence[str], metadata_dir: Path, reuse_dirs: Sequence[Path], downloader: Downloader
) -> dict[str, Path]:
    """Locate verified copies of the metadata files, downloading only the missing ones."""
    found: dict[str, Path] = {}
    for name in names:
        spec = METADATA[name]
        for directory in (metadata_dir, *reuse_dirs):
            candidate = directory / name
            if (
                candidate.exists()
                and candidate.stat().st_size == spec.size
                and file_digest(candidate, "md5") == spec.md5
            ):
                found[name] = candidate
                break
        else:
            item = DownloadItem(spec.url, metadata_dir / name, spec.size, spec.md5)
            found[name] = downloader.fetch(item).dest
    return found


def read_class_mids(path: Path, names: Iterable[str]) -> dict[str, str]:
    """Map MID -> display name for ``names`` (raises if a name is unknown)."""
    wanted = set(names)
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        for mid, display in csv.reader(f):
            if display in wanted:
                out[mid] = display
    missing = wanted - set(out.values())
    if missing:
        raise ValueError(f"classes not in Open Images: {sorted(missing)}")
    return out


def collect_images(
    split: str,
    bbox_csv: Path,
    labels_csv: Path,
    images_csv: Path,
    mid_to_name: Mapping[str, str],
    include_verified_negatives: bool = False,
) -> dict[str, OIImage]:
    """Images of ``split`` with at least one box of the target classes (plus, optionally, those
    with a human-verified negative label for one), with labels and meta."""
    images: dict[str, OIImage] = {}
    with bbox_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            name = mid_to_name.get(row["LabelName"])
            if name is None:
                continue
            img = images.setdefault(row["ImageID"], OIImage(row["ImageID"], split))
            img.boxes.append(
                OIBox(
                    label=name,
                    xmin=float(row["XMin"]),
                    xmax=float(row["XMax"]),
                    ymin=float(row["YMin"]),
                    ymax=float(row["YMax"]),
                    group_of=row["IsGroupOf"] == "1",
                    depiction=row["IsDepiction"] == "1",
                    occluded=row["IsOccluded"] == "1",
                    truncated=row["IsTruncated"] == "1",
                )
            )
    with labels_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            name = mid_to_name.get(row["LabelName"])
            if name is None:
                continue
            labelled = images.get(row["ImageID"])
            if labelled is None and include_verified_negatives and row["Confidence"] == "0":
                labelled = images.setdefault(row["ImageID"], OIImage(row["ImageID"], split))
            if labelled is not None:
                labelled.verified[name] = row["Confidence"] == "1"
    with images_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            described = images.get(row["ImageID"])
            if described is not None:
                described.meta = dict(row)
    for img in images.values():
        # Boxes are drawn only for verified-present classes; record that if the label is missing.
        for box in img.boxes:
            img.verified.setdefault(box.label, True)
    # A box-less image kept for its negative labels must not also claim a target object (it would
    # have no box to match); none do in the V7 validation/test labels, but keep that guaranteed.
    return {k: img for k, img in images.items() if img.boxes or not any(img.verified.values())}


def exif_orientation(path: Path) -> int:
    with Image.open(path) as im:
        value = im.getexif().get(_EXIF_ORIENTATION, 1)
    return int(value)


def build(config: BuildConfig, downloader: Downloader | None = None) -> dict[str, Any]:
    """Download the images and write labels, ``annotations.json``, ``manifest.csv``, ``data.yaml``.

    Returns per-split/per-class counts (also written to ``summary.json``).
    """
    names = [
        "oidv7-class-descriptions-boxable.csv",
        "bbox_labels_600_hierarchy.json",
        *(
            f"{s}-{kind}.csv"
            for s in config.splits
            for kind in (
                "annotations-bbox",
                "annotations-human-imagelabels-boxable",
                "images-with-rotation",
            )
        ),
    ]
    class_index = {name: i for i, name in enumerate(config.classes)}
    with downloader or Downloader(config.policy) as dl:
        meta = ensure_metadata(names, config.metadata_dir, config.reuse_dirs, dl)
        mid_to_name = read_class_mids(meta["oidv7-class-descriptions-boxable.csv"], config.classes)
        hierarchy = load_hierarchy(
            meta["bbox_labels_600_hierarchy.json"], mid_to_name, config.classes
        )
        images: list[OIImage] = []
        for split in config.splits:
            found = collect_images(
                split,
                meta[f"{split}-annotations-bbox.csv"],
                meta[f"{split}-annotations-human-imagelabels-boxable.csv"],
                meta[f"{split}-images-with-rotation.csv"],
                mid_to_name,
                config.include_verified_negatives,
            )
            images.extend(found[k] for k in sorted(found))
        if config.max_images is not None:
            images = images[: config.max_images]
        items = [
            DownloadItem(
                IMAGE_URL.format(split=img.split, image_id=img.image_id),
                config.out / "images" / img.split / f"{img.image_id}.jpg",
            )
            for img in images
        ]
        results = dl.fetch_all(
            items, on_result=lambda r: log.info("%s %s (%d bytes)", r.status, r.dest.name, r.bytes)
        )

    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for img, item, result in zip(images, items, results, strict=True):
        decoded = load_image(item.dest)
        height, width = decoded.shape[:2]
        boxes = [
            (b.xmin * width, b.ymin * height, b.xmax * width, b.ymax * height) for b in img.boxes
        ]
        labels = [class_index[b.label] for b in img.boxes]
        rel = item.dest.relative_to(config.out).as_posix()
        label_path = config.out / "labels" / img.split / f"{img.image_id}.txt"
        write_labels(label_path, labels, as_boxes(boxes), width, height)
        positive, negative = expand_verified_labels(img.verified, hierarchy)
        records.append(
            {
                "image_id": img.image_id,
                "split": img.split,
                "file": rel,
                "width": width,
                "height": height,
                "boxes": [[round(v, 3) for v in b] for b in boxes],
                "labels": labels,
                "group_of": [b.group_of for b in img.boxes],
                "depiction": [b.depiction for b in img.boxes],
                "occluded": [b.occluded for b in img.boxes],
                "truncated": [b.truncated for b in img.boxes],
                "verified_positive": sorted(class_index[n] for n in positive),
                "verified_negative": sorted(class_index[n] for n in negative),
            }
        )
        manifest.append(
            {
                "image_id": img.image_id,
                "split": img.split,
                "file": rel,
                "bytes": result.bytes,
                "sha256": result.sha256,
                "width": width,
                "height": height,
                "num_boxes": len(boxes),
                "rotation": img.meta.get("Rotation", ""),
                "exif_orientation": exif_orientation(item.dest),
                "license": img.meta.get("License", ""),
                "author": img.meta.get("Author", ""),
                "author_profile_url": img.meta.get("AuthorProfileURL", ""),
                "original_landing_url": img.meta.get("OriginalLandingURL", ""),
                "source_url": item.url,
            }
        )

    description = (
        "Open Images V7 validation+test images with Traffic sign / Stop sign / Traffic light boxes"
    )
    if config.include_verified_negatives:
        description += (
            ", plus images without such boxes that have a human-verified negative label for one"
            " of these classes (verified negatives)"
        )
    annotations = {
        "info": {
            "description": description + "; boxes in absolute pixels (x1, y1, x2, y2).",
            "annotations_license": "CC BY 4.0 (Google LLC); images: see manifest.csv",
            "class_parents": {k: list(v) for k, v in hierarchy.parents.items()},
        },
        "classes": list(config.classes),
        "images": records,
    }
    config.out.mkdir(parents=True, exist_ok=True)
    (config.out / "annotations.json").write_text(json.dumps(annotations, indent=1))
    write_manifest(config.out / "manifest.csv", manifest)
    write_data_yaml(
        config.out / "data.yaml",
        list(config.classes),
        {"val": [f"images/{s}" for s in config.splits]},
    )
    summary = summarize(records, list(config.classes))
    (config.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if config.manifest_copy_dir is not None:
        config.manifest_copy_dir.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.csv", "annotations.json", "summary.json"):
            shutil.copyfile(config.out / name, config.manifest_copy_dir / name)
    return summary


def write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image_id"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: Sequence[Mapping[str, Any]], classes: Sequence[str]) -> dict[str, Any]:
    """Per split and class: images, boxes, boxes under 1% of the image area, group-of boxes."""
    out: dict[str, Any] = {}
    for split in sorted({r["split"] for r in records}):
        rows = [r for r in records if r["split"] == split]
        per_class: dict[str, dict[str, int]] = {}
        for ci, name in enumerate(classes):
            n_img = n_box = n_small = n_group = 0
            for r in rows:
                idx = [i for i, lab in enumerate(r["labels"]) if lab == ci]
                n_img += bool(idx)
                n_box += len(idx)
                area = r["width"] * r["height"]
                for i in idx:
                    x1, y1, x2, y2 = r["boxes"][i]
                    n_small += (x2 - x1) * (y2 - y1) < 0.01 * area
                    n_group += r["group_of"][i]
            per_class[name] = {
                "images": n_img,
                "boxes": n_box,
                "boxes_lt_1pct_area": n_small,
                "group_of": n_group,
            }
        out[split] = {"images": len(rows), "classes": per_class}
    return out
