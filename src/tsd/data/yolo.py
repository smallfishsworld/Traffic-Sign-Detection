"""Reading and writing YOLO-format datasets (one ``.txt`` label file per image, plus ``data.yaml``).

A label line is ``<class_id> <cx> <cy> <w> <h>`` with the box centre and size normalised by the
image width and height. Ultralytics finds an image's label file by replacing the ``images``
path component with ``labels`` and the extension with ``.txt``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from tsd.boxes import FloatArray, as_boxes, xywhn_to_xyxy, xyxy_to_xywhn

IntArray = NDArray[np.int64]


def format_label_lines(
    class_ids: Sequence[int] | IntArray, boxes_xyxy: FloatArray, width: int, height: int
) -> str:
    """Return YOLO label-file text for ``boxes_xyxy`` (absolute pixels) in a ``width x height``
    image. Coordinates are written with 6 decimals (sub-pixel even for 2048 px images)."""
    norm = xyxy_to_xywhn(boxes_xyxy, width, height)
    lines = [
        f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, (cx, cy, w, h) in zip(class_ids, norm, strict=True)
    ]
    return "".join(line + "\n" for line in lines)


def write_labels(
    path: Path,
    class_ids: Sequence[int] | IntArray,
    boxes_xyxy: FloatArray,
    width: int,
    height: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_label_lines(class_ids, boxes_xyxy, width, height))


def read_labels(path: Path, width: int, height: int) -> tuple[IntArray, FloatArray]:
    """Read a YOLO label file; returns ``(class_ids, boxes_xyxy)`` in absolute pixels.

    A missing file means "no objects" (a background image), as in Ultralytics.
    """
    if not path.exists():
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
    if any(len(r) != 5 for r in rows):
        raise ValueError(f"{path}: every label line must have 5 fields")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0].astype(np.int64), xywhn_to_xyxy(as_boxes(arr[:, 1:]), width, height)


def write_data_yaml(
    path: Path,
    names: Sequence[str],
    splits: Mapping[str, str | Sequence[str]],
    root: Path | None = None,
) -> None:
    """Write an Ultralytics ``data.yaml``.

    ``splits`` maps split names (``train``/``val``/``test``) to image directories relative to
    ``root`` (default: the directory containing ``data.yaml``).
    """
    root = root or path.parent
    doc: dict[str, object] = {"path": str(root.resolve())}
    for split, rel in splits.items():
        doc[split] = rel if isinstance(rel, str) else list(rel)
    doc["names"] = dict(enumerate(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
