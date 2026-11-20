"""Sign templates for the replacement augmentation.

TT100K ships standard sign images ("marks") that the 2024 project pasted into scenes. They are not
available here, so this module also renders simple synthetic templates (a speed-limit disc and a
warning triangle) that are good enough for tests and for illustrating the augmentation.
Templates are BGRA ``uint8`` arrays whose alpha channel marks the sign's silhouette.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

BGRA = NDArray[np.uint8]
Shape = Literal["circle", "triangle", "inverted_triangle", "octagon", "rect", "unknown"]


def shape_for_class(name: str) -> Shape:
    """Sign outline from a TT100K class name: ``ps`` (stop) octagon, ``pg`` (give way) inverted
    triangle, other ``p*`` (prohibitory, speed limits ``pl*``, ``pm*``, ...) circles, ``w*``
    (warning) triangles, ``ip`` (pedestrian crossing) a rectangle, other ``i*`` (mandatory)
    circles; anything else is unknown."""
    if name == "ps":
        return "octagon"
    if name == "pg":
        return "inverted_triangle"
    if name.startswith("w"):
        return "triangle"
    if name == "ip":
        return "rect"
    if name.startswith(("p", "i")):
        return "circle"
    return "unknown"


def silhouette_mask(shape: Shape, width: int, height: int) -> NDArray[np.bool_]:
    """The outline ``shape`` inscribed in a ``width`` x ``height`` box (``unknown``: the box)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    w, h = width - 1, height - 1
    if shape == "circle":
        cv2.ellipse(mask, (w // 2, h // 2), (max(w // 2, 1), max(h // 2, 1)), 0, 0, 360, 1, -1)
    elif shape == "triangle":
        cv2.fillPoly(mask, [np.array([[w // 2, 0], [w, h], [0, h]], dtype=np.int32)], 1)
    elif shape == "inverted_triangle":
        cv2.fillPoly(mask, [np.array([[0, 0], [w, 0], [w // 2, h]], dtype=np.int32)], 1)
    elif shape == "octagon":
        dx, dy = round(w * 0.29), round(h * 0.29)
        pts = [[dx, 0], [w - dx, 0], [w, dy], [w, h - dy], [w - dx, h], [dx, h], [0, h - dy],
               [0, dy]]  # fmt: skip
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
    else:
        mask[:] = 1
    return mask.astype(bool)


@dataclass(frozen=True)
class SignTemplate:
    class_id: int
    bgra: BGRA
    name: str = ""
    shape: Shape = "unknown"

    def __post_init__(self) -> None:
        if self.bgra.ndim != 3 or self.bgra.shape[2] != 4 or self.bgra.dtype != np.uint8:
            raise ValueError("templates must be HxWx4 uint8 (BGRA)")


def load_templates(directory: Path, class_names: Sequence[str]) -> dict[int, list[SignTemplate]]:
    """Load ``<class_name>.png`` / ``<class_name>_<k>.png`` files with an alpha channel.

    Files whose class is not in ``class_names`` are ignored; images without alpha get an opaque
    alpha channel.
    """
    index = {name: i for i, name in enumerate(class_names)}
    out: dict[int, list[SignTemplate]] = {}
    for path in sorted(directory.glob("*.png")):
        stem = path.stem
        name = stem if stem in index else stem.rsplit("_", 1)[0]
        if name not in index:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot read template {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra = np.ascontiguousarray(img, dtype=np.uint8)
        if int(bgra[..., 3].min()) == 255:
            log.warning("template %s has no transparent pixels: pasted as a full rectangle", path)
        out.setdefault(index[name], []).append(
            SignTemplate(index[name], bgra, name, shape_for_class(name))
        )
    return out


def render_speed_limit(text: str, size: int = 128) -> BGRA:
    """A prohibitory disc (red ring, white face, black text), e.g. ``"15"`` for TT100K pl15."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    c = size // 2
    r = size // 2 - 1
    cv2.circle(img, (c, c), r, (40, 40, 220, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, (c, c), int(r * 0.78), (255, 255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    _put_centered_text(img, text, (0, 0, 0, 255), max_width=int(r * 1.2))
    return img


def render_warning(size: int = 128) -> BGRA:
    """A warning triangle (black border, yellow face, black exclamation mark)."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    h = int(size * 0.87)
    top = (size - h) // 2
    outer = np.array([[size // 2, top], [size - 1, top + h], [0, top + h]], dtype=np.int32)
    cv2.fillPoly(img, [outer], (0, 0, 0, 255), lineType=cv2.LINE_AA)
    inset = size * 0.12
    centroid = outer.mean(axis=0)
    inner = (centroid + (outer - centroid) * (1 - 2 * inset / size)).astype(np.int32)
    cv2.fillPoly(img, [inner], (0, 200, 255, 255), lineType=cv2.LINE_AA)
    _put_centered_text(img, "!", (0, 0, 0, 255), max_width=int(size * 0.2), dy=int(size * 0.1))
    return img


def synthetic_templates(class_names: Mapping[int, str]) -> dict[int, list[SignTemplate]]:
    """One synthetic template per class: speed-limit discs for ``pl<N>``, triangles otherwise
    (with the matching ``shape``)."""
    out: dict[int, list[SignTemplate]] = {}
    for cid, name in class_names.items():
        if name.startswith("pl") and name[2:].isdigit():
            out[cid] = [SignTemplate(cid, render_speed_limit(name[2:]), name, "circle")]
        else:
            out[cid] = [SignTemplate(cid, render_warning(), name, "triangle")]
    return out


def _put_centered_text(
    img: BGRA, text: str, color: tuple[int, int, int, int], max_width: int, dy: int = 0
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, img.shape[0] // 24)
    (tw, _), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = max_width / max(tw, 1)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = ((img.shape[1] - tw) // 2, (img.shape[0] + th) // 2 + dy)
    cv2.putText(img, text, org, font, scale, color, thickness, lineType=cv2.LINE_AA)
