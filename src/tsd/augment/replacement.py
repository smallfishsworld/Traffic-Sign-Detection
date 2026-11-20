"""Replacement (copy-paste) augmentation for class balancing.

TT100K is long-tailed: of its 201 sign classes, 84 have fewer than 10 image instances, 72 have
10-100 and 45 more than 100 (2024 course report, section 4.2.4). The 2024 project pasted standard
sign templates over existing signs so that rare classes such as ``pl15`` appear more often. This
implementation:

* samples the pasted class from an inverse-frequency distribution over classes that have
  templates (``p(c) ~ (count_c + 1) ** -power``);
* **replaces** existing signs: only with a template of the same outline (disc for disc, triangle
  for triangle, see ``templates.shape_for_class``), only if the source class is more common than
  the pasted one (the rarest classes are never overwritten), and only if the new silhouette,
  scaled to the old box (1.0-1.1x) when the aspect ratios are compatible, covers at least
  ``min_coverage`` (95%) of the old sign's outline; the label's class and box are updated to the
  pasted template's silhouette;
* **adds** signs at sampled locations that do not overlap existing labels, at a scale taken from
  the signs already in the image (or a configured pixel range);
* perturbs each template photometrically (brightness, contrast, colour, blur, noise and a
  non-uniform illumination gradient), optionally pulls its brightness towards the scene it is
  pasted into, and feathers the alpha edge so the paste boundary is not a trivial cue.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from tsd.augment.photometric import F32, PhotometricJitter, jitter_bgr
from tsd.augment.templates import BGRA, Shape, SignTemplate, silhouette_mask
from tsd.boxes import FloatArray, as_boxes, box_ioa
from tsd.io import Image


@dataclass(frozen=True)
class PasteOp:
    kind: Literal["replace", "add"]
    class_id: int
    box: tuple[float, float, float, float]
    replaced_index: int | None = None


@dataclass(frozen=True)
class AugmentResult:
    image: Image
    boxes: FloatArray
    class_ids: NDArray[np.int64]
    ops: tuple[PasteOp, ...]


def class_sampling_weights(
    counts: Mapping[int, int], available: Iterable[int], power: float = 1.0
) -> dict[int, float]:
    """Inverse-frequency sampling distribution over the classes in ``available``."""
    classes = sorted(set(available))
    if not classes:
        raise ValueError("no classes with templates")
    raw = np.array([(counts.get(c, 0) + 1.0) ** -power for c in classes])
    probs = raw / raw.sum()
    return {c: float(p) for c, p in zip(classes, probs, strict=True)}


def resize_template(bgra: BGRA, width: int, height: int) -> BGRA:
    shrinking = width < bgra.shape[1] or height < bgra.shape[0]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return np.asarray(cv2.resize(bgra, (width, height), interpolation=interp), dtype=np.uint8)


Window = tuple[tuple[slice, slice], tuple[slice, slice]]


def clip_window(
    image_shape: tuple[int, ...], patch_shape: tuple[int, ...], x0: int, y0: int
) -> Window | None:
    """(image slices, patch slices) of a patch placed at ``(x0, y0)``, clipped to the image;
    None if they do not overlap."""
    height, width = image_shape[:2]
    h, w = patch_shape[:2]
    ix0, iy0, ix1, iy1 = max(x0, 0), max(y0, 0), min(x0 + w, width), min(y0 + h, height)
    if ix0 >= ix1 or iy0 >= iy1:
        return None
    px, py = ix0 - x0, iy0 - y0
    return (
        (slice(iy0, iy1), slice(ix0, ix1)),
        (slice(py, py + iy1 - iy0), slice(px, px + ix1 - ix0)),
    )


def alpha_blend(image: Image, patch: F32, alpha: F32, x0: int, y0: int) -> None:
    """Blend ``patch`` (float BGR) into ``image`` in place at ``(x0, y0)``, clipped to the image."""
    window = clip_window(image.shape, alpha.shape, x0, y0)
    if window is None:
        return
    iw, pw = window
    a = alpha[pw][..., None]
    region = image[iw].astype(np.float32)
    image[iw] = np.clip(np.rint(a * patch[pw] + (1.0 - a) * region), 0, 255).astype(np.uint8)


def coverage(alpha: F32, x0: int, y0: int, mask: np.ndarray, mx0: int, my0: int) -> float:
    """Share of the ``mask`` pixels (placed at ``(mx0, my0)``) under the pasted silhouette
    ``alpha`` (placed at ``(x0, y0)``, alpha >= 0.5)."""
    total = int(mask.sum())
    if total == 0:
        return 1.0
    window = clip_window(mask.shape, alpha.shape, x0 - mx0, y0 - my0)
    if window is None:
        return 0.0
    mw, aw = window
    return float((mask[mw] & (alpha[aw] >= 0.5)).sum()) / total


def crop_to_alpha(bgra: BGRA, threshold: int = 128) -> BGRA:
    """``bgra`` cropped to the bounding box of its pixels with alpha >= ``threshold`` (unchanged
    if there are none)."""
    ys, xs = np.nonzero(bgra[..., 3] >= threshold)
    if len(xs) == 0:
        return bgra
    return bgra[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def alpha_bbox(
    alpha: F32, x0: int, y0: int, width: int, height: int, threshold: float = 0.5
) -> tuple[float, float, float, float] | None:
    """Tight box of the pasted silhouette (alpha >= threshold), clipped to the image."""
    ys, xs = np.nonzero(alpha >= threshold)
    if len(xs) == 0:
        return None
    x1 = max(float(x0 + xs.min()), 0.0)
    y1 = max(float(y0 + ys.min()), 0.0)
    x2 = min(float(x0 + xs.max() + 1), float(width))
    y2 = min(float(y0 + ys.max() + 1), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _luminance(bgr: F32) -> F32:
    return np.asarray(bgr @ np.array([0.114, 0.587, 0.299], dtype=np.float32), dtype=np.float32)


@dataclass
class ReplacementAugment:
    """Copy-paste class balancing with sign templates.

    ``class_counts`` are training-set instance counts per class id; they define the sampling
    distribution. Call the object with a BGR image, ``xyxy`` boxes and class ids.
    """

    templates: Mapping[int, Sequence[SignTemplate]]
    class_counts: Mapping[int, int]
    p_replace: float = 0.5
    p_add: float = 0.5
    max_add: int = 2
    power: float = 1.0
    aspect_tolerance: float = 1.35
    cover_scale: tuple[float, float] = (1.0, 1.1)
    add_size_range: tuple[float, float] = (16.0, 96.0)
    max_overlap: float = 0.05
    min_size: int = 6
    min_coverage: float = 0.95
    match_brightness: float = 0.5
    feather: bool = True
    jitter: PhotometricJitter = field(default_factory=PhotometricJitter)
    _classes: list[int] = field(init=False, repr=False)
    _probs: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        weights = class_sampling_weights(
            self.class_counts, [c for c, t in self.templates.items() if t], self.power
        )
        self._classes = list(weights)
        self._probs = np.array(list(weights.values()))

    def __call__(
        self,
        image: Image,
        boxes: FloatArray,
        class_ids: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> AugmentResult:
        img = image.copy()
        out_boxes = as_boxes(boxes).copy()
        out_cls = np.asarray(class_ids, dtype=np.int64).copy()
        ops: list[PasteOp] = []
        for i in range(len(out_boxes)):
            if rng.random() < self.p_replace:
                op = self._replace(img, out_boxes, out_cls, i, rng)
                if op is not None:
                    ops.append(op)
        for _ in range(self.max_add):
            if rng.random() < self.p_add:
                op, out_boxes, out_cls = self._add(img, out_boxes, out_cls, rng)
                if op is not None:
                    ops.append(op)
        return AugmentResult(img, out_boxes, out_cls, tuple(ops))

    def shape_of(self, class_id: int) -> Shape:
        """A class's sign outline: that of its first template (``unknown`` without one)."""
        options = self.templates.get(class_id) or []
        return options[0].shape if options else "unknown"

    def sample_class(
        self, rng: np.random.Generator, exclude: int | None = None, shape: Shape | None = None
    ) -> int | None:
        """A class from the inverse-frequency distribution, optionally without ``exclude`` and
        only among classes whose templates have the outline ``shape``."""
        keep = np.array(
            [
                c != exclude and (shape is None or shape == "unknown" or self.shape_of(c) == shape)
                for c in self._classes
            ]
        )
        if not keep.any():
            return None
        classes = [c for c, k in zip(self._classes, keep, strict=True) if k]
        probs = self._probs[keep] / self._probs[keep].sum()
        return int(classes[int(rng.choice(len(classes), p=probs))])

    def _template(self, class_id: int, rng: np.random.Generator) -> SignTemplate:
        """A random template of ``class_id``, cropped to its silhouette, so that stretching it to
        a box makes the sign itself (not a canvas with margins) fill the box."""
        options = self.templates[class_id]
        tpl = options[int(rng.integers(len(options)))]
        return replace(tpl, bgra=crop_to_alpha(tpl.bgra))

    def _replace(
        self,
        img: Image,
        boxes: FloatArray,
        cls: NDArray[np.int64],
        index: int,
        rng: np.random.Generator,
    ) -> PasteOp | None:
        x1, y1, x2, y2 = boxes[index]
        w, h = x2 - x1, y2 - y1
        if min(w, h) < self.min_size:
            return None
        source = int(cls[index])
        shape = self.shape_of(source)
        target = self.sample_class(rng, exclude=source, shape=shape)
        if target is None:
            return None
        if self.class_counts.get(source, 0) <= self.class_counts.get(target, 0):
            return None  # never overwrite a class that is rarer than the pasted one
        tpl = self._template(target, rng)
        tpl_aspect = tpl.bgra.shape[1] / tpl.bgra.shape[0]
        if max(tpl_aspect, w / h) / min(tpl_aspect, w / h) > self.aspect_tolerance:
            return None
        scale = rng.uniform(*self.cover_scale)
        tw, th = max(round(w * scale), 1), max(round(h * scale), 1)
        x0, y0 = round((x1 + x2) / 2 - tw / 2), round((y1 + y2) / 2 - th / 2)
        old = (silhouette_mask(shape, max(round(w), 1), max(round(h), 1)), round(x1), round(y1))
        new_box = self._paste(img, tpl, x0, y0, tw, th, rng, cover=old)
        if new_box is None:
            return None
        boxes[index] = new_box
        cls[index] = target
        return PasteOp("replace", target, new_box, index)

    def _add(
        self,
        img: Image,
        boxes: FloatArray,
        cls: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> tuple[PasteOp | None, FloatArray, NDArray[np.int64]]:
        target = self.sample_class(rng)
        if target is None:
            return None, boxes, cls
        tpl = self._template(target, rng)
        height, width = img.shape[:2]
        if len(boxes):
            ref = boxes[int(rng.integers(len(boxes)))]
            size = float(np.sqrt((ref[2] - ref[0]) * (ref[3] - ref[1]))) * rng.uniform(0.7, 1.3)
        else:
            lo, hi = self.add_size_range
            size = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        aspect = tpl.bgra.shape[1] / tpl.bgra.shape[0]
        tw = max(round(size * np.sqrt(aspect)), self.min_size)
        th = max(round(size / np.sqrt(aspect)), self.min_size)
        if tw >= width or th >= height:
            return None, boxes, cls
        for _ in range(20):
            x0, y0 = int(rng.integers(0, width - tw + 1)), int(rng.integers(0, height - th + 1))
            cand = np.array([[x0, y0, x0 + tw, y0 + th]], dtype=np.float64)
            if len(boxes) and (
                box_ioa(cand, boxes).max() > self.max_overlap
                or box_ioa(boxes, cand).max() > self.max_overlap
            ):
                continue
            new_box = self._paste(img, tpl, x0, y0, tw, th, rng)
            if new_box is None:
                return None, boxes, cls
            boxes = np.vstack([boxes, np.array(new_box)[None]])
            cls = np.append(cls, target)
            return PasteOp("add", target, new_box), boxes, cls
        return None, boxes, cls

    def _paste(
        self,
        img: Image,
        tpl: SignTemplate,
        x0: int,
        y0: int,
        tw: int,
        th: int,
        rng: np.random.Generator,
        cover: tuple[np.ndarray, int, int] | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Paste ``tpl`` resized to ``tw`` x ``th`` at ``(x0, y0)``; with ``cover`` (an old sign's
        outline mask and position), only if the new silhouette covers ``min_coverage`` of it."""
        resized = resize_template(tpl.bgra, tw, th)
        alpha = resized[..., 3].astype(np.float32) / 255.0
        if cover is not None and coverage(alpha, x0, y0, *cover) < self.min_coverage:
            return None
        patch = jitter_bgr(resized[..., :3].astype(np.float32), self.jitter, rng)
        if self.match_brightness > 0:
            patch = self._match_brightness(img, patch, alpha, x0, y0)
        box = alpha_bbox(alpha, x0, y0, img.shape[1], img.shape[0])
        if box is None:
            return None
        if self.feather and min(tw, th) >= 12:  # zero outside: opaque canvas edges feather too
            alpha = np.asarray(
                cv2.GaussianBlur(alpha, (3, 3), 0, borderType=cv2.BORDER_CONSTANT),
                dtype=np.float32,
            )
        alpha_blend(img, patch, alpha, x0, y0)
        return box

    def _match_brightness(self, img: Image, patch: F32, alpha: F32, x0: int, y0: int) -> F32:
        """Scale the patch so its mean luminance moves towards the scene region it covers."""
        window = clip_window(img.shape, alpha.shape, x0, y0)
        if window is None:
            return patch
        iw, pw = window
        a = alpha[pw]
        if a.sum() < 1:
            return patch
        scene = _luminance(img[iw].astype(np.float32))
        tpl = _luminance(patch[pw])
        scene_mean = float((scene * a).sum() / a.sum())
        tpl_mean = float((tpl * a).sum() / a.sum())
        if tpl_mean < 1.0:
            return patch
        gain = float(np.clip((scene_mean / tpl_mean) ** self.match_brightness, 0.5, 2.0))
        return np.asarray(np.clip(patch * gain, 0, 255), dtype=np.float32)
