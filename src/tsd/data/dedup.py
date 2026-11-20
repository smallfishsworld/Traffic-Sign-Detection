"""Near-duplicate detection between two image sets with difference hashes (dHash).

Used to check that the INT8 calibration images (COCO128) share no photo with the Open Images
evaluation set: both datasets were collected from Flickr, so "different dataset" alone does not
guarantee disjoint images.

The check has two stages:

1. **Candidates**: pairs whose 64-bit dHash differs in at most ``threshold`` (10) bits. dHash
   survives resizing and re-encoding (simulated copies of the 219 evaluation images, resized to
   640 px and re-saved at JPEG quality 70, stay within 5 bits with every resize recipe tried),
   but photos dominated by smooth sky also land close, because their horizontal gradients
   carry little information.
2. **Confirmation**: a candidate is a duplicate if the two images have the same aspect ratio
   (within 2%; COCO and the Open Images S3 copies are both uniformly rescaled, never cropped) and
   either the 64-bit hashes are a strong match (at most ``strong_threshold``, 5, bits: what every
   simulated copy achieves) or the 256-bit dHash differs in at most ``fine_threshold`` (40)
   bits. The 256-bit hash alone is not robust enough to veto a match: JPEG re-encoding moves it
   by up to 41 bits on low-texture photos (``bench/dedup_copy_sim.py``, raw results in
   ``bench/results/dedup_copy_sim.json``), while unrelated pairs are at 53 or more there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from tsd.io import Image, load_image

ASPECT_TOLERANCE = 0.02


def dhash(image: Image, hash_size: int = 8) -> int:
    """Difference hash: signs of horizontal gradients of a small grey thumbnail."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class ImageHashes:
    name: str
    coarse: int  # 64-bit dHash
    fine: int  # 256-bit dHash
    aspect: float  # width / height

    @classmethod
    def of(cls, path: Path) -> ImageHashes:
        image = load_image(path)
        h, w = image.shape[:2]
        return cls(path.name, dhash(image, 8), dhash(image, 16), w / h)


@dataclass(frozen=True)
class CandidatePair:
    a: str
    b: str
    distance: int
    fine_distance: int
    aspect_log_ratio: float
    duplicate: bool


@dataclass(frozen=True)
class OverlapReport:
    pairs_compared: int
    min_distance: int
    closest_pair: tuple[str, str]
    threshold: int
    fine_threshold: int
    candidates: list[CandidatePair] = field(default_factory=list)

    @property
    def duplicates(self) -> list[CandidatePair]:
        return [c for c in self.candidates if c.duplicate]


def _pairwise_distance(a: Sequence[int], b: Sequence[int]) -> NDArray[np.int64]:
    ha = np.array(a, dtype=np.uint64)
    hb = np.array(b, dtype=np.uint64)
    xor = ha[:, None] ^ hb[None, :]
    bits = np.unpackbits(xor.view(np.uint8).reshape(len(a), len(b), 8), axis=-1)
    out: NDArray[np.int64] = bits.sum(axis=-1, dtype=np.int64)
    return out


def is_duplicate(
    coarse: int, fine: int, aspect_log_ratio: float, strong_threshold: int = 5,
    fine_threshold: int = 40,
) -> bool:  # fmt: skip
    """The confirmation rule of the module docstring (for a candidate pair)."""
    return aspect_log_ratio <= ASPECT_TOLERANCE and (
        coarse <= strong_threshold or fine <= fine_threshold
    )


def cross_overlap(
    a: Sequence[Path],
    b: Sequence[Path],
    threshold: int = 10,
    fine_threshold: int = 40,
    strong_threshold: int = 5,
) -> OverlapReport:
    """Compare every image of ``a`` with every image of ``b`` (see the module docstring)."""
    if not a or not b:
        raise ValueError("both image sets must be non-empty")
    ha = [ImageHashes.of(p) for p in a]
    hb = [ImageHashes.of(p) for p in b]
    dist = _pairwise_distance([h.coarse for h in ha], [h.coarse for h in hb])
    i, j = (int(k) for k in np.unravel_index(int(dist.argmin()), dist.shape))
    candidates = []
    for ci, cj in zip(*np.nonzero(dist <= threshold), strict=True):
        x, y = ha[int(ci)], hb[int(cj)]
        fine = hamming(x.fine, y.fine)
        aspect = abs(math.log(x.aspect / y.aspect))
        coarse = int(dist[ci, cj])
        duplicate = is_duplicate(coarse, fine, aspect, strong_threshold, fine_threshold)
        candidates.append(CandidatePair(x.name, y.name, coarse, fine, round(aspect, 4), duplicate))
    return OverlapReport(
        pairs_compared=int(dist.size),
        min_distance=int(dist[i, j]),
        closest_pair=(ha[i].name, hb[j].name),
        threshold=threshold,
        fine_threshold=fine_threshold,
        candidates=candidates,
    )
