"""How far simulated copies and unrelated pairs land on the near-duplicate hashes.

    nice -n 19 uv run python bench/dedup_copy_sim.py

For every evaluation image, copies are made with several resize + JPEG recipes (longest side
640 px, quality 70; plus re-encoding alone), and the 64-bit and 256-bit dHash distances between
original and copy are recorded. Unrelated pairs are every (evaluation, COCO128 calibration)
pair. The output shows the margins behind ``tsd.data.dedup``'s thresholds and how the old rule
(256-bit hash must confirm) and the current one (a strong 64-bit match suffices) decide.
Output: ``bench/results/dedup_copy_sim.json``. Not a timing measurement (no lock needed).
"""

from __future__ import annotations

import argparse
import io
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from tsd.data.dedup import ASPECT_TOLERANCE, _pairwise_distance, dhash, hamming, is_duplicate
from tsd.io import Image, decode_image, load_image
from tsd.provenance import provenance

REPO = Path(__file__).resolve().parents[1]


def _cv2_copy(interpolation: int) -> Callable[[Image], Image]:
    def copy(image: Image) -> Image:
        h, w = image.shape[:2]
        r = 640 / max(h, w)
        small = cv2.resize(image, (round(w * r), round(h * r)), interpolation=interpolation)
        ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        assert ok
        return decode_image(enc.tobytes())

    return copy


def _pil_copy(resample: PILImage.Resampling | None) -> Callable[[Image], Image]:
    def copy(image: Image) -> Image:
        pil = PILImage.fromarray(image[..., ::-1])
        if resample is not None:
            r = 640 / max(pil.size)
            pil = pil.resize((round(pil.width * r), round(pil.height * r)), resample)
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=70)
        return decode_image(buf.getvalue())

    return copy


RECIPES: dict[str, Callable[[Image], Image]] = {
    "cv2_area_640_q70": _cv2_copy(cv2.INTER_AREA),
    "cv2_linear_640_q70": _cv2_copy(cv2.INTER_LINEAR),
    "cv2_cubic_640_q70": _cv2_copy(cv2.INTER_CUBIC),
    "pil_lanczos_640_q70": _pil_copy(PILImage.Resampling.LANCZOS),
    "pil_bilinear_640_q70": _pil_copy(PILImage.Resampling.BILINEAR),
    "pil_q70_no_resize": _pil_copy(None),
}


def hashes(image: Image) -> tuple[int, int, float]:
    h, w = image.shape[:2]
    return dhash(image, 8), dhash(image, 16), w / h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-images", type=Path, default=REPO / "data/openimages-traffic/images")
    parser.add_argument("--calibration", type=Path, default=REPO / "data/calibration/coco128")
    parser.add_argument("--out", type=Path, default=REPO / "bench/results/dedup_copy_sim.json")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    eval_paths = sorted(args.eval_images.rglob("*.jpg"))
    calib_paths = sorted(args.calibration.rglob("*.jpg"))
    originals = {p.name: hashes(load_image(p)) for p in eval_paths}
    copies: dict[str, dict[str, Any]] = {}
    for name, recipe in RECIPES.items():
        coarse, fine, old_rule, new_rule = [], [], 0, 0
        worst = ("", -1)
        for p in eval_paths:
            c0, f0, a0 = originals[p.name]
            c1, f1, a1 = hashes(recipe(load_image(p)))
            dc, df = hamming(c0, c1), hamming(f0, f1)
            aspect = abs(math.log(a0 / a1))
            coarse.append(dc)
            fine.append(df)
            old_rule += int(df <= 40 and aspect <= ASPECT_TOLERANCE and dc <= 10)
            new_rule += int(dc <= 10 and is_duplicate(dc, df, aspect))
            if df > worst[1]:
                worst = (p.name, df)
        copies[name] = {
            "images": len(eval_paths),
            "coarse_max": max(coarse),
            "coarse_p99": float(np.percentile(coarse, 99)),
            "fine_max": max(fine),
            "fine_p99": float(np.percentile(fine, 99)),
            "fine_worst_image": worst[0],
            "flagged_old_rule": old_rule,
            "flagged_current_rule": new_rule,
        }
        print(name, copies[name])
    calib = [hashes(load_image(p)) for p in calib_paths]
    ev = list(originals.values())
    coarse_d = _pairwise_distance([h[0] for h in ev], [h[0] for h in calib])
    fine_d = _pairwise_distance_wide([h[1] for h in ev], [h[1] for h in calib])
    aspect_d = np.abs(
        np.log(np.array([h[2] for h in ev])[:, None] / np.array([h[2] for h in calib])[None, :])
    )
    same_aspect = aspect_d <= ASPECT_TOLERANCE
    flagged = sum(
        is_duplicate(int(coarse_d[i, j]), int(fine_d[i, j]), float(aspect_d[i, j]))
        for i, j in zip(*np.nonzero(coarse_d <= 10), strict=True)
    )
    unrelated = {
        "pairs": int(coarse_d.size),
        "coarse_min": int(coarse_d.min()),
        "fine_min": int(fine_d.min()),
        "same_aspect_pairs": int(same_aspect.sum()),
        "coarse_min_same_aspect": int(coarse_d[same_aspect].min()) if same_aspect.any() else None,
        "fine_min_same_aspect": int(fine_d[same_aspect].min()) if same_aspect.any() else None,
        "candidates_coarse_le_10": int((coarse_d <= 10).sum()),
        "flagged_current_rule": int(flagged),
    }
    print("unrelated", unrelated)
    doc = {"copies": copies, "unrelated_eval_vs_calibration": unrelated, "provenance": provenance()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")


def _pairwise_distance_wide(a: list[int], b: list[int]) -> np.ndarray:
    """Hamming distances of 256-bit hashes (four 64-bit words each)."""
    total = np.zeros((len(a), len(b)), dtype=np.int64)
    mask = (1 << 64) - 1
    for word in range(4):
        total += _pairwise_distance(
            [(h >> (64 * word)) & mask for h in a], [(h >> (64 * word)) & mask for h in b]
        )
    return total


if __name__ == "__main__":
    main()
