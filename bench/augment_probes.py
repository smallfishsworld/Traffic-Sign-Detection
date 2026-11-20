"""Geometry probes for the two augmentations (synthetic images, no timing, no lock needed).

    nice -n 19 uv run python bench/augment_probes.py

1. Perspective, boxes cut by the image border: 2,000 seeded homographies per corner shift
   (``corner_homography``, 640 x 480 image) applied to a filled 6-40 px box placed within 30 px
   of the left or right border; the label from ``warp_boxes`` (default ``BoxFilter``) is compared
   with the extent of the warped pixels (``cv2.warpPerspective``, nearest neighbour). The rule
   used before the fix (enclosing box of the warped corners, clipped to the image; visibility on
   that box) is re-implemented here and measured on the same seeds.
2. Sign replacement: a sign drawn as a solid colour that no template contains (the old sign's
   silhouette, stretched to a 48 x 48 box) is replaced (``p_replace=1``) on 50 seeds; the share
   of its pixels still showing that colour afterwards is the part of the old sign left visible
   under the new label. Four cases, each with templates of two classes (the old sign's class,
   100 instances, and a rarer one, 10): disc by disc and triangle by triangle (same outline),
   and disc by triangle and triangle by disc (the only rarer template has another outline).

The replacement probe also runs against older code: ``--only replacement`` with ``PYTHONPATH``
pointing at another checkout's ``src`` (for example a ``git archive`` of cfff803, before the
shape-aware replacement; ``--code-label`` records which code that was, because an archive has
no git metadata).

Output: ``bench/results/augment_probes.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import tsd
from tsd.augment.photometric import PhotometricJitter
from tsd.augment.replacement import ReplacementAugment, resize_template
from tsd.augment.templates import SignTemplate, render_speed_limit, render_warning
from tsd.provenance import provenance

REPO = Path(__file__).resolve().parents[1]
W, H = 640, 480
MARKER = (255, 0, 255)


def warp_boxes_before(box: np.ndarray, hm: np.ndarray, f: Any) -> np.ndarray | None:
    """The rule before the fix: clip the enclosing box of the warped corners."""
    from tsd.augment.perspective import corners, transform_points

    pts, w = transform_points(corners(box).reshape(-1, 2), hm)
    if (w <= 1e-9).any():
        return None
    enc = np.r_[pts.min(0), pts.max(0)]
    clipped = np.clip(enc, 0, [W, H, W, H])
    full = np.prod(enc[2:] - enc[:2])
    wh = clipped[2:] - clipped[:2]
    visible = np.prod(wh) / full if full > 0 else 0.0
    aspect = wh.max() / max(wh.min(), 1e-9)
    if (wh < f.min_size).any() or visible < f.min_visibility or aspect > f.max_aspect:
        return None
    return np.asarray(clipped, dtype=float)


def perspective_probe(shift: float, seeds: int) -> dict[str, Any]:
    from tsd.augment.perspective import BoxFilter, corner_homography
    from tsd.augment.perspective import warp_boxes as warp_boxes_now

    f = BoxFilter()
    errors: dict[str, list[tuple[float, float]]] = {"before": [], "after": []}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        s = rng.uniform(6, 40)
        x = rng.choice([rng.uniform(0, 30), rng.uniform(W - 30 - s, W - s)])
        y = rng.uniform(0, H - s)
        box = np.array([[round(x), round(y), round(x + s), round(y + s)]], float)
        img = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = box[0].astype(int)
        img[y1:y2, x1:x2] = 255
        hm = corner_homography(W, H, shift, rng)
        ys, xs = np.nonzero(cv2.warpPerspective(img, hm, (W, H), flags=cv2.INTER_NEAREST) > 127)
        if len(xs) == 0:
            continue
        pix = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
        out, keep = warp_boxes_now(box, hm, W, H, f)
        labels = {"before": warp_boxes_before(box, hm, f), "after": out[0] if keep.any() else None}
        for name, label in labels.items():
            if label is None:
                continue
            area_ratio = np.prod(label[2:] - label[:2]) / np.prod(pix[2:] - pix[:2])
            errors[name].append((float(np.abs(label - pix).max()), float(area_ratio)))
    out_doc: dict[str, Any] = {"corner_shift": shift, "seeds": seeds}
    for name, rows in errors.items():
        e = np.array(rows)
        out_doc[name] = {
            "kept": len(rows),
            "max_abs_error_px": round(float(e[:, 0].max()), 2),
            "share_error_gt_1_5px": round(float(np.mean(e[:, 0] > 1.5)), 4),
            "label_to_pixel_area": {
                "p50": round(float(np.median(e[:, 1])), 3),
                "p99": round(float(np.percentile(e[:, 1], 99)), 3),
                "max": round(float(e[:, 1].max()), 3),
            },
        }
    return out_doc


def _template(class_id: int, bgra: np.ndarray, name: str, shape: Any) -> SignTemplate:
    """A template with its outline where the code under test knows outlines (the replacement
    before 41f6edd had none)."""
    if "shape" in {f.name for f in dataclasses.fields(SignTemplate)}:
        return SignTemplate(class_id, bgra, name, shape)
    return SignTemplate(class_id, bgra, name)


# (case, old sign's template and outline, the rarer class's template and outline)
REPLACEMENT_CASES = (
    ("disc_by_disc", "disc15", "circle", "disc60", "circle"),
    ("triangle_by_triangle", "triangle_a", "triangle", "triangle_b", "triangle"),
    ("disc_by_triangle", "disc15", "circle", "triangle_b", "triangle"),
    ("triangle_by_disc", "triangle_a", "triangle", "disc60", "circle"),
)


def replacement_probe(seeds: int) -> dict[str, Any]:
    art = {
        "disc15": render_speed_limit("15"),
        "disc60": render_speed_limit("60"),
        "triangle_a": render_warning(),
        "triangle_b": render_warning(),
    }
    counts = {0: 100, 1: 10}
    out: dict[str, Any] = {}
    box = (100, 60, 148, 108)
    for name, old_art, old_shape, new_art, new_shape in REPLACEMENT_CASES:
        templates = {
            0: [_template(0, art[old_art], old_art, old_shape)],
            1: [_template(1, art[new_art], new_art, new_shape)],
        }
        replaced, visible, targets = 0, [], set()
        for seed in range(seeds):
            img = np.full((192, 256, 3), (90, 140, 90), np.uint8)
            x1, y1, x2, y2 = box
            alpha = resize_template(art[old_art], x2 - x1, y2 - y1)[..., 3] >= 128
            mask = np.zeros(img.shape[:2], bool)
            mask[y1:y2, x1:x2] = alpha
            img[mask] = MARKER
            aug = ReplacementAugment(
                templates=templates, class_counts=counts, p_replace=1.0, p_add=0.0,
                jitter=PhotometricJitter.none(), match_brightness=0.0,
            )  # fmt: skip
            res = aug(img, np.array([box], float), np.array([0]), np.random.default_rng(seed))
            if [op.kind for op in res.ops] != ["replace"]:
                continue
            replaced += 1
            targets.add(int(res.class_ids[0]))
            still = mask & np.all(res.image == MARKER, axis=2)
            visible.append(float(still.sum() / mask.sum()))
        out[name] = {
            "old": f"{old_art} ({old_shape})",
            "rarer_template": f"{new_art} ({new_shape})",
            "seeds": seeds,
            "replaced": replaced,
            "new_classes": sorted(targets),
            "old_pixels_visible_mean": round(float(np.mean(visible)), 4) if visible else None,
            "old_pixels_visible_max": round(float(np.max(visible)), 4) if visible else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=REPO / "bench/results/augment_probes.json")
    parser.add_argument("--only", choices=("perspective", "replacement"))
    parser.add_argument(
        "--code-label", help="Which code ``tsd`` was imported from (recorded in the output)."
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    doc: dict[str, Any] = {}
    if args.only != "replacement":
        doc["perspective_border_boxes"] = [perspective_probe(s, args.seeds) for s in (0.08, 0.15)]
    if args.only != "perspective":
        doc["replacement"] = replacement_probe(50)
    doc["code_under_test"] = {
        "label": args.code_label,
        "tsd_source": str(Path(tsd.__file__).resolve().parent),
    }
    doc["provenance"] = provenance()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
