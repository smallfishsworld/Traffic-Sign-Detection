"""Photometric jitter for pasted sign templates.

The 2024 replacement transform only changed a template's brightness uniformly; the report lists
"simulate different illumination on different parts of the standard sign" as future work. This
module adds that: a smooth gain field (linear ramp in a random direction plus an optional soft
highlight) multiplied into the template, on top of brightness/contrast/saturation/hue jitter,
blur and sensor noise.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]


@dataclass(frozen=True)
class PhotometricJitter:
    """Ranges of the random template perturbations. All-zero values disable jitter."""

    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.3
    hue_deg: float = 6.0
    gradient_strength: float = 0.6
    p_gradient: float = 0.8
    highlight_strength: float = 0.25
    blur_sigma_max: float = 0.8
    noise_std: float = 3.0

    @classmethod
    def none(cls) -> PhotometricJitter:
        return cls(**{f.name: 0.0 for f in dataclasses.fields(cls)})


def illumination_gain(
    height: int,
    width: int,
    strength: float,
    angle: float,
    highlight: float = 0.0,
    highlight_center: tuple[float, float] = (0.5, 0.5),
) -> F32:
    """Smooth multiplicative gain map of shape ``(height, width)``.

    The linear part goes from ``1 - strength/2`` to ``1 + strength/2`` along direction ``angle``
    (radians) across the template, so its mean is 1 and it only redistributes brightness. The
    optional highlight adds a Gaussian bump of peak ``highlight`` at ``highlight_center``
    (fractions of width/height), as from a street light or specular reflection.
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    u = (xs + 0.5) / max(width, 1) - 0.5
    v = (ys + 0.5) / max(height, 1) - 0.5
    proj = u * np.cos(angle) + v * np.sin(angle)
    span = float(proj.max() - proj.min())
    ramp = (proj - proj.mean()) / span if span > 0 else np.zeros_like(proj)
    gain = 1.0 + strength * ramp
    if highlight > 0:
        cx, cy = highlight_center
        d2 = (u + 0.5 - cx) ** 2 + (v + 0.5 - cy) ** 2
        gain = gain + highlight * np.exp(-d2 / (2 * 0.2**2))
    return np.asarray(gain, dtype=np.float32)


def jitter_bgr(bgr: F32, jitter: PhotometricJitter, rng: np.random.Generator) -> F32:
    """Perturb a float32 image in ``[0, 255]`` (3 channels, BGR order) and return a new array."""
    out = bgr.astype(np.float32, copy=True)
    if jitter.saturation > 0 or jitter.hue_deg > 0:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv_f = hsv.astype(np.float32)
        # OpenCV stores 8-bit hue as degrees / 2 in [0, 180).
        hsv_f[..., 0] = (hsv_f[..., 0] + rng.uniform(-jitter.hue_deg, jitter.hue_deg) / 2) % 180
        hsv_f[..., 1] *= 1.0 + rng.uniform(-jitter.saturation, jitter.saturation)
        hsv_u8 = np.clip(hsv_f, 0, [179, 255, 255]).astype(np.uint8)
        out = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR).astype(np.float32)
    if jitter.contrast > 0:
        mean = out.mean()
        out = (out - mean) * (1.0 + rng.uniform(-jitter.contrast, jitter.contrast)) + mean
    if jitter.brightness > 0:
        out *= 1.0 + rng.uniform(-jitter.brightness, jitter.brightness)
    if jitter.gradient_strength > 0 and rng.random() < jitter.p_gradient:
        h, w = out.shape[:2]
        gain = illumination_gain(
            h,
            w,
            strength=rng.uniform(0.0, jitter.gradient_strength),
            angle=rng.uniform(0.0, 2 * np.pi),
            highlight=rng.uniform(0.0, jitter.highlight_strength),
            highlight_center=(float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))),
        )
        out *= gain[..., None]
    if jitter.blur_sigma_max > 0:
        sigma = rng.uniform(0.0, jitter.blur_sigma_max)
        if sigma > 0.1:
            out = np.asarray(cv2.GaussianBlur(out, (0, 0), sigmaX=sigma), dtype=np.float32)
    if jitter.noise_std > 0:
        out += rng.normal(0.0, jitter.noise_std, size=out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.float32)
