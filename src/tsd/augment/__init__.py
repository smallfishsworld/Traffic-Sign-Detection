"""Training-time augmentations from the 2024 report: perspective and replacement transforms."""

from tsd.augment.perspective import BoxFilter, RandomPerspective, apply_homography, warp_boxes
from tsd.augment.photometric import PhotometricJitter, illumination_gain
from tsd.augment.replacement import ReplacementAugment, class_sampling_weights
from tsd.augment.templates import SignTemplate, load_templates, synthetic_templates

__all__ = [
    "BoxFilter",
    "PhotometricJitter",
    "RandomPerspective",
    "ReplacementAugment",
    "SignTemplate",
    "apply_homography",
    "class_sampling_weights",
    "illumination_gain",
    "load_templates",
    "synthetic_templates",
    "warp_boxes",
]
