"""Image decoding shared by dataset building, evaluation, benchmarks and the HTTP service.

Every component decodes images the same way (OpenCV, 8-bit BGR, EXIF orientation applied), so a
box measured on a decoded image in one place is valid everywhere else. Decoding from a file and
from bytes (the HTTP upload path) give identical arrays.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


class ImageDecodeError(ValueError):
    """The bytes are not a decodable image."""


def decode_image(data: bytes) -> Image:
    """Decode encoded image bytes (JPEG/PNG/...) into an ``HxWx3`` BGR ``uint8`` array."""
    if not data:
        raise ImageDecodeError("empty image payload")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("could not decode image")
    return np.ascontiguousarray(img, dtype=np.uint8)


def load_image(path: Path) -> Image:
    """Read an image file; same result as ``decode_image(path.read_bytes())``."""
    return decode_image(path.read_bytes())
