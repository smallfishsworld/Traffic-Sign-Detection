"""Image decoding: file and bytes paths agree, EXIF orientation is applied consistently."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tsd.io import ImageDecodeError, decode_image, load_image


def _jpeg_with_orientation(orientation: int) -> bytes:
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)  # 80 wide, 40 tall
    rgb[:, :10] = 255  # bright stripe on the left edge
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", exif=exif, quality=95)
    return buf.getvalue()


def test_file_and_bytes_decode_identically(tmp_path: Path) -> None:
    data = _jpeg_with_orientation(1)
    path = tmp_path / "a.jpg"
    path.write_bytes(data)
    a, b = load_image(path), decode_image(data)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (40, 80, 3)
    assert a.dtype == np.uint8


def test_exif_rotation_is_applied() -> None:
    upright = decode_image(_jpeg_with_orientation(1))
    rotated = decode_image(_jpeg_with_orientation(6))  # "rotate 90 CW to display"
    assert rotated.shape == (80, 40, 3)
    assert upright[:, :10].mean() > 200
    assert rotated[:10, :].mean() > 200  # the left stripe is now at the top


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_invalid_payload(payload: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(payload)
