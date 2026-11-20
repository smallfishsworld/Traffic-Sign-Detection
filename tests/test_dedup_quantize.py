"""Near-duplicate hashing and the INT8 ignored-scope selection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tsd.data.dedup import cross_overlap, dhash, hamming
from tsd.infer.quantize import detect_head_prefix, head_decode_ops


def _photo(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (640, 480), interpolation=cv2.INTER_CUBIC)


def test_dhash_is_robust_to_resize_and_jpeg() -> None:
    img = _photo(0)
    resized = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert ok
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    assert hamming(dhash(img), dhash(recompressed)) <= 4
    assert hamming(dhash(img), dhash(_photo(1))) > 15


def test_cross_overlap_finds_the_duplicate(tmp_path: Path) -> None:
    a, b = [], []
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"a{i}.jpg"), _photo(i))
        a.append(tmp_path / f"a{i}.jpg")
    for i in range(3, 6):
        cv2.imwrite(str(tmp_path / f"b{i}.jpg"), _photo(i))
        b.append(tmp_path / f"b{i}.jpg")
    clean = cross_overlap(a, b)
    assert clean.pairs_compared == 9
    assert clean.duplicates == []
    cv2.imwrite(str(tmp_path / "dup.jpg"), cv2.resize(_photo(1), (300, 225)))
    dirty = cross_overlap(a, [*b, tmp_path / "dup.jpg"])
    assert dirty.closest_pair == ("a1.jpg", "dup.jpg")
    assert [(d.a, d.b) for d in dirty.duplicates] == [("a1.jpg", "dup.jpg")]


def test_smooth_sky_false_candidate_is_rejected(tmp_path: Path) -> None:
    """Two unrelated images that share a smooth gradient (like sky) collide on the 64-bit hash;
    the different aspect ratio rejects the pair."""
    ramp = np.tile(np.linspace(40, 220, 640, dtype=np.float32), (480, 1))
    landscape = np.dstack([ramp] * 3).astype(np.uint8)
    portrait = cv2.resize(landscape, (360, 480))
    portrait[300:420, 100:260] = (0, 200, 255)  # a sign-like blob the coarse hash barely sees
    cv2.imwrite(str(tmp_path / "land.jpg"), landscape)
    cv2.imwrite(str(tmp_path / "port.jpg"), portrait)
    report = cross_overlap([tmp_path / "land.jpg"], [tmp_path / "port.jpg"], threshold=64)
    (pair,) = report.candidates
    assert pair.aspect_log_ratio > 0.2
    assert not pair.duplicate
    assert report.duplicates == []
    with pytest.raises(ValueError, match="non-empty"):
        cross_overlap([], [tmp_path / "port.jpg"])


def test_strong_coarse_match_is_a_duplicate_whatever_the_fine_hash() -> None:
    from tsd.data.dedup import is_duplicate

    assert is_duplicate(3, 41, 0.0009)  # the resize+q70 copy that the fine hash used to veto
    assert not is_duplicate(9, 41, 0.0)  # weak coarse match: the fine hash decides
    assert is_duplicate(9, 30, 0.0)
    assert not is_duplicate(2, 10, 0.3)  # different aspect ratio: never a copy


def test_low_texture_resize_jpeg_copy_is_flagged(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    ramp = np.tile(np.linspace(60, 200, 1024, dtype=np.float32), (768, 1))
    image = np.dstack([ramp, ramp * 0.9, ramp * 0.8]) + rng.normal(0, 2, (768, 1024, 3))
    image[500:540, 300:330] = (30, 30, 200)  # a small sign on a sky-like background
    cv2.imwrite(str(tmp_path / "orig.jpg"), np.clip(image, 0, 255).astype(np.uint8))
    copy = cv2.resize(
        cv2.imread(str(tmp_path / "orig.jpg")), (640, 480), interpolation=cv2.INTER_AREA
    )
    cv2.imwrite(str(tmp_path / "copy.jpg"), copy, [cv2.IMWRITE_JPEG_QUALITY, 70])
    (pair,) = cross_overlap([tmp_path / "orig.jpg"], [tmp_path / "copy.jpg"]).candidates
    assert pair.distance <= 5
    assert pair.duplicate


def test_head_decode_ops_keep_conv_branches_quantized() -> None:
    ops = [
        ("/model.22/cv1/conv/Conv", "Convolution"),
        ("/model.23/cv2.0/cv2.0.0/conv/Conv", "Convolution"),
        ("/model.23/cv3.1/cv3.1.2/Conv", "Convolution"),
        ("/model.23/dfl/Softmax", "Softmax"),
        ("/model.23/dfl/conv/Conv", "Convolution"),
        ("/model.23/Sub", "Subtract"),
        ("/model.23/Constant_3", "Constant"),
        ("/model.23/Sigmoid", "Sigmoid"),
    ]
    assert detect_head_prefix([n for n, _ in ops]) == "/model.23"
    assert head_decode_ops(ops) == [
        "/model.23/dfl/Softmax",
        "/model.23/dfl/conv/Conv",
        "/model.23/Sub",
        "/model.23/Sigmoid",
    ]
    with pytest.raises(ValueError, match="DFL"):
        detect_head_prefix(["/model.1/conv"])
