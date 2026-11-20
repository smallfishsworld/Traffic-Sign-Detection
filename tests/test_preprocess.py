"""Letterbox pre-processing: geometry, inverse mapping and parity with Ultralytics."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import has_module
from tsd.infer.preprocess import LetterboxInfo, letterbox, preprocess, to_input_tensor


def _image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_landscape_geometry() -> None:
    out, info = letterbox(_image(768, 1024))
    assert out.shape == (640, 640, 3)
    assert info == LetterboxInfo(0.625, 0, 80, 1024, 768, 0.625, 0.625)
    assert (out[:80] == 114).all()
    assert (out[-80:] == 114).all()


def test_portrait_odd_padding_and_identity() -> None:
    out, info = letterbox(_image(1000, 480))
    assert out.shape == (640, 640, 3)
    assert (info.pad_x, info.pad_y) == (
        166,
        0,
    )  # 307 px wide image, 333 px of padding split 166/167
    same = _image(640, 640)
    out2, info2 = letterbox(same)
    np.testing.assert_array_equal(out2, same)
    assert info2 == LetterboxInfo(1.0, 0, 0, 640, 640, 1.0, 1.0)


@pytest.mark.parametrize(("h", "w"), [(768, 1024), (1000, 480), (333, 777), (3888, 8100)])
def test_inverse_mapping_round_trip(h: int, w: int) -> None:
    _, info = letterbox(np.zeros((h, w, 3), np.uint8))
    rng = np.random.default_rng(h + w)
    xy = rng.uniform(0, [w * 0.8, h * 0.8], size=(20, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(1, [w * 0.2, h * 0.2], size=(20, 2))], axis=1)
    gains = [info.gain_x, info.gain_y, info.gain_x, info.gain_y]
    forward = boxes * gains + [info.pad_x, info.pad_y, info.pad_x, info.pad_y]
    np.testing.assert_allclose(info.to_original(forward), boxes, atol=1e-9)
    clipped = info.to_original(np.array([[-50.0, -50.0, 700.0, 700.0]]))
    np.testing.assert_allclose(clipped, [[0, 0, w, h]])


def test_tensor_layout_and_range() -> None:
    blue = np.zeros((640, 640, 3), np.uint8)
    blue[..., 0] = 255  # BGR blue
    t = to_input_tensor([blue, blue])
    assert t.shape == (2, 3, 640, 640)
    assert t.dtype == np.float32
    assert t.flags["C_CONTIGUOUS"]
    assert t[:, 2].min() == 1.0  # RGB order: blue is the last channel
    assert t[:, 0].max() == 0.0
    batch, infos = preprocess([_image(768, 1024), _image(500, 500)])
    assert batch.shape == (2, 3, 640, 640)
    assert [i.orig_width for i in infos] == [1024, 500]


@pytest.mark.skipif(not has_module("ultralytics"), reason="ultralytics not installed")
@pytest.mark.parametrize(("h", "w"), [(768, 1024), (1000, 480), (333, 777), (640, 640)])
def test_matches_ultralytics_letterbox(h: int, w: int) -> None:
    from tsd.infer.backends import import_ultralytics

    import_ultralytics()
    from ultralytics.data.augment import LetterBox

    img = _image(h, w, seed=3)
    ours, _ = letterbox(img)
    theirs = LetterBox((640, 640), auto=False, stride=32)(image=img)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize(("h", "w"), [(1, 4000), (2000, 1), (1, 1281)])
def test_thin_images_keep_one_pixel(h: int, w: int) -> None:
    out, info = letterbox(np.zeros((h, w, 3), np.uint8))
    assert out.shape == (640, 640, 3)
    assert min(round(info.gain_x * w), round(info.gain_y * h)) == 1


@pytest.mark.parametrize(("h", "w"), [(1911, 2552), (2086, 3944), (8100, 3888), (683, 1024)])
def test_inverse_mapping_matches_ultralytics_scale_boxes(h: int, w: int) -> None:
    ops = pytest.importorskip("ultralytics.utils.ops")
    _, info = letterbox(np.zeros((h, w, 3), np.uint8))
    boxes = np.array([[0.0, 0.0, 640.0, 640.0], [100.0, 200.0, 300.0, 400.0]], np.float32)
    mine = info.to_original(boxes)
    torch = pytest.importorskip("torch")
    ref = ops.scale_boxes((640, 640), torch.from_numpy(boxes.copy()), (h, w)).numpy()
    np.testing.assert_allclose(mine, ref, atol=1e-3)
