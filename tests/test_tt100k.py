"""TT100K converter tests on a synthetic dataset in the TT100K 2021 layout."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from tsd.data import tt100k
from tsd.data.yolo import read_labels

W, H = 64, 48
TYPES = ["pl15", "pl80", "w57", "i5", "rare"]


def _write_jpeg(path: Path, size: tuple[int, int] = (W, H)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(path.name))
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def _obj(cat: str, xmin: float, ymin: float, xmax: float, ymax: float) -> dict[str, object]:
    return {
        "category": cat,
        "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
        "ellipse_org": [[xmin, ymin], [xmax, ymax]],
    }


def make_tt100k(root: Path, *, imgs_first: bool = False, shuffle_ids: bool = False) -> Path:
    """Synthetic TT100K root: 20 train images, 3 test images, plus edge cases."""
    imgs: dict[str, dict[str, object]] = {}
    train_ids = [str(i) for i in range(1, 21)]
    for i in train_ids:
        imgs[i] = {
            "path": f"train/{i}.jpg",
            "id": int(i),
            "objects": [_obj("pl80", 10, 12, 30, 36)],
        }
    imgs["1"]["objects"] = [
        _obj("pl80", 10, 12, 30, 36),
        _obj("w57", 50, 10, 70, 20),  # extends past the right edge -> clipped to x=64
        _obj("pl15", 40, 30, 20, 44),  # xmin > xmax -> corners sorted
        _obj("i5", 5, 5, 6, 6),  # 1x1 px -> dropped (min_box_size=2)
        _obj("rare", 2, 2, 12, 12),  # single instance -> removed by min_instances=2
    ]
    imgs["2"]["objects"] = [_obj("w57", 0, 0, 8, 8), _obj("pl15", 20, 20, 30, 30)]
    imgs["3"]["objects"] = [_obj("i5", 1, 1, 9, 9), _obj("i5", 20, 1, 29, 9)]
    imgs["4"]["objects"] = []  # background image
    test_ids = ["100", "101", "102"]
    for i in test_ids:
        imgs[i] = {"path": f"test/{i}.jpg", "id": int(i), "objects": [_obj("pl15", 1, 1, 11, 11)]}
    imgs["13"]["path"] = "train/13.jpg"  # annotated, but the image file is missing (see below)

    for i in train_ids:
        if i != "13":
            _write_jpeg(root / "train" / f"{i}.jpg")
    for i in test_ids:
        _write_jpeg(root / "test" / f"{i}.jpg")
    listed_train = [*train_ids, "999"]  # "999" is listed but has no annotation
    if shuffle_ids:
        random.Random(0).shuffle(listed_train)
    (root / "train" / "ids.txt").write_text("\n".join(listed_train) + "\n")
    (root / "test" / "ids.txt").write_text("\n".join(test_ids) + "\n\n")
    doc = {"imgs": imgs, "types": TYPES} if imgs_first else {"types": TYPES, "imgs": imgs}
    (root / "annotations_all.json").write_text(json.dumps(doc))
    return root


@pytest.fixture
def tt100k_root(tmp_path: Path) -> Path:
    return make_tt100k(tmp_path / "tt100k_2021")


def _convert(root: Path, out: Path, **kwargs: object) -> tt100k.ConversionReport:
    params: dict[str, object] = {"val_fraction": 0.25, "workers": 1, "min_instances": 2}
    params.update(kwargs)
    return tt100k.convert(tt100k.ConvertConfig(root=root, out=out, **params))  # type: ignore[arg-type]


def _split_of(out: Path, image_id: str) -> str:
    matches = list(out.glob(f"labels/*/{image_id}.txt"))
    assert len(matches) == 1
    return matches[0].parent.name


def test_read_types_stops_before_imgs_and_handles_key_order(tmp_path: Path) -> None:
    a = make_tt100k(tmp_path / "a")
    b = make_tt100k(tmp_path / "b", imgs_first=True)
    assert tt100k.read_types(a / "annotations_all.json") == TYPES
    assert tt100k.read_types(b / "annotations_all.json") == TYPES


def test_label_values_clipping_and_dropping(tt100k_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "yolo"
    report = _convert(tt100k_root, out)
    assert report.classes == ["pl15", "pl80", "w57", "i5"]  # "rare" has 1 instance < 2
    split = _split_of(out, "1")
    cls, boxes = read_labels(out / "labels" / split / "1.txt", W, H)
    names = [report.classes[c] for c in cls]
    assert names == ["pl80", "w57", "pl15"]  # i5 1x1 dropped, "rare" filtered
    np.testing.assert_allclose(boxes[0], [10, 12, 30, 36], atol=1e-4)
    np.testing.assert_allclose(boxes[1], [50, 10, 64, 20], atol=1e-4)  # clipped at x = W
    np.testing.assert_allclose(boxes[2], [20, 30, 40, 44], atol=1e-4)  # corners sorted
    text = (out / "labels" / split / "1.txt").read_text().splitlines()[0]
    assert text == f"1 {20 / W:.6f} {24 / H:.6f} {20 / W:.6f} {24 / H:.6f}"
    assert report.dropped_boxes == {"filtered_class": 1, "too_small_after_clip": 1}


def test_class_filter_explicit_list_and_unknown(tt100k_root: Path, tmp_path: Path) -> None:
    report = _convert(tt100k_root, tmp_path / "o", classes=("w57", "pl15"))
    assert report.classes == ["pl15", "w57"]  # kept in TT100K "types" order
    with pytest.raises(ValueError, match="unknown"):
        _convert(tt100k_root, tmp_path / "o2", classes=("nope",))


def test_count_split_train_only(tt100k_root: Path, tmp_path: Path) -> None:
    # pl15 has 2 train + 3 test instances: kept at min_instances=3 only if test counts.
    both = _convert(tt100k_root, tmp_path / "a", min_instances=3, count_split="train+test")
    train = _convert(tt100k_root, tmp_path / "b", min_instances=3, count_split="train")
    assert "pl15" in both.classes
    assert "pl15" not in train.classes


def test_split_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    root_a = make_tt100k(tmp_path / "a")
    root_b = make_tt100k(tmp_path / "b", shuffle_ids=True)
    _convert(root_a, tmp_path / "oa")
    _convert(root_b, tmp_path / "ob")
    ids = [str(i) for i in range(1, 21) if i != 13] + ["100", "101", "102"]
    splits_a = {i: _split_of(tmp_path / "oa", i) for i in ids}
    splits_b = {i: _split_of(tmp_path / "ob", i) for i in ids}
    assert splits_a == splits_b
    assert {splits_a[i] for i in ["100", "101", "102"]} == {"test"}
    assert {"train", "val"} <= set(splits_a.values())


def test_assign_split_fraction_is_close_to_requested() -> None:
    ids = [str(i) for i in range(20000)]
    val = sum(tt100k.assign_split(i, "train", 0.1, "salt") == "val" for i in ids)
    assert 0.09 < val / len(ids) < 0.11


def test_report_skips_and_background_images(tt100k_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    report = _convert(tt100k_root, out)
    assert report.skipped_images == {
        "listed_but_not_annotated": 1,
        "missing_or_unreadable_image": 1,
    }
    assert sum(report.images.values()) == 22  # 20 train - 1 missing + 3 test
    assert (out / "labels" / _split_of(out, "4") / "4.txt").read_text() == ""
    saved = json.loads((out / "conversion_report.json").read_text())
    assert saved["classes"] == report.classes
    no_empty = _convert(tt100k_root, tmp_path / "o2", keep_empty=False)
    assert no_empty.skipped_images["no_kept_boxes"] == 1


def test_data_yaml_and_image_links(tt100k_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    _convert(tt100k_root, out)
    doc = yaml.safe_load((out / "data.yaml").read_text())
    assert doc["names"] == {0: "pl15", 1: "pl80", 2: "w57", 3: "i5"}
    assert doc["test"] == "images/test"
    assert Path(doc["path"]) == out.resolve()
    link = out / "images" / "test" / "100.jpg"
    assert link.is_symlink()
    assert link.resolve() == (tt100k_root / "test" / "100.jpg").resolve()
    copied = tmp_path / "copied"
    _convert(tt100k_root, copied, link_mode="copy")
    assert not (copied / "images" / "test" / "100.jpg").is_symlink()


def test_process_pool_matches_sequential(tt100k_root: Path, tmp_path: Path) -> None:
    _convert(tt100k_root, tmp_path / "seq", workers=1)
    _convert(tt100k_root, tmp_path / "par", workers=2)
    seq = {
        p.relative_to(tmp_path / "seq"): p.read_text()
        for p in (tmp_path / "seq").glob("labels/*/*.txt")
    }
    par = {
        p.relative_to(tmp_path / "par"): p.read_text()
        for p in (tmp_path / "par").glob("labels/*/*.txt")
    }
    assert seq == par
    assert len(seq) == 22


def test_rerun_needs_overwrite_and_leaves_no_stale_files(tt100k_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    _convert(tt100k_root, out, val_fraction=0.1)
    with pytest.raises(FileExistsError, match="overwrite"):
        _convert(tt100k_root, out, val_fraction=0.5)
    (out / "labels" / "train.cache").write_text("stale Ultralytics cache")
    report = _convert(tt100k_root, out, val_fraction=0.5, overwrite=True)
    stems: dict[str, list[str]] = {}
    for label in (out / "labels").rglob("*.txt"):
        stems.setdefault(label.stem, []).append(label.parent.name)
    assert all(len(splits) == 1 for splits in stems.values())  # every image in one split
    assert len(stems) == sum(report.images.values())
    assert sum(1 for p in (out / "images").rglob("*") if not p.is_dir()) == len(stems)
    assert not (out / "labels" / "train.cache").exists()


def test_val_fraction_must_leave_a_val_split(tt100k_root: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="val_fraction"):
        _convert(tt100k_root, tmp_path / "o", val_fraction=0.0)
    # a valid fraction that happens to put no image into val (a small input) is refused too,
    # before a data.yaml without a val split is written
    with pytest.raises(ValueError, match="no image fell into the val split"):
        _convert(tt100k_root, tmp_path / "o2", val_fraction=1e-12)
    assert not (tmp_path / "o2" / "data.yaml").exists()
