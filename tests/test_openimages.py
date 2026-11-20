"""Open Images evaluation-set builder tests (synthetic metadata, mock S3, no network)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from PIL import Image

from tsd.data import openimages as oi
from tsd.data.yolo import read_labels
from tsd.net import Downloader, DownloadPolicy, RateLimiter

MIDS = {"/m/ts": "Traffic sign", "/m/stop": "Stop sign", "/m/tl": "Traffic light", "/m/car": "Car"}
HIERARCHY = {
    "LabelName": "/m/entity",
    "Subcategory": [
        {"LabelName": "/m/ts", "Subcategory": [{"LabelName": "/m/stop"}]},
        {"LabelName": "/m/tl"},
        {"LabelName": "/m/car"},
    ],
}
BBOX_HEADER = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
               "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]  # fmt: skip
SIZES = {
    "img_a": (200, 100),
    "img_b": (120, 160),
    "img_c": (64, 64),
    "img_n": (80, 60),
    "img_p": (50, 50),
}


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    rng = np.random.default_rng(width * height)
    Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _box(
    img: str, mid: str, x1: float, x2: float, y1: float, y2: float, group: int = 0
) -> list[object]:
    return [img, "xclick", mid, 1, x1, x2, y1, y2, 0, 0, group, 0, 0]


@pytest.fixture
def metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "meta"
    d.mkdir()
    with (d / "oidv7-class-descriptions-boxable.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["LabelName", "DisplayName"], *MIDS.items()])
    (d / "bbox_labels_600_hierarchy.json").write_text(json.dumps(HIERARCHY))
    _write_csv(d / "validation-annotations-bbox.csv", BBOX_HEADER, [
        _box("img_a", "/m/stop", 0.1, 0.3, 0.2, 0.6),
        _box("img_a", "/m/tl", 0.5, 0.54, 0.1, 0.3, group=1),  # 0.8% of the image
        _box("img_a", "/m/car", 0.0, 1.0, 0.0, 1.0),  # not a target class
        _box("img_b", "/m/ts", 0.25, 0.5, 0.25, 0.5),
        _box("img_z", "/m/car", 0.1, 0.2, 0.1, 0.2),  # no target box: not selected
    ])  # fmt: skip
    _write_csv(
        d / "test-annotations-bbox.csv", BBOX_HEADER, [_box("img_c", "/m/tl", 0, 0.5, 0, 0.5)]
    )
    _write_csv(d / "validation-annotations-human-imagelabels-boxable.csv",
               ["ImageID", "Source", "LabelName", "Confidence"], [
        ["img_a", "verification", "/m/stop", 1],
        ["img_a", "verification", "/m/tl", 1],
        ["img_b", "verification", "/m/ts", 1],
        ["img_b", "verification", "/m/tl", 0],
        ["img_n", "verification", "/m/tl", 0],  # no boxes, verified negative
        ["img_n", "verification", "/m/car", 1],  # not a target class
        ["img_p", "verification", "/m/stop", 0],
        ["img_p", "verification", "/m/tl", 1],  # claims a light but has no box: never kept
    ])  # fmt: skip
    _write_csv(d / "test-annotations-human-imagelabels-boxable.csv",
               ["ImageID", "Source", "LabelName", "Confidence"], [
        ["img_c", "verification", "/m/ts", 0],  # implies "no stop sign" as well
    ])  # fmt: skip
    meta_header = ["ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
                   "AuthorProfileURL", "Author", "Title", "OriginalSize", "OriginalMD5",
                   "Thumbnail300KURL", "Rotation"]  # fmt: skip
    validation = ["img_a", "img_b", "img_z", "img_n", "img_p"]
    for split, ids in (("validation", validation), ("test", ["img_c"])):
        _write_csv(d / f"{split}-images-with-rotation.csv", meta_header, [
            [i, split, "", f"https://flickr.example/{i}", "https://creativecommons.org/licenses/by/2.0/",
             "", f"author {i}", "", 0, "", "", "0.0"] for i in ids
        ])  # fmt: skip
    for name in list(oi.METADATA):
        p = d / name
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        monkeypatch.setitem(
            oi.METADATA, name, oi.MetadataFile(f"https://meta.test/{name}", p.stat().st_size, md5)
        )
    return d


class FakeS3:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.bodies = {f"/{s}/{i}.jpg": _jpeg(*wh) for i, wh in SIZES.items()
                       for s in ("validation", "test")}  # fmt: skip

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        body = self.bodies.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, headers={"ETag": f'"{hashlib.md5(body).hexdigest()}"'},
                              content=body)  # fmt: skip


def _downloader(s3: FakeS3) -> Downloader:
    client = httpx.Client(transport=httpx.MockTransport(s3))
    return Downloader(DownloadPolicy(retries=0), client=client, limiter=RateLimiter(1e12))


def test_hierarchy_and_label_expansion(metadata: Path) -> None:
    mids = {k: v for k, v in MIDS.items() if v != "Car"}
    h = oi.load_hierarchy(metadata / "bbox_labels_600_hierarchy.json", mids, oi.TARGET_CLASSES)
    assert h.parents == {"Stop sign": ("Traffic sign",)}
    assert h.ancestors("Stop sign") == {"Traffic sign"}
    assert h.descendants("Traffic sign") == {"Stop sign"}
    pos, neg = oi.expand_verified_labels({"Stop sign": True, "Traffic light": False}, h)
    assert pos == {"Stop sign", "Traffic sign"}
    assert neg == {"Traffic light"}
    pos, neg = oi.expand_verified_labels({"Traffic sign": False}, h)
    assert neg == {"Traffic sign", "Stop sign"}
    with pytest.raises(ValueError, match="contradictory"):
        oi.expand_verified_labels({"Stop sign": True, "Traffic sign": False}, h)


def test_build_writes_labels_annotations_and_manifest(metadata: Path, tmp_path: Path) -> None:
    s3 = FakeS3()
    out = tmp_path / "eval"
    config = oi.BuildConfig(
        out=out,
        metadata_dir=tmp_path / "unused",
        reuse_dirs=(metadata,),
        manifest_copy_dir=tmp_path / "manifests",
    )
    summary = oi.build(config, downloader=_downloader(s3))

    assert sorted(s3.requests) == [
        "/test/img_c.jpg",
        "/validation/img_a.jpg",
        "/validation/img_b.jpg",
    ]
    ann = json.loads((out / "annotations.json").read_text())
    assert ann["classes"] == list(oi.TARGET_CLASSES)
    assert "verified negatives" not in ann["info"]["description"]
    assert ann["info"]["class_parents"] == {"Stop sign": ["Traffic sign"]}
    by_id = {r["image_id"]: r for r in ann["images"]}
    a = by_id["img_a"]
    assert (a["width"], a["height"]) == (200, 100)
    assert a["labels"] == [1, 2]  # Stop sign, Traffic light (Car dropped)
    np.testing.assert_allclose(a["boxes"][0], [20, 20, 60, 60])
    assert a["group_of"] == [False, True]
    assert a["verified_positive"] == [0, 1, 2]  # Stop sign implies Traffic sign
    assert by_id["img_b"]["verified_negative"] == [2]
    assert by_id["img_c"]["verified_negative"] == [0, 1]  # no traffic sign -> no stop sign

    cls, boxes = read_labels(out / "labels" / "validation" / "img_a.txt", 200, 100)
    np.testing.assert_array_equal(cls, [1, 2])
    np.testing.assert_allclose(boxes[1], [100, 10, 108, 30], atol=1e-3)

    with (out / "manifest.csv").open() as f:
        rows = {r["image_id"]: r for r in csv.DictReader(f)}
    jpg = out / "images" / "validation" / "img_a.jpg"
    assert rows["img_a"]["sha256"] == hashlib.sha256(jpg.read_bytes()).hexdigest()
    assert rows["img_a"]["bytes"] == str(jpg.stat().st_size)
    assert rows["img_a"]["license"].startswith("https://creativecommons.org/")
    assert rows["img_c"]["source_url"].endswith("/test/img_c.jpg")
    doc = yaml.safe_load((out / "data.yaml").read_text())
    assert doc["val"] == ["images/validation", "images/test"]
    assert doc["names"] == {0: "Traffic sign", 1: "Stop sign", 2: "Traffic light"}
    assert summary["validation"]["classes"]["Traffic light"] == {
        "images": 1, "boxes": 1, "boxes_lt_1pct_area": 1, "group_of": 1}  # fmt: skip

    copied = tmp_path / "manifests" / "manifest.csv"
    assert copied.read_text() == (out / "manifest.csv").read_text()

    # A rebuild finds every image on disk and makes no new image requests.
    s3_again = FakeS3()
    oi.build(config, downloader=_downloader(s3_again))
    assert s3_again.requests == []


def test_build_can_add_verified_negative_images(metadata: Path, tmp_path: Path) -> None:
    s3 = FakeS3()
    out = tmp_path / "eval_neg"
    config = oi.BuildConfig(
        out=out, metadata_dir=tmp_path / "unused", reuse_dirs=(metadata,),
        include_verified_negatives=True,
    )  # fmt: skip
    oi.build(config, downloader=_downloader(s3))
    assert "/validation/img_n.jpg" in s3.requests
    assert "/validation/img_p.jpg" not in s3.requests
    by_id = {r["image_id"]: r for r in json.loads((out / "annotations.json").read_text())["images"]}
    assert sorted(by_id) == ["img_a", "img_b", "img_c", "img_n"]
    assert by_id["img_n"]["boxes"] == []
    assert by_id["img_n"]["verified_negative"] == [2]
    assert by_id["img_n"]["verified_positive"] == []
    description = json.loads((out / "annotations.json").read_text())["info"]["description"]
    assert "verified negatives" in description


def test_ensure_metadata_prefers_verified_local_copies(metadata: Path, tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200, content=(metadata / "oidv7-class-descriptions-boxable.csv").read_bytes()
        )

    dl = Downloader(client=httpx.Client(transport=httpx.MockTransport(handler)),
                    limiter=RateLimiter(1e12))  # fmt: skip
    name = "oidv7-class-descriptions-boxable.csv"
    found = oi.ensure_metadata([name], tmp_path / "dl", [metadata], dl)
    assert found[name] == metadata / name
    assert calls == []
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / name).write_bytes(b"x" * oi.METADATA[name].size)  # right size, wrong content
    found = oi.ensure_metadata([name], tmp_path / "dl", [corrupt], dl)
    assert found[name] == tmp_path / "dl" / name
    assert calls == [f"/{name}"]


def test_unknown_class_is_rejected(metadata: Path) -> None:
    with pytest.raises(ValueError, match="not in Open Images"):
        oi.read_class_mids(metadata / "oidv7-class-descriptions-boxable.csv", ["Unicorn"])
