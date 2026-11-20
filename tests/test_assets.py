"""Pinned assets: safe archive extraction and checksum enforcement."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import httpx
import pytest

from tsd import assets
from tsd.net import Downloader, DownloadError, RateLimiter


def test_extract_jpegs_flattens_and_ignores_unsafe_members(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("coco128/images/train2017/000001.jpg", b"one")
        zf.writestr("../escape.jpg", b"two")
        zf.writestr("/abs/three.JPG", b"three")
        zf.writestr("coco128/labels/000001.txt", b"0 0.5 0.5 0.1 0.1")
        zf.writestr("coco128/images/", b"")
    out = tmp_path / "out"
    written = assets.extract_jpegs(archive, out)
    assert [p.name for p in written] == ["000001.jpg", "escape.jpg", "three.JPG"]
    assert all(p.parent == out for p in written)
    assert not (tmp_path / "escape.jpg").exists()
    assert (out / "000001.jpg").read_bytes() == b"one"


def test_fetch_asset_enforces_pinned_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"weights"
    good = assets.Asset("https://example.test/w.pt", len(body), hashlib.sha256(body).hexdigest())
    monkeypatch.setitem(assets.ASSETS, "w.pt", good)
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    )
    with Downloader(client=client, limiter=RateLimiter(1e12)) as dl:
        assert assets.fetch_asset("w.pt", tmp_path, dl).read_bytes() == body
        bad = assets.Asset(good.url, good.size, "0" * 64)
        monkeypatch.setitem(assets.ASSETS, "bad.pt", bad)
        with pytest.raises(DownloadError, match="sha256"):
            assets.fetch_asset("bad.pt", tmp_path, dl)
    assert not (tmp_path / "bad.pt").exists()


def test_pinned_assets_are_https_github_releases() -> None:
    for name, asset in assets.ASSETS.items():
        assert asset.url.startswith("https://github.com/ultralytics/assets/releases/download/")
        assert asset.url.endswith(name)
        assert len(asset.sha256) == 64
