"""Pinned downloadable assets: COCO-pretrained YOLO11 weights and the COCO128 calibration images.

Each asset is fetched at most once through :class:`tsd.net.Downloader` (polite defaults) and
verified by size and SHA-256. COCO128 is only used as the INT8 calibration set: its images come
from COCO train2017, so they cannot overlap the Open Images evaluation images.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tsd.net import Downloader, DownloadError, DownloadItem

_RELEASES = "https://github.com/ultralytics/assets/releases/download"


@dataclass(frozen=True)
class Asset:
    url: str
    size: int
    sha256: str


ASSETS: dict[str, Asset] = {
    "yolo11n.pt": Asset(
        f"{_RELEASES}/v8.3.0/yolo11n.pt",
        5613764,
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    ),
    "yolo11s.pt": Asset(
        f"{_RELEASES}/v8.3.0/yolo11s.pt",
        19313732,
        "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    ),
    "coco128.zip": Asset(
        f"{_RELEASES}/v0.0.0/coco128.zip",
        6983030,
        "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e",
    ),
}


def fetch_asset(name: str, dest_dir: Path, downloader: Downloader) -> Path:
    """Download (or find) ``name`` in ``dest_dir`` and verify its size and SHA-256."""
    asset = ASSETS[name]
    result = downloader.fetch(DownloadItem(asset.url, dest_dir / name, expected_size=asset.size))
    if result.sha256 != asset.sha256:
        result.dest.unlink()
        raise DownloadError(f"{name}: sha256 {result.sha256} != pinned {asset.sha256}")
    return result.dest


def extract_jpegs(archive: Path, out_dir: Path) -> list[Path]:
    """Extract the ``.jpg`` members of a zip into ``out_dir`` (flattened to their file names).

    Archive members are untrusted: only regular files with a ``.jpg`` suffix are written, under
    their base name, so a member path cannot escape ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename).name
            if info.is_dir() or not name.lower().endswith(".jpg") or name in {"", ".", ".."}:
                continue
            dest = out_dir / name
            if not dest.exists() or dest.stat().st_size != info.file_size:
                dest.write_bytes(zf.read(info))
            written.append(dest)
    return sorted(written)
