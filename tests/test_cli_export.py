"""`tsd export` installs exports only if they pass the parity gate; `tsd parity` re-checks
exports in place. The exporters and runtimes are replaced by fakes (no PyTorch needed)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from tsd.cli import app

runner = CliRunner()


class FakeRuntime:
    """One confident box; ``shift`` moves it by that many pixels (a broken export)."""

    def __init__(self, name: str, shift: float) -> None:
        self.name = name
        self.shift = shift

    def infer(self, batch: np.ndarray) -> np.ndarray:
        raw = np.zeros((batch.shape[0], 4 + 3, 10), np.float32)
        raw[:, :4, 0] = [320 + self.shift, 320, 64, 32]
        raw[:, 4 + 1, 0] = 0.9
        return raw


def fake_export_onnx(weights: Path, out: Path, imgsz: int = 640, opset: int = 18) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"onnx")
    out.with_name(out.name + ".json").write_text("{}")
    return out


def fake_export_openvino(onnx_model: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / f"{onnx_model.stem}.xml"
    xml.write_text("<net/>")
    xml.with_suffix(".bin").write_bytes(b"\0")
    xml.with_name(xml.name + ".json").write_text("{}")
    return xml


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, float]:
    """Patch the exporters and runtimes; set ``shift`` to break both exports."""
    import tsd.infer.backends as backends
    import tsd.infer.export as export

    knobs = {"shift": 0.0}
    monkeypatch.setattr(export, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(export, "export_openvino", fake_export_openvino)
    monkeypatch.setattr(backends, "TorchBackend", lambda *a: FakeRuntime("pytorch", 0.0))
    for name in ("OnnxRuntimeBackend", "OpenVINOBackend"):
        monkeypatch.setattr(backends, name, lambda *a, n=name: FakeRuntime(n, knobs["shift"]))
    return knobs


@pytest.fixture
def parity_images(tmp_path: Path) -> Path:
    d = tmp_path / "parity"
    d.mkdir()
    for i in range(2):
        cv2.imwrite(str(d / f"{i}.jpg"), np.full((120, 160, 3), 60 + i, np.uint8))
    return d


def _export(tmp_path: Path, images: Path, out: Path) -> tuple[int, dict[str, object]]:
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"weights")
    report = tmp_path / "parity.json"
    result = runner.invoke(
        app,
        ["export", str(weights), "--out-dir", str(out), "--parity-images", str(images),
         "--report", str(report)],
    )  # fmt: skip
    return result.exit_code, json.loads(report.read_text())


def test_failed_parity_installs_nothing(
    fakes: dict[str, float], parity_images: Path, tmp_path: Path
) -> None:
    fakes["shift"] = 5.0  # 5 px, beyond the 0.5 px tolerance
    out = tmp_path / "models"
    code, report = _export(tmp_path, parity_images, out)
    assert code == 1
    assert report["passed"] is False
    assert report["installed"] is False
    assert list(out.iterdir()) == []  # no ONNX, no IR, no staging directory left behind


def test_passing_exports_are_moved_into_place(
    fakes: dict[str, float], parity_images: Path, tmp_path: Path
) -> None:
    out = tmp_path / "models"
    (out / "yolo11n_openvino_fp32").mkdir(parents=True)
    (out / "yolo11n_openvino_fp32" / "stale.txt").write_text("from an earlier export")
    code, report = _export(tmp_path, parity_images, out)
    assert code == 0
    assert report["passed"] is True
    assert sorted(p.name for p in out.iterdir()) == [
        "yolo11n.onnx",
        "yolo11n.onnx.json",
        "yolo11n_openvino_fp32",
    ]
    assert sorted(p.name for p in (out / "yolo11n_openvino_fp32").iterdir()) == [
        "yolo11n.bin",
        "yolo11n.xml",
        "yolo11n.xml.json",
    ]
    assert set(report["files_sha256"]) == {  # type: ignore[arg-type]
        "yolo11n.pt", "yolo11n.onnx", "yolo11n.xml", "yolo11n.bin",
    }  # fmt: skip


def test_parity_checks_exports_in_place(
    fakes: dict[str, float], parity_images: Path, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    weights = models / "yolo11n.pt"
    models.mkdir()
    weights.write_bytes(b"weights")
    fake_export_openvino(
        fake_export_onnx(weights, models / "yolo11n.onnx"), models / "yolo11n_openvino_fp32"
    )
    report = tmp_path / "check.json"
    args = ["parity", str(weights), "--parity-images", str(parity_images), "--report", str(report)]
    assert runner.invoke(app, args).exit_code == 0
    doc = json.loads(report.read_text())
    assert doc["passed"] is True
    assert doc["checked"]["openvino"] == str(models / "yolo11n_openvino_fp32" / "yolo11n.xml")
    assert doc["images"] == ["0.jpg", "1.jpg"]
    fakes["shift"] = 5.0
    assert runner.invoke(app, args).exit_code == 1
    assert json.loads(report.read_text())["passed"] is False
    missing = runner.invoke(app, ["parity", str(models / "yolo11s.pt")])
    assert missing.exit_code == 2  # a usage error: nothing to check
