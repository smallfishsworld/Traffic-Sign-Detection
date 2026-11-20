"""The command-line interface: command list, model-argument suffixes, serve overrides."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from tsd.cli import _parse_model_arg, app

runner = CliRunner()


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    commands = ("tt100k", "openimages", "fetch-assets", "export", "parity", "quantize", "eval")
    for command in (*commands, "serve"):
        assert command in result.output


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("models/yolo11n.pt", (Path("models/yolo11n.pt"), "f32", 0)),
        ("m/yolo11n.xml@bf16", (Path("m/yolo11n.xml"), "bf16", 0)),
        ("m/yolo11n.xml@tiles=512", (Path("m/yolo11n.xml"), "f32", 512)),
        ("m/yolo11n.xml@bf16@tiles=640", (Path("m/yolo11n.xml"), "bf16", 640)),
    ],
)
def test_model_argument_suffixes(arg: str, expected: tuple[Path, str, int]) -> None:
    assert _parse_model_arg(arg) == expected


@pytest.mark.parametrize("arg", ["m.xml@fp16", "m.xml@tiles=", "m.xml@tiles=big"])
def test_unknown_model_suffixes_are_refused(arg: str) -> None:
    with pytest.raises(typer.BadParameter, match="unknown model suffix"):
        _parse_model_arg(arg)


def test_serve_options_override_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))
    monkeypatch.setenv("TSD_PORT", "8100")
    monkeypatch.setenv("TSD_MAX_BATCH_SIZE", "4")
    result = runner.invoke(app, ["serve", "--max-batch-size", "1", "--model", "x.xml"])
    assert result.exit_code == 0, result.output
    (call,) = calls
    assert (call["host"], call["port"]) == ("127.0.0.1", 8100)  # loopback by default
    settings = call["app"].state.settings
    assert settings.max_batch_size == 1  # the option wins over TSD_MAX_BATCH_SIZE
    assert settings.model_path == Path("x.xml")


def test_quantize_refuses_a_missing_eval_set(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tsd.cli import app

    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "a.jpg").write_bytes(b"x")
    result = CliRunner().invoke(
        app,
        ["quantize", str(tmp_path / "m.xml"), "--calibration-dir", str(calib),
         "--eval-images", str(tmp_path / "missing")],
    )  # fmt: skip
    assert result.exit_code != 0
    # Rich colours and wraps the error box when it thinks it is in a terminal (it does on
    # GitHub Actions), so check the plain text with the colour codes and box removed.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output).replace("\u2502", " ")
    assert "skip-eval-check" in " ".join(plain.split())


def test_tt100k_val_fraction_bounds(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tsd.cli import app

    result = CliRunner().invoke(
        app, ["tt100k", str(tmp_path), str(tmp_path / "o"), "--val-fraction", "0"]
    )
    assert result.exit_code != 0


def test_eval_tile_merge_iou_bounds(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tsd.cli import app

    result = CliRunner().invoke(
        app, ["eval", "m.xml@tiles=512", "--data-dir", str(tmp_path), "--tile-merge-iou", "1.5"]
    )
    assert result.exit_code == 2  # usage error, before any data is read
