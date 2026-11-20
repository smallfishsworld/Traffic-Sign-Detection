"""Shared fixtures. Tests never use the network; the only external file is the cached
``models/yolo11n.pt`` (override with ``TSD_TEST_WEIGHTS``), and tests that need it are skipped
with a clear reason when it is absent."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def pytest_configure(config: pytest.Config) -> None:
    # Tracing YOLO for ONNX export warns about data-dependent shapes by design. The filter names
    # a torch class, so it can only be registered where torch is installed (not in CI).
    if has_module("torch"):
        config.addinivalue_line("filterwarnings", "ignore::torch.jit.TracerWarning")


@pytest.fixture(scope="session")
def weights_path() -> Path:
    path = Path(os.environ.get("TSD_TEST_WEIGHTS", REPO / "models" / "yolo11n.pt"))
    if not path.exists():
        pytest.skip(f"{path} not found: run `tsd fetch-assets` to download the yolo11n weights")
    if not has_module("ultralytics"):
        pytest.skip("ultralytics not installed (install the 'torch' extra)")
    return path


@pytest.fixture(scope="session")
def sample_images() -> list[np.ndarray]:
    """Two real photos shipped inside the ultralytics wheel (bus.jpg, zidane.jpg)."""
    if not has_module("ultralytics"):
        pytest.skip("ultralytics not installed (install the 'torch' extra)")
    spec = importlib.util.find_spec("ultralytics")
    assert spec is not None
    assert spec.origin is not None
    from tsd.io import load_image

    assets = Path(spec.origin).parent / "assets"
    return [load_image(assets / "bus.jpg"), load_image(assets / "zidane.jpg")]
