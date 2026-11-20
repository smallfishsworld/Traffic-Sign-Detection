"""The :class:`Detector`: one interface over every backend, plus model loading.

Exported models carry a JSON sidecar (``<model file>.json``) with the class names, input size,
precision and provenance (source weights and their SHA-256), so ONNX and OpenVINO models can be
served without PyTorch or Ultralytics installed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from tsd.infer.backends import Backend, OnnxRuntimeBackend, OpenVINOBackend, TorchBackend
from tsd.infer.postprocess import NMSConfig, select_batch
from tsd.infer.preprocess import Image, LetterboxInfo, letterbox, to_input_tensor
from tsd.infer.types import Detections

BackendName = Literal["pytorch", "onnxruntime", "openvino"]


@dataclass(frozen=True)
class ModelMeta:
    names: list[str]
    input_size: tuple[int, int] = (640, 640)
    precision: str = "fp32"
    source: str = ""
    source_sha256: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def sidecar(model_path: Path) -> Path:
        return model_path.with_name(model_path.name + ".json")

    def save(self, model_path: Path) -> Path:
        path = self.sidecar(model_path)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, model_path: Path) -> ModelMeta:
        doc = json.loads(cls.sidecar(model_path).read_text())
        doc["input_size"] = tuple(doc["input_size"])
        return cls(**doc)


@dataclass(frozen=True)
class StageTimes:
    """Wall-clock milliseconds per stage of one :meth:`Detector.predict_timed` call."""

    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.infer_ms + self.postprocess_ms


class Detector:
    """Letterbox -> backend -> NMS -> boxes in original-image coordinates."""

    def __init__(
        self,
        backend: Backend,
        names: Sequence[str],
        input_size: tuple[int, int] = (640, 640),
        nms: NMSConfig | None = None,
    ) -> None:
        self.backend = backend
        self.names = list(names)
        self.input_size = input_size
        self.nms = nms or NMSConfig()

    @property
    def name(self) -> str:
        return self.backend.name

    def runtime_settings(self) -> dict[str, Any]:
        """Effective runtime settings reported by the backend (empty for test fakes)."""
        settings = getattr(self.backend, "settings", None)
        return dict(settings()) if callable(settings) else {}

    def prepare(self, image: Image) -> tuple[Image, LetterboxInfo]:
        """Letterbox one image (cheap; can run on request threads before batching)."""
        return letterbox(image, self.input_size)

    def infer_prepared(
        self, prepared: Sequence[tuple[Image, LetterboxInfo]], nms: NMSConfig | None = None
    ) -> list[Detections]:
        raw = self.backend.infer(to_input_tensor([p for p, _ in prepared]))
        return self._to_detections(raw, [info for _, info in prepared], nms)

    def predict(self, images: Sequence[Image], nms: NMSConfig | None = None) -> list[Detections]:
        return self.infer_prepared([self.prepare(im) for im in images], nms)

    def predict_timed(
        self, images: Sequence[Image], nms: NMSConfig | None = None
    ) -> tuple[list[Detections], StageTimes]:
        t0 = time.perf_counter()
        prepared = [self.prepare(im) for im in images]
        batch = to_input_tensor([p for p, _ in prepared])
        t1 = time.perf_counter()
        raw = self.backend.infer(batch)
        t2 = time.perf_counter()
        dets = self._to_detections(raw, [info for _, info in prepared], nms)
        t3 = time.perf_counter()
        return dets, StageTimes((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)

    def _to_detections(
        self, raw: Any, infos: Sequence[LetterboxInfo], nms: NMSConfig | None
    ) -> list[Detections]:
        out = []
        for (boxes, scores, class_ids), info in zip(
            select_batch(raw, nms or self.nms), infos, strict=True
        ):
            out.append(Detections(info.to_original(boxes).astype("float32"), scores, class_ids))
        return out


def backend_for(path: Path) -> BackendName:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return "pytorch"
    if suffix == ".onnx":
        return "onnxruntime"
    if suffix == ".xml":
        return "openvino"
    raise ValueError(f"cannot infer a backend from {path.name} (expected .pt, .onnx or .xml)")


def openvino_label(
    ir_precision: str, hint: Literal["f32", "bf16"], cpu_pinning: bool | None
) -> str:
    """Backend label from the IR's own precision and the precision hint: ``fp32`` and ``int8``
    with the f32 hint, ``bf16`` for the FP32 IR with the bf16 hint, ``int8-bf16`` for the INT8 IR
    with the bf16 hint (its non-quantized ops in BF16), plus ``-nopin`` without thread pinning."""
    label = ir_precision
    if hint == "bf16":
        label = "bf16" if ir_precision == "fp32" else f"{ir_precision}-bf16"
    return f"{label}-nopin" if cpu_pinning is False else label


def load_detector(
    model: Path,
    threads: int | None = None,
    nms: NMSConfig | None = None,
    openvino_precision: Literal["f32", "bf16"] = "f32",
    openvino_cpu_pinning: bool | None = None,
) -> Detector:
    """Load a detector for a ``.pt`` (PyTorch), ``.onnx`` (ONNX Runtime) or ``.xml`` (OpenVINO)
    model. ``threads`` fixes the runtime's intra-op thread count; ``openvino_cpu_pinning`` set to
    ``False`` turns off OpenVINO's thread pinning (``None`` keeps its default, pinned)."""
    kind = backend_for(model)
    if kind == "pytorch":
        torch_backend = TorchBackend(model, threads)
        return Detector(torch_backend, torch_backend.names, nms=nms)
    meta = ModelMeta.load(model)
    backend: Backend
    if kind == "onnxruntime":
        backend = OnnxRuntimeBackend(model, threads)
    else:
        label = openvino_label(meta.precision, openvino_precision, openvino_cpu_pinning)
        backend = OpenVINOBackend(
            model, threads, openvino_precision, label, cpu_pinning=openvino_cpu_pinning
        )
    return Detector(backend, meta.names, meta.input_size, nms)
