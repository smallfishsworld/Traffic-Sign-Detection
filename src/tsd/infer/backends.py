"""Inference runtimes behind one small interface.

A backend only runs the network: it maps a pre-processed ``float32`` NCHW batch to the raw head
output ``(batch, 4 + num_classes, num_anchors)``. Pre- and post-processing live in
:class:`tsd.infer.detector.Detector`, so the runtimes are compared on identical inputs and
identical NMS. Thread counts are set explicitly; ``None`` keeps the runtime's default.

* ``pytorch``: the fused Ultralytics ``DetectionModel`` in eager mode, FP32 (the same network
  Ultralytics' ``predict`` runs, and how the 2024 project ran the model). Every runtime, this one
  included, gets the same fixed 640x640 letterbox (``auto=False``); Ultralytics' own ``predict``
  pads ``.pt`` inputs only to the smallest stride-32 rectangle (about 73% of the pixels on the
  benchmark images), so it would be somewhat faster than this baseline.
* ``onnxruntime``: CPU execution provider, all graph optimisations, sequential execution,
  ``intra_op_num_threads`` = threads, one inter-op thread.
* ``openvino``: CPU plugin, latency hint, one stream, ``INFERENCE_NUM_THREADS`` = threads and an
  explicit ``INFERENCE_PRECISION_HINT``. The hint matters on CPUs with AMX/AVX512-BF16: there
  OpenVINO runs the non-quantized part of any model in BF16 unless ``f32`` is requested. The CPU
  plugin pins its threads to cores by default (``ENABLE_CPU_PINNING``); PyTorch and ONNX Runtime
  threads are not pinned (the benchmarks record this, see ``settings()``).

Each backend's ``settings()`` reports the effective runtime settings (queried from the runtime,
not the requested values). Benchmark and profile results record them in their provenance,
``tsd serve`` reports them on ``/healthz`` (and the serving benchmark stores that), and new
evaluation detections keep them in ``extra.runtime_settings``.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

BatchArray = NDArray[np.float32]
Precision = Literal["f32", "bf16"]
OPENVINO_PROPERTIES = (
    "INFERENCE_NUM_THREADS",
    "NUM_STREAMS",
    "INFERENCE_PRECISION_HINT",
    "PERFORMANCE_HINT",
    "ENABLE_CPU_PINNING",
    "ENABLE_HYPER_THREADING",
    "SCHEDULING_CORE_TYPE",
    "EXECUTION_MODE_HINT",
    "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
)


class Backend(Protocol):
    name: str

    def infer(self, batch: BatchArray) -> BatchArray:
        """Raw head output for a pre-processed batch."""
        ...


def import_ultralytics() -> ModuleType:
    """Import Ultralytics without its implicit network activity.

    ``YOLO_OFFLINE`` stops the DNS "is online" probe (and with it hub/telemetry calls) at import;
    ``YOLO_AUTOINSTALL=false`` stops it from pip-installing missing packages at runtime.
    """
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_VERBOSE", "false")
    # import_module is typed (ModuleType) whether or not ultralytics is installed, so mypy gives
    # the same result in the full environment and in CI, which has no PyTorch/Ultralytics.
    return importlib.import_module("ultralytics")


class TorchBackend:
    """Ultralytics YOLO weights (``.pt``) in PyTorch eager mode, FP32 on CPU."""

    def __init__(self, weights: Path, threads: int | None = None) -> None:
        import torch

        if threads is not None:
            torch.set_num_threads(threads)
        self._torch = torch
        yolo = import_ultralytics().YOLO(str(weights), task="detect")
        model = yolo.model.float().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.fuse(verbose=False)
        self.names = [str(yolo.names[i]) for i in range(len(yolo.names))]
        self.name = "pytorch-fp32"

    def settings(self) -> dict[str, Any]:
        torch = self._torch
        return {
            "runtime": "pytorch",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "thread_pinning": os.environ.get("OMP_PROC_BIND", "not set (OpenMP default)"),
            "input": "fixed 640x640 letterbox (auto=False)",
        }

    def infer(self, batch: BatchArray) -> BatchArray:
        with self._torch.inference_mode():
            out = self.model(self._torch.from_numpy(batch))
        if isinstance(out, tuple | list):  # eval mode returns (decoded, raw feature dict)
            out = out[0]
        return np.asarray(out.float().numpy(), dtype=np.float32)


class OnnxRuntimeBackend:
    """An ONNX model on ONNX Runtime's CPU execution provider."""

    def __init__(self, model: Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        if threads is not None:
            options.intra_op_num_threads = threads
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self.session.get_inputs()[0].name
        self._output = self.session.get_outputs()[0].name
        self._options = options
        self.name = "onnxruntime-fp32"

    def settings(self) -> dict[str, Any]:
        o = self._options
        return {
            "runtime": "onnxruntime",
            "providers": self.session.get_providers(),
            "intra_op_num_threads": o.intra_op_num_threads,  # 0 = runtime default
            "inter_op_num_threads": o.inter_op_num_threads,
            "execution_mode": str(o.execution_mode),
            "graph_optimization_level": str(o.graph_optimization_level),
            "thread_pinning": "none (ONNX Runtime default)",
        }

    def infer(self, batch: BatchArray) -> BatchArray:
        out = self.session.run([self._output], {self._input: batch})[0]
        return np.asarray(out, dtype=np.float32)


class OpenVINOBackend:
    """An OpenVINO IR (``.xml`` + ``.bin``) on the OpenVINO CPU plugin.

    One infer request is reused for every call, so an instance must not be used from several
    threads at once (the HTTP service runs all inference on one worker thread).
    """

    def __init__(
        self,
        model: Path,
        threads: int | None = None,
        precision: Precision = "f32",
        label: str = "fp32",
        cpu_pinning: bool | None = None,
        extra_config: dict[Any, Any] | None = None,
    ) -> None:
        import openvino as ov
        import openvino.properties as props
        import openvino.properties.hint as hints

        core = ov.Core()
        config: dict[Any, Any] = {
            hints.performance_mode: hints.PerformanceMode.LATENCY,
            props.num_streams: 1,
            hints.inference_precision: ov.Type.bf16 if precision == "bf16" else ov.Type.f32,
        }
        if threads is not None:
            config[props.inference_num_threads] = threads
        if cpu_pinning is not None:
            config[hints.enable_cpu_pinning] = cpu_pinning
        config.update(extra_config or {})
        self.compiled = core.compile_model(core.read_model(model), "CPU", config)
        self._request = self.compiled.create_infer_request()
        self.name = f"openvino-{label}"

    @property
    def request(self) -> Any:
        """The reused infer request (profiling reads its per-node counters)."""
        return self._request

    def settings(self) -> dict[str, Any]:
        out: dict[str, Any] = {"runtime": "openvino"}
        for key in OPENVINO_PROPERTIES:
            try:
                out[key] = str(self.compiled.get_property(key))
            except Exception:  # not every property exists in every OpenVINO version
                continue
        return out

    def infer(self, batch: BatchArray) -> BatchArray:
        self._request.infer({0: batch})
        return np.array(self._request.get_output_tensor(0).data, dtype=np.float32)
