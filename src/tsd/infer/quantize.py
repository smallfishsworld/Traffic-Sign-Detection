"""Post-training INT8 quantization of an OpenVINO IR with NNCF.

Calibration runs the FP32 model on images pre-processed exactly as at inference time (the shared
letterbox) to collect activation ranges. Two choices matter for detection accuracy:

* ``preset=MIXED``: symmetric INT8 weights, asymmetric activations. YOLO uses SiLU, whose
  outputs are not centred on zero, so asymmetric activation ranges waste fewer INT8 levels.
* The detect head's box-decoding subgraph is kept in floating point (``ignore_head=True``): the
  DFL softmax/expectation, the anchor-grid arithmetic that turns distances into pixel coordinates
  (values up to 640), and the class sigmoid. Only the head's convolution branches (``cv2``,
  ``cv3``) are quantized. Quantizing the decode step would round box coordinates to a few pixels.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tsd.infer.detector import ModelMeta
from tsd.infer.preprocess import letterbox, to_input_tensor
from tsd.io import load_image

_SKIP_TYPES = {"Constant", "Parameter", "Result"}


def detect_head_prefix(op_names: Sequence[str]) -> str:
    """Name prefix of the detect head (the module that owns the DFL), e.g. ``/model.23``."""
    for name in op_names:
        match = re.match(r"(/model\.\d+)/dfl/", name)
        if match:
            return match.group(1)
    raise ValueError("no DFL node found: is this a YOLOv8/YOLO11 detection model?")


def head_decode_ops(ops: Sequence[tuple[str, str]]) -> list[str]:
    """Names of the detect-head ops outside its convolution branches.

    ``ops`` are ``(friendly_name, type_name)`` pairs of the model's operations.
    """
    prefix = detect_head_prefix([name for name, _ in ops])
    branches = (f"{prefix}/cv2", f"{prefix}/cv3")
    return [
        name
        for name, kind in ops
        if name.startswith(prefix + "/")
        and not name.startswith(branches)
        and kind not in _SKIP_TYPES
    ]


def quantize_int8(
    fp32_xml: Path,
    calibration_images: Sequence[Path],
    out_xml: Path,
    ignore_head: bool = True,
) -> dict[str, Any]:
    """Quantize ``fp32_xml`` with NNCF and save the INT8 IR to ``out_xml``; returns a summary."""
    import nncf
    import openvino as ov

    if not calibration_images:
        raise ValueError("no calibration images")
    meta = ModelMeta.load(fp32_xml)
    model = ov.Core().read_model(str(fp32_xml))

    def transform(path: Path) -> Any:
        boxed, _ = letterbox(load_image(path), meta.input_size)
        return to_input_tensor([boxed])

    ops = [(op.get_friendly_name(), op.get_type_name()) for op in model.get_ordered_ops()]
    ignored = head_decode_ops(ops) if ignore_head else []
    quantized = nncf.quantize(
        model,
        nncf.Dataset(list(calibration_images), transform),
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calibration_images),
        ignored_scope=nncf.IgnoredScope(names=ignored) if ignored else None,
    )
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(out_xml), compress_to_fp16=False)
    fake_quantize = sum(op.get_type_name() == "FakeQuantize" for op in quantized.get_ops())
    summary = {
        "calibration_images": len(calibration_images),
        "preset": "mixed",
        "ignore_head_decode": ignore_head,
        "ignored_ops": len(ignored),
        "fake_quantize_ops": fake_quantize,
        "nncf": version("nncf"),
        "openvino": version("openvino"),
    }
    extra = {**meta.extra, "quantization": summary}
    precision = "int8" if ignore_head else "int8-fullq"  # also names the backend in reports
    ModelMeta(meta.names, meta.input_size, precision, meta.source, meta.source_sha256, extra).save(
        out_xml
    )
    return summary
