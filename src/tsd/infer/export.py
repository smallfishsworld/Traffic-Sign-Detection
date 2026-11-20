"""Export YOLO weights to ONNX and OpenVINO IR and check numerical parity with PyTorch.

* ONNX: the model is prepared the way Ultralytics' exporter prepares it (eval, FP32, Conv+BN
  fused, detect head in export mode) and traced with the TorchScript-based exporter. Only the
  batch dimension is dynamic: a fixed 640x640 input lets the anchor grid be a constant and lets
  the runtimes plan memory, while the HTTP service can still batch requests.
* OpenVINO IR: converted from the ONNX file, so both runtimes execute the same graph. Weights are
  saved in FP32 (``compress_to_fp16=False``); OpenVINO's default would store them as FP16, which
  would make the "FP32" comparison quietly lossy.
* Parity: raw head outputs of each exported model are compared with PyTorch on real images
  (maximum absolute difference of box coordinates in pixels and of class scores), and the final
  detections after NMS are matched.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from tsd.boxes import box_iou
from tsd.infer.backends import Backend, import_ultralytics
from tsd.infer.detector import ModelMeta
from tsd.infer.postprocess import NMSConfig, select_detections
from tsd.infer.preprocess import Image, preprocess

DEFAULT_OPSET = 18


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions(*packages: str) -> dict[str, str]:
    return {p: version(p) for p in packages}


def export_onnx(weights: Path, out: Path, imgsz: int = 640, opset: int = DEFAULT_OPSET) -> Path:
    """Export ``weights`` (Ultralytics ``.pt``) to ONNX with a dynamic batch dimension."""
    import onnx
    import torch

    ultralytics = import_ultralytics()
    from ultralytics.nn.modules import SPPF, Detect

    yolo = ultralytics.YOLO(str(weights), task="detect")
    model = copy.deepcopy(yolo.model)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse(verbose=False)
    for m in model.modules():
        if isinstance(m, SPPF):
            m.export = True
        if isinstance(m, Detect):
            m.dynamic = False
            m.export = True
            # declared as None in Ultralytics; unused-ignore: CI type-checks without Ultralytics
            m.format = "onnx"  # type: ignore[assignment, unused-ignore]
            m.xyxy = False
            m.shape = None
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    model(dummy)  # builds the anchor grid for this input size
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        opset_version=opset,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(out)))
    ModelMeta(
        names=[str(yolo.names[i]) for i in range(len(yolo.names))],
        input_size=(imgsz, imgsz),
        precision="fp32",
        source=weights.name,
        source_sha256=_sha256(weights),
        extra={"format": "onnx", "opset": opset, **_versions("torch", "ultralytics", "onnx")},
    ).save(out)
    return out


def export_openvino(onnx_model: Path, out_dir: Path) -> Path:
    """Convert an exported ONNX model to OpenVINO IR with FP32 weights."""
    import openvino as ov

    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / f"{onnx_model.stem}.xml"
    ov.save_model(ov.convert_model(str(onnx_model)), str(xml), compress_to_fp16=False)
    meta = ModelMeta.load(onnx_model)
    extra = {**meta.extra, "format": "openvino", **_versions("openvino")}
    ModelMeta(meta.names, meta.input_size, "fp32", meta.source, meta.source_sha256, extra).save(xml)
    return xml


# Gate for FP32 exports (both runtimes compute the same FP32 network): raw outputs within these
# bounds and at least this share of detections matched in both directions.
PARITY_TOLERANCE: dict[str, float] = {
    "max_abs_box_px": 0.5,
    "max_abs_score": 1e-3,
    "min_match_rate": 0.98,
}


@dataclass(frozen=True)
class ParityReport:
    reference: str
    candidate: str
    images: int
    input_size: tuple[int, int]
    max_abs_box_px: float
    mean_abs_box_px: float
    max_abs_score: float
    mean_abs_score: float
    reference_detections: int
    matched_detections: int
    candidate_detections: int
    unmatched_candidates: int
    max_matched_score_diff: float

    def failures(self, tolerance: dict[str, float] | None = None) -> list[str]:
        """Which tolerances this comparison exceeds (empty: it passes the gate)."""
        tol = tolerance or PARITY_TOLERANCE
        out = []
        if self.max_abs_box_px > tol["max_abs_box_px"]:
            out.append(f"box difference {self.max_abs_box_px:.4f} px")
        if self.max_abs_score > tol["max_abs_score"]:
            out.append(f"score difference {self.max_abs_score:.2e}")
        if self.matched_detections < tol["min_match_rate"] * self.reference_detections:
            out.append(f"{self.matched_detections}/{self.reference_detections} reference matched")
        matched_cand = self.candidate_detections - self.unmatched_candidates
        if matched_cand < tol["min_match_rate"] * self.candidate_detections:
            out.append(f"{self.unmatched_candidates} unmatched candidate detections")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "failures": self.failures()}


def compare_raw_outputs(
    reference: Backend,
    candidate: Backend,
    images: Sequence[Image],
    nms: NMSConfig | None = None,
    match_iou: float = 0.9,
    input_size: tuple[int, int] = (640, 640),
) -> ParityReport:
    """Compare two backends on the same pre-processed images (letterboxed to ``input_size``, the
    exported model's input).

    A detection counts as matched when the other backend has a detection of the same class with
    IoU >= ``match_iou``; matching is checked in both directions.
    """
    batch, _ = preprocess(images, input_size)
    ref = np.concatenate([reference.infer(batch[i : i + 1]) for i in range(len(batch))])
    cand = np.concatenate([candidate.infer(batch[i : i + 1]) for i in range(len(batch))])
    box_diff = np.abs(ref[:, :4] - cand[:, :4])
    score_diff = np.abs(ref[:, 4:] - cand[:, 4:])
    config = nms or NMSConfig()
    total = matched = cand_total = cand_unmatched = 0
    worst = 0.0
    for r, c in zip(ref, cand, strict=True):
        rb, rs, rc = select_detections(r, config)
        cb, cs, cc = select_detections(c, config)
        total += len(rb)
        cand_total += len(cb)
        if len(rb) == 0 or len(cb) == 0:
            cand_unmatched += len(cb)
            continue
        iou = box_iou(rb, cb) * (rc[:, None] == cc[None, :])
        best = iou.argmax(axis=1)
        ok = iou[np.arange(len(rb)), best] >= match_iou
        matched += int(ok.sum())
        cand_unmatched += int((iou.max(axis=0) < match_iou).sum())
        if ok.any():
            worst = max(worst, float(np.abs(rs[ok] - cs[best[ok]]).max()))
    return ParityReport(
        reference=reference.name,
        candidate=candidate.name,
        images=len(images),
        input_size=input_size,
        max_abs_box_px=float(box_diff.max()),
        mean_abs_box_px=float(box_diff.mean()),
        max_abs_score=float(score_diff.max()),
        mean_abs_score=float(score_diff.mean()),
        reference_detections=total,
        matched_detections=matched,
        candidate_detections=cand_total,
        unmatched_candidates=cand_unmatched,
        max_matched_score_diff=worst,
    )
