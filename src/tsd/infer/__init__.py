"""Multi-backend CPU inference: one Detector interface over PyTorch, ONNX Runtime and OpenVINO."""

from tsd.infer.detector import Detector, ModelMeta, StageTimes, backend_for, load_detector
from tsd.infer.postprocess import EVAL_NMS, NMSConfig
from tsd.infer.preprocess import LetterboxInfo, letterbox
from tsd.infer.types import Detections

__all__ = [
    "EVAL_NMS",
    "Detections",
    "Detector",
    "LetterboxInfo",
    "ModelMeta",
    "NMSConfig",
    "StageTimes",
    "backend_for",
    "letterbox",
    "load_detector",
]
