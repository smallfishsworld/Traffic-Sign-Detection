"""Service configuration from environment variables (prefix ``TSD_``) or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TSD_", env_file=".env", extra="ignore")

    model_path: Path = Field(
        Path("models/yolo11n_openvino_int8/yolo11n.xml"),
        description="Model file: .xml (OpenVINO), .onnx (ONNX Runtime) or .pt (PyTorch).",
    )
    openvino_precision: Literal["f32", "bf16"] = "f32"
    threads: int = Field(2, ge=1, le=64, description="Inference threads.")
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)

    # Dynamic micro-batching: requests that arrive within max_wait_ms of the first queued one are
    # run as one batch of at most max_batch_size images. max_batch_size=1 disables batching.
    # batch_wait "adaptive" skips the wait when nothing else is queued (no wait at low load);
    # "fixed" always waits up to max_wait_ms after the first request arrived.
    max_batch_size: int = Field(8, ge=1, le=64)
    max_wait_ms: float = Field(5.0, ge=0.0, le=1000.0)
    batch_wait: Literal["adaptive", "fixed"] = "adaptive"
    queue_size: int = Field(64, ge=1, le=10_000, description="Pending images; more -> HTTP 429.")
    preprocess_threads: int = Field(2, ge=1, le=32, description="Decode/letterbox threads.")
    # Admission control: /v1/detect requests admitted (being received, decoded, queued or run)
    # at once; more get HTTP 429 before their upload is read. None: queue_size +
    # max_batch_size + preprocess_threads.
    max_inflight: int | None = Field(None, ge=1, le=10_000)

    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_detections: int = Field(300, ge=1, le=10_000)

    max_upload_bytes: int = Field(10 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(40_000_000, ge=1)
    warmup_iterations: int = Field(3, ge=0, le=100)
