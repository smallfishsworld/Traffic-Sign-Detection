"""Prometheus metrics of one service instance (own registry, so several apps can coexist)."""

from __future__ import annotations

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0)


@dataclass
class ServeMetrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        r = self.registry
        self.requests = Counter(
            "tsd_requests", "HTTP requests by endpoint and status code.", ["endpoint", "status"],
            registry=r,
        )  # fmt: skip
        self.http_duration = Histogram(
            "tsd_http_request_duration_seconds",
            "Time from the request reaching the application to the end of the response,"
            " every status code.",
            ["endpoint", "status"], buckets=LATENCY_BUCKETS, registry=r,
        )  # fmt: skip
        self.handler_latency = Histogram(
            "tsd_detect_handler_seconds",
            "Successful /v1/detect handler time after the upload was parsed (decode, queue,"
            " inference, NMS).",
            buckets=LATENCY_BUCKETS, registry=r,
        )  # fmt: skip
        self.queue_wait = Histogram(
            "tsd_queue_wait_seconds", "Time from enqueue to the start of its batch.",
            buckets=LATENCY_BUCKETS, registry=r,
        )  # fmt: skip
        self.inference_latency = Histogram(
            "tsd_batch_inference_seconds", "Inference + NMS time per batch.",
            buckets=LATENCY_BUCKETS, registry=r,
        )  # fmt: skip
        self.batch_size = Histogram(
            "tsd_batch_size", "Images per inference batch.",
            buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64), registry=r,
        )  # fmt: skip
        self.queue_depth = Gauge("tsd_queue_depth", "Images waiting for inference.", registry=r)
        self.rejected = Counter(
            "tsd_rejected", "Requests rejected with 429: admission limit or full queue.",
            ["reason"], registry=r,
        )  # fmt: skip
        self.disconnects = Counter(
            "tsd_client_disconnects", "Requests dropped because the client went away (499).",
            registry=r,
        )  # fmt: skip
        self.model_info = Gauge(
            "tsd_model_info", "Loaded model (value is always 1).", ["model", "backend"], registry=r
        )
