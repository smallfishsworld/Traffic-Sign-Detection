"""FastAPI inference service with dynamic micro-batching and Prometheus metrics."""

from tsd.serve.app import create_app
from tsd.serve.settings import ServeSettings

__all__ = ["ServeSettings", "create_app"]
