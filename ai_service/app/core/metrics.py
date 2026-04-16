from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator


def setup_metrics(app: FastAPI) -> None:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


INFERENCE_LATENCY = Histogram(
    "vetai_inference_latency_seconds",
    "Thoi gian suy luan AI (giay)",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10),
)
INFERENCE_REQUESTS = Counter("vetai_inference_requests_total", "So luong goi /predict", ["clinic_id", "status"])
PREDICTION_CONFIDENCE = Histogram(
    "vetai_prediction_confidence",
    "Do tu tin cua AI tren tung ca",
    buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
)


def observe_inference(
    clinic_id: Optional[str],
    ok: bool,
    latency_seconds: float,
    confidence: Optional[float] = None,
) -> None:
    cid = clinic_id or "global"
    INFERENCE_LATENCY.observe(latency_seconds)
    INFERENCE_REQUESTS.labels(clinic_id=cid, status="success" if ok else "error").inc()
    if confidence is not None:
        try:
            PREDICTION_CONFIDENCE.observe(float(confidence))
        except Exception:
            pass


def timing() -> Any:
    class _Timer:
        def __enter__(self) -> "_Timer":
            self._start = time.perf_counter()
            self.elapsed = 0.0
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.elapsed = time.perf_counter() - self._start

    return _Timer()
