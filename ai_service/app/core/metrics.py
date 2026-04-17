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
TRAINING_JOBS = Counter(
    "vetai_training_job_events_total",
    "So luong su kien training job theo trang thai",
    ["status"],
)
MODEL_RELOADS = Counter(
    "vetai_model_reload_total",
    "So lan doi active model sau training",
    ["scope", "status"],
)
TRAINING_DURATION = Histogram(
    "vetai_training_duration_seconds",
    "Thoi gian huan luyen model (giay)",
    buckets=(1, 3, 5, 10, 20, 30, 60, 120, 300, 600),
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


def inc_training_job(status: str) -> None:
    st = (status or "unknown").strip().lower() or "unknown"
    TRAINING_JOBS.labels(status=st).inc()


def inc_model_reload(scope: str, status: str) -> None:
    sc = (scope or "global").strip().lower() or "global"
    st = (status or "unknown").strip().lower() or "unknown"
    MODEL_RELOADS.labels(scope=sc, status=st).inc()


def observe_training_duration(seconds: float) -> None:
    try:
        TRAINING_DURATION.observe(float(seconds))
    except Exception:
        pass
