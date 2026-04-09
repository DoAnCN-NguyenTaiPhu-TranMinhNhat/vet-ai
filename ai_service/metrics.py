from __future__ import annotations

import time
from typing import Any, Optional

from prometheus_client import Counter, Histogram


INFERENCE_LATENCY = Histogram(
    "vetai_inference_latency_seconds",
    "Thoi gian suy luan AI (giay)",
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10),
)

INFERENCE_REQUESTS = Counter(
    "vetai_inference_requests_total",
    "So luong goi /predict",
    ["clinic_id", "status"],
)

PREDICTION_CONFIDENCE = Histogram(
    "vetai_prediction_confidence",
    "Do tu tin cua AI tren tung ca",
    buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
)

FEEDBACK_ACCEPT = Counter(
    "vetai_feedback_accept_total",
    "So ca bac si dong y voi AI",
    ["clinic_id"],
)

FEEDBACK_REJECT = Counter(
    "vetai_feedback_reject_total",
    "So ca bac si sua lai ket qua AI",
    ["clinic_id"],
)

TRAINING_JOBS = Counter(
    "vetai_training_job_runs_total",
    "So job training da chay",
    ["status"],
)

TRAINING_DURATION = Histogram(
    "vetai_training_duration_seconds",
    "Thoi gian chay 1 job training (giay)",
    buckets=(10, 30, 60, 120, 300, 900),
)

MODEL_RELOAD = Counter(
    "vetai_model_reload_total",
    "So lan reload model sau training/promote",
    ["scope", "result"],
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
            # metric khong duoc quyen lam hong request
            pass


def inc_feedback(clinic_id: Optional[str], is_correct: Optional[bool]) -> None:
    """
    Feedback accept/reject:
      - True  -> accept
      - False -> reject
      - None  -> ignore (khong ro)
    """
    cid = clinic_id or "global"
    if is_correct is True:
        FEEDBACK_ACCEPT.labels(clinic_id=cid).inc()
    elif is_correct is False:
        FEEDBACK_REJECT.labels(clinic_id=cid).inc()


def inc_training_job(status: str) -> None:
    # status: pending|running|completed|failed
    TRAINING_JOBS.labels(status=str(status)).inc()


def observe_training_duration(seconds: float) -> None:
    try:
        TRAINING_DURATION.observe(float(seconds))
    except Exception:
        pass


def inc_model_reload(scope: str, result: str) -> None:
    # scope: global|clinic ; result: success|fallback|error
    MODEL_RELOAD.labels(scope=str(scope), result=str(result)).inc()


def timing() -> Any:
    """Context manager nho gon de do thoi gian (second)."""

    class _Timer:
        def __enter__(self) -> "_Timer":
            self._start = time.perf_counter()
            self.elapsed = 0.0
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.elapsed = time.perf_counter() - self._start

    return _Timer()

