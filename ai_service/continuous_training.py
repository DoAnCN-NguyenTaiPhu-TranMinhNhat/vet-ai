"""
Continuous Training endpoints for FastAPI service
Handles prediction logging and training trigger logic
"""

from datetime import datetime, timedelta
from io import StringIO
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from fastapi import HTTPException, BackgroundTasks, APIRouter, Depends
from fastapi.responses import StreamingResponse
import asyncio
import logging
import os
import json
import uuid
import time
import subprocess
import boto3
import csv
from prometheus_client import Gauge, Counter
import psycopg2
from psycopg2.extras import RealDictCursor, Json

# Import kubernetes only when needed for EKS training
try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    client = None
    config = None

# Import training engine
try:
    from ai_service.training_engine import execute_training
except ImportError:
    from training_engine import execute_training

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/continuous-training", tags=["continuous-training"])
from ai_service.auth import verify_admin

# Configuration (mutable for P2 UI)
TRAINING_THRESHOLD = int(os.getenv("TRAINING_THRESHOLD", "10"))  # Reduced from 100 for easier testing
TRAINING_WINDOW_DAYS = int(os.getenv("TRAINING_WINDOW_DAYS", "30"))
EKS_CLUSTER_NAME = os.getenv("EKS_CLUSTER_NAME", "vet-ai-dev")
EKS_REGION = os.getenv("EKS_REGION", "us-east-1")
TRAINING_NODE_GROUP = os.getenv("TRAINING_NODE_GROUP", "training-nodes")
DATABASE_URL = os.getenv("DATABASE_URL")


class CTPostgresStore:
    """Lightweight PostgreSQL store for continuous-training data."""

    def __init__(self, dsn: Optional[str]) -> None:
        self.dsn = dsn
        self.enabled = bool(dsn and str(dsn).strip())
        self._initialized = False

    def _conn(self):
        if not self.enabled:
            raise RuntimeError("DATABASE_URL is not configured for continuous training.")
        return psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)

    def _ensure_schema(self) -> None:
        if not self.enabled or self._initialized:
            return
        ddl = """
        CREATE TABLE IF NOT EXISTS ai_prediction_logs (
            id BIGINT PRIMARY KEY,
            visit_id BIGINT NULL,
            pet_id BIGINT NOT NULL,
            prediction_input JSONB NOT NULL,
            prediction_output JSONB NOT NULL,
            model_version TEXT NOT NULL,
            confidence_score DOUBLE PRECISION NOT NULL,
            top_k_predictions JSONB NOT NULL,
            veterinarian_id BIGINT NULL,
            clinic_id BIGINT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ai_feedback (
            id BIGSERIAL PRIMARY KEY,
            prediction_id BIGINT NOT NULL REFERENCES ai_prediction_logs(id) ON DELETE CASCADE,
            final_diagnosis TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            ai_diagnosis TEXT NULL,
            confidence_rating INTEGER NULL,
            comments TEXT NULL,
            veterinarian_id BIGINT NOT NULL,
            is_training_eligible BOOLEAN NOT NULL DEFAULT TRUE,
            data_quality_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_ai_feedback_created_at ON ai_feedback(created_at);
        CREATE INDEX IF NOT EXISTS idx_ai_feedback_eligible ON ai_feedback(is_training_eligible);

        -- Persisted training job status so FastAPI restart won't lose history.
        CREATE TABLE IF NOT EXISTS ai_training_jobs (
            training_id BIGSERIAL PRIMARY KEY,
            status TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL DEFAULT NOW(),
            end_time TIMESTAMP NULL,
            total_predictions BIGINT NOT NULL DEFAULT 0,
            eligible_feedback_count BIGINT NOT NULL DEFAULT 0,
            previous_model_version TEXT NULL,
            new_model_version TEXT NULL,
            training_accuracy DOUBLE PRECISION NULL,
            validation_accuracy DOUBLE PRECISION NULL,
            f1_score DOUBLE PRECISION NULL,
            is_deployed BOOLEAN NOT NULL DEFAULT FALSE,
            error_message TEXT NULL,
            trigger_type TEXT NULL,
            training_mode TEXT NULL,
            eks_node_group TEXT NULL,
            dataset_row_count BIGINT NULL,
            small_sample_warning BOOLEAN NULL,
            metrics_note TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_ai_training_jobs_status ON ai_training_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_ai_training_jobs_start_time ON ai_training_jobs(start_time);
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        self._initialized = True

    def insert_prediction(self, payload: Dict[str, Any]) -> int:
        self._ensure_schema()
        sql = """
        INSERT INTO ai_prediction_logs
            (id, visit_id, pet_id, prediction_input, prediction_output, model_version, confidence_score, top_k_predictions, veterinarian_id, clinic_id)
        VALUES
            (%(id)s, %(visit_id)s, %(pet_id)s, %(prediction_input)s, %(prediction_output)s, %(model_version)s, %(confidence_score)s, %(top_k_predictions)s, %(veterinarian_id)s, %(clinic_id)s)
        ON CONFLICT (id) DO UPDATE SET
            visit_id = EXCLUDED.visit_id,
            pet_id = EXCLUDED.pet_id,
            prediction_input = EXCLUDED.prediction_input,
            prediction_output = EXCLUDED.prediction_output,
            model_version = EXCLUDED.model_version,
            confidence_score = EXCLUDED.confidence_score,
            top_k_predictions = EXCLUDED.top_k_predictions,
            veterinarian_id = EXCLUDED.veterinarian_id,
            clinic_id = EXCLUDED.clinic_id
        """
        data = dict(payload)
        data["prediction_input"] = Json(payload.get("prediction_input") or {})
        data["prediction_output"] = Json(payload.get("prediction_output") or {})
        data["top_k_predictions"] = Json(payload.get("top_k_predictions") or [])
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, data)
            conn.commit()
        return int(payload["id"])

    def insert_feedback(self, payload: Dict[str, Any]) -> bool:
        self._ensure_schema()
        sql = """
        INSERT INTO ai_feedback
            (prediction_id, final_diagnosis, is_correct, ai_diagnosis, confidence_rating, comments, veterinarian_id, is_training_eligible, data_quality_score)
        VALUES
            (%(prediction_id)s, %(final_diagnosis)s, %(is_correct)s, %(ai_diagnosis)s, %(confidence_rating)s, %(comments)s, %(veterinarian_id)s, %(is_training_eligible)s, %(data_quality_score)s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, payload)
            conn.commit()
        return True

    def count_predictions(self) -> int:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*)::BIGINT AS c FROM ai_prediction_logs")
                return int(cur.fetchone()["c"])

    def count_feedback(self, eligible_only: bool = False, days: Optional[int] = None) -> int:
        self._ensure_schema()
        cond = []
        params: List[Any] = []
        if eligible_only:
            cond.append("is_training_eligible = TRUE")
        if days is not None:
            cond.append("created_at >= NOW() - (%s || ' days')::INTERVAL")
            params.append(int(days))
        where = f"WHERE {' AND '.join(cond)}" if cond else ""
        sql = f"SELECT COUNT(*)::BIGINT AS c FROM ai_feedback {where}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return int(cur.fetchone()["c"])

    def fetch_predictions(self) -> List[Dict[str, Any]]:
        self._ensure_schema()
        sql = """
        SELECT id, visit_id, pet_id, prediction_input, prediction_output, model_version, confidence_score,
               top_k_predictions, veterinarian_id, clinic_id, created_at
        FROM ai_prediction_logs
        ORDER BY created_at ASC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_feedback(self, eligible_only: bool = False, days: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_schema()
        cond = []
        params: List[Any] = []
        if eligible_only:
            cond.append("is_training_eligible = TRUE")
        if days is not None:
            cond.append("created_at >= NOW() - (%s || ' days')::INTERVAL")
            params.append(int(days))
        where = f"WHERE {' AND '.join(cond)}" if cond else ""
        sql = f"""
        SELECT prediction_id, final_diagnosis, is_correct, ai_diagnosis, confidence_rating, comments,
               veterinarian_id, is_training_eligible, data_quality_score, created_at AS timestamp
        FROM ai_feedback
        {where}
        ORDER BY created_at ASC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def clear_all(self) -> Dict[str, int]:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*)::BIGINT AS c FROM ai_prediction_logs")
                p = int(cur.fetchone()["c"])
                cur.execute("SELECT COUNT(*)::BIGINT AS c FROM ai_feedback")
                f = int(cur.fetchone()["c"])
                cur.execute("DELETE FROM ai_feedback")
                cur.execute("DELETE FROM ai_prediction_logs")
            conn.commit()
        return {"predictions": p, "feedback": f}

    def clear_feedback(self) -> int:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*)::BIGINT AS c FROM ai_feedback")
                f = int(cur.fetchone()["c"])
                cur.execute("DELETE FROM ai_feedback")
            conn.commit()
        return f

    def create_training_job(
        self,
        *,
        status: str,
        total_predictions: int,
        eligible_feedback_count: int,
        previous_model_version: Optional[str],
        trigger_type: Optional[str],
        training_mode: Optional[str],
        eks_node_group: Optional[str],
        dataset_row_count: Optional[int],
    ) -> int:
        """Create a persisted training job record (returns training_id)."""
        self._ensure_schema()
        sql = """
        INSERT INTO ai_training_jobs (
            status,
            start_time,
            total_predictions,
            eligible_feedback_count,
            previous_model_version,
            trigger_type,
            training_mode,
            eks_node_group,
            dataset_row_count,
            created_at,
            updated_at
        ) VALUES (
            %(status)s,
            NOW(),
            %(total_predictions)s,
            %(eligible_feedback_count)s,
            %(previous_model_version)s,
            %(trigger_type)s,
            %(training_mode)s,
            %(eks_node_group)s,
            %(dataset_row_count)s,
            NOW(),
            NOW()
        )
        RETURNING training_id
        """
        params = {
            "status": status,
            "total_predictions": int(total_predictions),
            "eligible_feedback_count": int(eligible_feedback_count),
            "previous_model_version": previous_model_version,
            "trigger_type": trigger_type,
            "training_mode": training_mode,
            "eks_node_group": eks_node_group,
            "dataset_row_count": dataset_row_count,
        }
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                training_id = cur.fetchone()["training_id"]
        return int(training_id)

    def update_training_job_completed(
        self,
        training_id: int,
        *,
        model_version: Optional[str],
        training_metrics: Dict[str, Any],
        is_deployed: bool = False,
    ) -> None:
        """Update training job with completion result."""
        self._ensure_schema()
        tm = training_metrics or {}
        sql = """
        UPDATE ai_training_jobs
        SET
            status = 'completed',
            end_time = NOW(),
            new_model_version = %(new_model_version)s,
            training_accuracy = %(training_accuracy)s,
            validation_accuracy = %(validation_accuracy)s,
            f1_score = %(f1_score)s,
            is_deployed = %(is_deployed)s,
            error_message = NULL,
            small_sample_warning = %(small_sample_warning)s,
            metrics_note = %(metrics_note)s,
            updated_at = NOW()
        WHERE training_id = %(training_id)s
        """
        params = {
            "training_id": int(training_id),
            "new_model_version": model_version,
            "training_accuracy": tm.get("training_accuracy"),
            "validation_accuracy": tm.get("validation_accuracy"),
            "f1_score": tm.get("validation_f1"),
            "is_deployed": bool(is_deployed),
            "small_sample_warning": tm.get("small_sample_warning"),
            "metrics_note": tm.get("metrics_note"),
        }
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    def update_training_job_failed(
        self,
        training_id: int,
        *,
        error_message: str,
    ) -> None:
        """Update training job with failed result."""
        self._ensure_schema()
        sql = """
        UPDATE ai_training_jobs
        SET
            status = 'failed',
            end_time = NOW(),
            is_deployed = FALSE,
            error_message = %(error_message)s,
            updated_at = NOW()
        WHERE training_id = %(training_id)s
        """
        params = {"training_id": int(training_id), "error_message": str(error_message)[:4000]}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    def get_training_job(self, training_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a training job record by id."""
        self._ensure_schema()
        sql = """
        SELECT
            training_id,
            status,
            start_time,
            end_time,
            total_predictions,
            eligible_feedback_count,
            previous_model_version,
            new_model_version,
            training_accuracy,
            validation_accuracy,
            f1_score,
            is_deployed,
            error_message,
            trigger_type,
            training_mode,
            eks_node_group,
            dataset_row_count,
            small_sample_warning,
            metrics_note
        FROM ai_training_jobs
        WHERE training_id = %(training_id)s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"training_id": int(training_id)})
                row = cur.fetchone()
        return dict(row) if row else None

    def list_training_jobs(self, *, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List training jobs (recent first)."""
        self._ensure_schema()
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        sql = """
        SELECT
            training_id,
            status,
            start_time,
            end_time,
            total_predictions,
            eligible_feedback_count,
            trigger_type,
            training_mode,
            previous_model_version,
            new_model_version,
            training_accuracy,
            validation_accuracy,
            f1_score,
            is_deployed,
            error_message,
            dataset_row_count,
            small_sample_warning,
            metrics_note
        FROM ai_training_jobs
        ORDER BY training_id DESC
        LIMIT %(limit)s OFFSET %(offset)s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"limit": limit, "offset": offset})
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def count_training_jobs_by_status(self) -> Dict[str, int]:
        """Count training jobs by status."""
        self._ensure_schema()
        sql = """
        SELECT status, COUNT(*)::BIGINT AS c
        FROM ai_training_jobs
        GROUP BY status
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        base = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for r in rows:
            st = str(r["status"]).lower()
            if st in base:
                base[st] = int(r["c"])
        return base

    def count_training_jobs_total(self) -> int:
        """Count total training jobs."""
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*)::BIGINT AS c FROM ai_training_jobs")
                return int(cur.fetchone()["c"])


ct_store = CTPostgresStore(DATABASE_URL)

# Trên EKS, training thật = cùng process Python với local (execute_training). Luồng eks_hybrid (Batch Job riêng)
# chỉ bật khi ALLOW_EKS_HYBRID_TRAINING=true — cần image + node group + RBAC đúng; mặc định tắt.
def _effective_training_mode(requested: str) -> str:
    if requested != "eks_hybrid":
        return requested
    allow = os.getenv("ALLOW_EKS_HYBRID_TRAINING", "false").lower() in ("1", "true", "yes", "y")
    if not allow:
        logger.info(
            "training_mode=eks_hybrid -> dùng in-process training (giống local). "
            "Đặt ALLOW_EKS_HYBRID_TRAINING=true chỉ khi đã triển khai Job training riêng."
        )
        return "local"
    return "eks_hybrid"


# Pydantic models
class PredictionLog(BaseModel):
    id: int  # Use ID from main.py
    visit_id: Optional[int] = None
    pet_id: int
    prediction_input: Dict
    prediction_output: Dict
    model_version: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    top_k_predictions: List[Dict]
    veterinarian_id: Optional[int] = None
    clinic_id: Optional[int] = None

class DoctorFeedback(BaseModel):
    prediction_id: int
    final_diagnosis: str
    is_correct: bool
    # AI-suggested label (used to apply negative training signal on reject)
    ai_diagnosis: Optional[str] = None
    confidence_rating: Optional[int] = Field(ge=0, le=5, default=None)
    comments: Optional[str] = None
    veterinarian_id: int
    is_training_eligible: bool = True
    data_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)

class TrainingTriggerRequest(BaseModel):
    trigger_type: str = Field(pattern="^(scheduled|manual|automatic)$")
    trigger_reason: Optional[str] = None
    force: bool = False  # Override threshold check
    # local = execute_training() trong pod (mặc định trên cả máy dev và EKS). eks_hybrid chỉ khi ALLOW_EKS_HYBRID_TRAINING=true.
    training_mode: str = Field(default="local", pattern="^(local|eks_hybrid)$")
    eks_node_group: Optional[str] = None

class TrainingStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    training_id: int
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_predictions: int
    eligible_feedback_count: int
    previous_model_version: Optional[str]
    new_model_version: Optional[str]
    training_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    f1_score: Optional[float]
    is_deployed: bool
    error_message: Optional[str]
    small_sample_warning: Optional[bool] = None
    metrics_note: Optional[str] = None


class TrainingPolicyUpdate(BaseModel):
    training_threshold: int = Field(ge=1)
    training_window_days: int = Field(ge=1)

# Database functions
async def log_prediction(prediction: PredictionLog) -> int:
    """Log prediction to PostgreSQL storage."""
    logger.info(f"Logging prediction for visit {prediction.visit_id}, ID: {prediction.id}")

    prediction_data = {
        "id": prediction.id,
        "visit_id": prediction.visit_id,
        "pet_id": prediction.pet_id,
        "prediction_input": prediction.prediction_input,
        "prediction_output": prediction.prediction_output,
        "model_version": prediction.model_version,
        "confidence_score": prediction.confidence_score,
        "top_k_predictions": prediction.top_k_predictions,
        "veterinarian_id": prediction.veterinarian_id,
        "clinic_id": prediction.clinic_id,
    }
    ct_store.insert_prediction(prediction_data)
    _predictions_logged_total.inc()
    _refresh_training_metrics()

    logger.info(f"Prediction logged successfully. ID: {prediction.id}")
    return prediction.id

async def save_feedback(feedback: DoctorFeedback) -> bool:
    """Save doctor feedback to PostgreSQL storage."""
    logger.info(f"Saving feedback for prediction {feedback.prediction_id}")

    feedback_data_item = {
        "prediction_id": feedback.prediction_id,
        "final_diagnosis": feedback.final_diagnosis,
        "is_correct": feedback.is_correct,
        "ai_diagnosis": feedback.ai_diagnosis,
        "confidence_rating": feedback.confidence_rating,
        "comments": feedback.comments,
        "veterinarian_id": feedback.veterinarian_id or 0,
        "is_training_eligible": feedback.is_training_eligible,
        "data_quality_score": feedback.data_quality_score,
    }
    ct_store.insert_feedback(feedback_data_item)
    _feedback_saved_total.inc()
    _refresh_training_metrics()

    logger.info("Feedback saved successfully.")
    return True

# In-memory dataset snapshot for download endpoints.
# Training job status is persisted in PostgreSQL (ai_training_jobs).
training_jobs: Dict[int, Dict[str, Any]] = {}
training_datasets: Dict[int, List[Dict[str, Any]]] = {}

# Initialize in-memory runtime caches
training_jobs.clear()
training_datasets.clear()

logger.info("Production-ready continuous training system initialized")

_predictions_logged_total = Counter("vetai_predictions_logged_total", "Total predictions logged for continuous training")
_feedback_saved_total = Counter("vetai_feedback_saved_total", "Total feedback entries saved")
_eligible_feedback_gauge = Gauge("vetai_feedback_eligible_count", "Eligible feedback count for training")
_training_jobs_gauge = Gauge("vetai_training_jobs", "Training jobs by status", ["status"])
_training_last_success_timestamp = Gauge("vetai_training_last_success_timestamp_seconds", "Last successful training time (unix seconds)")


def _refresh_training_metrics() -> None:
    try:
        eligible = ct_store.count_feedback(eligible_only=True, days=TRAINING_WINDOW_DAYS)
        _eligible_feedback_gauge.set(float(eligible))

        counts = ct_store.count_training_jobs_by_status()
        for st in ["pending", "running", "completed", "failed"]:
            _training_jobs_gauge.labels(status=st).set(float(counts.get(st, 0)))

        completed = [
            j
            for j in ct_store.list_training_jobs(limit=20, offset=0)
            if str(j.get("status", "")).lower() == "completed" and j.get("end_time")
        ]
        if completed:
            last = max(completed, key=lambda j: j.get("end_time"))
            _training_last_success_timestamp.set(float(last["end_time"].timestamp()))
    except Exception:
        return

async def count_eligible_feedback(days: Optional[int] = None) -> int:
    """Count eligible feedback in the last N days"""
    if days is None:
        days = TRAINING_WINDOW_DAYS
    eligible_count = ct_store.count_feedback(eligible_only=True, days=days)
    logger.info(f"Current eligible feedback count: {eligible_count}")
    return eligible_count


def build_training_dataset_snapshot(
    feedback_items: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build the raw feedback-linked dataset that is fed into training.
    This captures the source rows before weighting / core-memory augmentation.
    """
    rows: List[Dict[str, Any]] = []
    if not feedback_items:
        return rows

    pred_by_id = {}
    for p in predictions:
        if isinstance(p, dict):
            pid = p.get("id")
            if pid is not None:
                pred_by_id[pid] = p

    for f in feedback_items:
        if not isinstance(f, dict):
            continue
        pred = pred_by_id.get(f.get("prediction_id"))
        if pred is None:
            continue
        pred_input = pred.get("prediction_input") or {}
        rows.append(
            {
                "prediction_id": f.get("prediction_id"),
                "timestamp": (
                    f.get("timestamp").isoformat()
                    if hasattr(f.get("timestamp"), "isoformat")
                    else str(f.get("timestamp", ""))
                ),
                "final_diagnosis": f.get("final_diagnosis"),
                "ai_diagnosis": f.get("ai_diagnosis"),
                "is_correct": f.get("is_correct"),
                "confidence_rating": f.get("confidence_rating"),
                "data_quality_score": f.get("data_quality_score"),
                "animal_type": pred_input.get("animal_type"),
                "gender": pred_input.get("gender"),
                "age_months": pred_input.get("age_months"),
                "weight_kg": pred_input.get("weight_kg"),
                "temperature": pred_input.get("temperature"),
                "heart_rate": pred_input.get("heart_rate"),
                "current_season": pred_input.get("current_season"),
                "vaccination_status": pred_input.get("vaccination_status"),
                "medical_history": pred_input.get("medical_history"),
                "symptom_duration": pred_input.get("symptom_duration"),
                "symptoms_list": pred_input.get("symptoms_list"),
            }
        )
    return rows


@router.get("/config")
async def get_training_policy():
    return {
        "training_threshold": TRAINING_THRESHOLD,
        "training_window_days": TRAINING_WINDOW_DAYS,
    }


@router.put("/config")
async def update_training_policy(update: TrainingPolicyUpdate, _: bool = Depends(verify_admin)):
    global TRAINING_THRESHOLD, TRAINING_WINDOW_DAYS
    TRAINING_THRESHOLD = int(update.training_threshold)
    TRAINING_WINDOW_DAYS = int(update.training_window_days)
    return {
        "training_threshold": TRAINING_THRESHOLD,
        "training_window_days": TRAINING_WINDOW_DAYS,
    }

async def trigger_training_job(request: TrainingTriggerRequest) -> int:
    """Trigger a training job and return training ID"""
    logger.info(f"Triggering {request.training_mode} training: {request.trigger_type}")

    total_predictions = ct_store.count_predictions()
    eligible_feedback_count = await count_eligible_feedback()
    all_predictions = ct_store.fetch_predictions()
    eligible_feedback_items = ct_store.fetch_feedback(eligible_only=True, days=TRAINING_WINDOW_DAYS)

    # Snapshot the exact feedback-linked rows used as training input at trigger time.
    dataset_rows = build_training_dataset_snapshot(
        eligible_feedback_items,
        all_predictions,
    )
    dataset_row_count = len(dataset_rows)

    training_id = ct_store.create_training_job(
        status="running",
        total_predictions=total_predictions,
        eligible_feedback_count=eligible_feedback_count,
        previous_model_version="v2.0",
        trigger_type=request.trigger_type,
        training_mode=request.training_mode,
        eks_node_group=request.eks_node_group,
        dataset_row_count=dataset_row_count,
    )

    # In-memory snapshot only for dataset download UI.
    training_datasets[training_id] = dataset_rows
    training_jobs[training_id] = {
        "training_id": training_id,
        "status": "running",
        "start_time": datetime.now(),
        "end_time": None,
        "total_predictions": total_predictions,
        "eligible_feedback_count": eligible_feedback_count,
        "previous_model_version": "v2.0",
        "new_model_version": None,
        "training_accuracy": None,
        "validation_accuracy": None,
        "f1_score": None,
        "is_deployed": False,
        "error_message": None,
        "trigger_type": request.trigger_type,
        "training_mode": request.training_mode,
        "dataset_row_count": dataset_row_count,
        "small_sample_warning": None,
        "metrics_note": None,
    }

    # Start actual training (in-process for local, K8s job for eks_hybrid).
    logger.info(f"Starting actual training for job {training_id}")
    task = asyncio.create_task(execute_actual_training(training_id, request.training_mode))
    logger.info(f"Training task created: {task}")
    _refresh_training_metrics()

    return training_id

async def execute_actual_training(training_id: int, training_mode: str):
    """Execute actual ML training"""
    try:
        effective_mode = _effective_training_mode(training_mode)
        logger.info(f"Starting actual training for job {training_id} (requested={training_mode}, effective={effective_mode})")
        current_feedback = ct_store.fetch_feedback(eligible_only=True, days=TRAINING_WINDOW_DAYS)
        current_predictions = ct_store.fetch_predictions()
        
        if effective_mode == "eks_hybrid":
            # Tùy chọn: node group + Job riêng (cần ALLOW_EKS_HYBRID_TRAINING=true và hạ tầng đầy đủ)
            result = await run_eks_training(training_id)
        else:
            # Cùng pipeline sklearn/MLflow như chạy local — chạy trong pod FastAPI (EKS hoặc docker-compose)
            result = execute_training(current_feedback, current_predictions, "local")

        if result.get("status") == "completed":
            tm = result.get("training_metrics") or {}
            model_version = result.get("model_version")

            # Persist completion status for UI & restart safety.
            ct_store.update_training_job_completed(
                training_id,
                model_version=model_version,
                training_metrics=tm,
                is_deployed=True,
            )

            if training_id in training_jobs:
                # Update in-memory snapshot for immediate UI usage.
                training_jobs[training_id].update({
                    "status": "completed",
                    "end_time": datetime.now(),
                    "new_model_version": model_version,
                    "training_accuracy": tm.get("training_accuracy"),
                    "validation_accuracy": tm.get("validation_accuracy"),
                    "f1_score": tm.get("validation_f1"),
                    "is_deployed": True,
                    "small_sample_warning": tm.get("small_sample_warning"),
                    "metrics_note": tm.get("metrics_note"),
                })

            # P1: Promote new model to active inference model (works only if model artifacts are shared)
            try:
                from ai_service.main import set_active_model_and_reload
                if model_version:
                    set_active_model_and_reload(model_version)
                    logger.info("Active model updated to %s after training job %s", model_version, training_id)
            except Exception as e:
                logger.warning("Failed to set active model after training: %s", e)

            # Reset feedback rows after successful training to prevent immediate retraining on same batch.
            cleared = ct_store.clear_feedback()
            logger.info("Feedback data reset after training completion. Cleared count: %s", cleared)

            logger.info(f"Training {training_id} completed successfully!")
            _refresh_training_metrics()
        else:
            error_msg = result.get("error", "Unknown error")
            ct_store.update_training_job_failed(training_id, error_message=error_msg)
            if training_id in training_jobs:
                training_jobs[training_id].update({
                    "status": "failed",
                    "error_message": error_msg,
                })
            _refresh_training_metrics()
            
    except Exception as e:
        logger.error(f"Training execution failed for {training_id}: {e}")
        try:
            ct_store.update_training_job_failed(training_id, error_message=str(e))
        except Exception:
            pass
        if training_id in training_jobs:
            training_jobs[training_id].update({
                "status": "failed",
                "error_message": str(e),
            })
        _refresh_training_metrics()

async def run_eks_training(training_id: int) -> Dict[str, Any]:
    """Run training on EKS with node group scaling"""
    try:
        logger.info(f"Scaling up EKS training node group")
        
        # Initialize boto3 and kubernetes clients
        eks_client = boto3.client('eks', region_name=EKS_REGION)
        
        # Scale up node group
        response = eks_client.update_nodegroup_config(
            clusterName=EKS_CLUSTER_NAME,
            nodegroupName=TRAINING_NODE_GROUP,
            scalingConfig={
                'minSize': 1,
                'desiredSize': 1,
                'maxSize': 1
            }
        )
        
        logger.info(f"Node group scaling initiated: {response['update']['status']}")
        
        # Wait for node group to be ready
        await wait_for_node_group_ready(eks_client, TRAINING_NODE_GROUP)
        
        # Run training job on EKS
        training_result = await run_training_job_on_eks(training_id)
        
        # Scale down node group
        logger.info(f"Scaling down EKS training node group")
        eks_client.update_nodegroup_config(
            clusterName=EKS_CLUSTER_NAME,
            nodegroupName=TRAINING_NODE_GROUP,
            scalingConfig={
                'minSize': 0,
                'desiredSize': 0,
                'maxSize': 1
            }
        )
        
        return training_result
        
    except Exception as e:
        logger.error(f"EKS training failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def wait_for_node_group_ready(eks_client, nodegroup_name: str, timeout: int = 600):
    """Wait for node group to become ready"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = eks_client.describe_nodegroup(
            clusterName=EKS_CLUSTER_NAME,
            nodegroupName=nodegroup_name
        )
        
        status = response['nodegroup']['status']
        if status == 'ACTIVE':
            logger.info(f"Node group {nodegroup_name} is ready")
            return True
        
        logger.info(f"Node group status: {status}, waiting...")
        await asyncio.sleep(30)
    
    raise TimeoutError(f"Node group {nodegroup_name} did not become ready in {timeout} seconds")

async def run_training_job_on_eks(training_id: int) -> Dict[str, Any]:
    """Run training job as Kubernetes pod on EKS"""
    if not KUBERNETES_AVAILABLE:
        return {'status': 'failed', 'error': 'Kubernetes client not available. Install kubernetes package.'}
    
    try:
        # Load kubernetes config
        config.load_kube_config()
        k8s_client = client.CoreV1Api()
        
        # Create training job manifest
        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"vet-ai-training-{training_id}",
                "namespace": "default"
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "training",
                            "image": "vet-ai/training:latest",
                            "env": [
                                {"name": "TRAINING_ID", "value": str(training_id)},
                                {"name": "DATA_SOURCE", "value": "postgres"},
                                {"name": "DATABASE_URL", "value": os.getenv("DATABASE_URL", "")},
                                {"name": "TRAINING_WINDOW_DAYS", "value": str(TRAINING_WINDOW_DAYS)},
                                {"name": "MLFLOW_TRACKING_URI", "value": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")},
                                {"name": "TRAINING_MIN_UNIQUE_CLASSES", "value": os.getenv("TRAINING_MIN_UNIQUE_CLASSES", "2")},
                                {"name": "FINETUNE_PREVIOUS_MODEL", "value": os.getenv("FINETUNE_PREVIOUS_MODEL", "true")},
                                {"name": "FINETUNE_ADD_TREES", "value": os.getenv("FINETUNE_ADD_TREES", "20")},
                                {"name": "FEEDBACK_DELTA_MAX", "value": os.getenv("FEEDBACK_DELTA_MAX", "0.25")},
                                {"name": "FEEDBACK_ACCEPT_STRENGTH", "value": os.getenv("FEEDBACK_ACCEPT_STRENGTH", "0.8")},
                                {"name": "FEEDBACK_REJECT_STRENGTH", "value": os.getenv("FEEDBACK_REJECT_STRENGTH", "0.8")},
                                {"name": "FEEDBACK_POSITIVE_MAX_WEIGHT", "value": os.getenv("FEEDBACK_POSITIVE_MAX_WEIGHT", "1.5")},
                                {"name": "FEEDBACK_NEGATIVE_MIN_WEIGHT", "value": os.getenv("FEEDBACK_NEGATIVE_MIN_WEIGHT", "0.7")},
                                {"name": "REGRESSION_GATE_ENABLED", "value": os.getenv("REGRESSION_GATE_ENABLED", "true")},
                                {"name": "REGRESSION_GATE_MIN_FEEDBACK_SAMPLES", "value": os.getenv("REGRESSION_GATE_MIN_FEEDBACK_SAMPLES", "20")},
                                {"name": "REGRESSION_TOLERANCE_F1", "value": os.getenv("REGRESSION_TOLERANCE_F1", "0.02")},
                                {"name": "REGRESSION_HIGH_BASE_F1_THRESHOLD", "value": os.getenv("REGRESSION_HIGH_BASE_F1_THRESHOLD", "0.99")},
                                {"name": "REGRESSION_TOLERANCE_F1_HIGH_BASE", "value": os.getenv("REGRESSION_TOLERANCE_F1_HIGH_BASE", "0.25")},
                                {"name": "FEEDBACK_IMPROVEMENT_GATE_ENABLED", "value": os.getenv("FEEDBACK_IMPROVEMENT_GATE_ENABLED", "true")},
                                {"name": "FEEDBACK_GATE_TOLERANCE", "value": os.getenv("FEEDBACK_GATE_TOLERANCE", "0.03")},
                                {"name": "METRICS_RELIABLE_MIN_SAMPLES", "value": os.getenv("METRICS_RELIABLE_MIN_SAMPLES", "80")},
                                {"name": "MODEL_ROOT_DIR", "value": os.getenv("MODEL_ROOT_DIR", "/app/ai_service/models")},
                                {"name": "MODEL_DIR", "value": os.getenv("MODEL_DIR", "/app/models/v2")},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "1000m",
                                    "memory": "4Gi"
                                },
                                "limits": {
                                    "cpu": "2000m",
                                    "memory": "8Gi"
                                }
                            }
                        }],
                        "restartPolicy": "OnFailure",
                        "nodeSelector": {
                            "node-group": "training"
                        }
                    }
                }
            }
        }
        
        # Submit job
        batch_client = client.BatchV1Api()
        job = batch_client.create_namespaced_job(namespace="default", body=job_manifest)
        
        logger.info(f"Training job submitted: {job.metadata.name}")
        
        # Wait for job completion
        await wait_for_job_completion(batch_client, job.metadata.name, "default")
        
        # Get job results
        job_status = batch_client.read_namespaced_job_status(job.metadata.name, "default")
        
        if job_status.status.succeeded:
            # Get training results from pod logs
            pod_name = await get_training_pod_name(k8s_client, job.metadata.name)
            logs = k8s_client.read_namespaced_pod_log(pod_name, "default")
            
            # Parse results from logs
            result = parse_training_results(logs)
            return result
        else:
            return {'status': 'failed', 'error': 'Training job failed on EKS'}
            
    except Exception as e:
        logger.error(f"Failed to run training job on EKS: {e}")
        return {'status': 'failed', 'error': str(e)}

async def wait_for_job_completion(batch_client, job_name: str, namespace: str, timeout: int = 3600):
    """Wait for Kubernetes job to complete"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        job = batch_client.read_namespaced_job_status(job_name, namespace)
        
        if job.status.succeeded or job.status.failed:
            logger.info(f"Job {job_name} completed with status: {job.status}")
            return
        
        logger.info(f"Job {job_name} still running...")
        await asyncio.sleep(60)
    
    raise TimeoutError(f"Job {job_name} did not complete in {timeout} seconds")

async def get_training_pod_name(k8s_client, job_name: str) -> str:
    """Get the pod name for a training job"""
    pods = k8s_client.list_namespaced_pod(
        namespace="default",
        label_selector=f"job-name={job_name}"
    )
    
    if pods.items:
        return pods.items[0].metadata.name
    else:
        raise ValueError(f"No pod found for job {job_name}")

def parse_training_results(logs: str) -> Dict[str, Any]:
    """Parse training results from pod logs"""
    try:
        # Look for JSON result in logs
        for line in logs.split('\n'):
            if line.startswith('TRAINING_RESULT:'):
                result_json = line.replace('TRAINING_RESULT:', '')
                return json.loads(result_json)
        
        return {'status': 'failed', 'error': 'No training result found in logs'}
    except Exception as e:
        return {'status': 'failed', 'error': f'Failed to parse results: {e}'}

async def get_training_status(training_id: int) -> Optional[TrainingStatus]:
    """Get training status by ID"""
    # Check if training job exists
    if training_id in training_jobs:
        job_data = training_jobs[training_id]
        return TrainingStatus(**job_data)
    
    # Try persisted DB record
    job_data = ct_store.get_training_job(training_id)
    if job_data:
        return TrainingStatus(
            training_id=job_data["training_id"],
            status=job_data["status"],
            start_time=job_data.get("start_time"),
            end_time=job_data.get("end_time"),
            total_predictions=job_data.get("total_predictions", 0),
            eligible_feedback_count=job_data.get("eligible_feedback_count", 0),
            previous_model_version=job_data.get("previous_model_version"),
            new_model_version=job_data.get("new_model_version"),
            training_accuracy=job_data.get("training_accuracy"),
            validation_accuracy=job_data.get("validation_accuracy"),
            f1_score=job_data.get("f1_score"),
            is_deployed=bool(job_data.get("is_deployed", False)),
            error_message=job_data.get("error_message"),
            small_sample_warning=job_data.get("small_sample_warning"),
            metrics_note=job_data.get("metrics_note"),
        )

    # Return default pending status for unknown jobs
    return TrainingStatus(
        training_id=training_id,
        status="pending",
        start_time=None,
        end_time=None,
        total_predictions=ct_store.count_predictions(),
        eligible_feedback_count=ct_store.count_feedback(eligible_only=True, days=TRAINING_WINDOW_DAYS),
        previous_model_version="v2.0",
        new_model_version=None,
        training_accuracy=None,
        validation_accuracy=None,
        f1_score=None,
        is_deployed=False,
        error_message="Training job not found",
        small_sample_warning=None,
        metrics_note=None,
    )

# API Endpoints
@router.post("/predictions/log", status_code=201)
async def log_prediction_endpoint(prediction: PredictionLog):
    """Log a prediction for continuous training"""
    try:
        prediction_id = await log_prediction(prediction)
        return {"predictionId": prediction_id, "status": "logged"}
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to log prediction")

@router.post("/feedback", status_code=201)
async def save_feedback_endpoint(feedback: DoctorFeedback):
    """Save doctor feedback for a prediction"""
    try:
        success = await save_feedback(feedback)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to save feedback")
        
        # Check if we should trigger retraining
        eligible_count = await count_eligible_feedback()
        should_trigger = eligible_count >= TRAINING_THRESHOLD
        
        return {
            "status": "saved",
            "eligible_feedback_count": eligible_count,
            "training_threshold": TRAINING_THRESHOLD,
            "should_trigger_training": should_trigger
        }
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@router.get("/training/status")
async def get_training_status_endpoint(training_id: int):
    """Get training job status"""
    try:
        status = await get_training_status(training_id)
        if not status:
            raise HTTPException(status_code=404, detail="Training job not found")
        return status
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")

@router.post("/training/trigger", status_code=201)
async def trigger_training_endpoint(
    request: TrainingTriggerRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin)
):
    """Trigger a training job"""
    try:
        # Check threshold unless forced
        if not request.force:
            eligible_count = await count_eligible_feedback()
            if eligible_count < TRAINING_THRESHOLD:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient data: {eligible_count} < {TRAINING_THRESHOLD}"
                )
        else:
            # Even when forced, training requires at least one eligible feedback+prediction pair
            eligible_count = await count_eligible_feedback()
            if eligible_count <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="No eligible feedback data to train on. Collect feedback first, then trigger training."
                )
        
        training_id = await trigger_training_job(request)
        
        # Add background task to monitor training
        background_tasks.add_task(
            monitor_training_progress,
            training_id
        )
        
        return {
            "training_id": training_id,
            "status": "triggered",
            "trigger_type": request.trigger_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger training")

@router.get("/training/eligibility")
async def check_training_eligibility():
    """Check if system is eligible for retraining"""
    try:
        eligible_count = await count_eligible_feedback()
        is_eligible = eligible_count >= TRAINING_THRESHOLD
        
        return {
            "eligible_feedback_count": eligible_count,
            "training_threshold": TRAINING_THRESHOLD,
            "is_eligible_for_training": is_eligible,
            "training_window_days": TRAINING_WINDOW_DAYS,
            "next_check_date": datetime.now() + timedelta(days=1)
        }
    except Exception as e:
        logger.error(f"Failed to check eligibility: {e}")
        raise HTTPException(status_code=500, detail="Failed to check eligibility")

@router.get("/training/history")
async def get_training_history(limit: int = 10, offset: int = 0):
    """Get training job history from actual training jobs"""
    try:
        rows = ct_store.list_training_jobs(limit=limit, offset=offset)
        training_runs = []
        for job in rows:
            training_runs.append({
                "training_id": job.get("training_id"),
                "status": job.get("status"),
                "trigger_type": job.get("trigger_type"),
                "created_at": job.get("start_time"),
                "training_accuracy": job.get("training_accuracy"),
                "validation_accuracy": job.get("validation_accuracy"),
                "f1_score": job.get("f1_score"),
                "is_deployed": job.get("is_deployed", False),
                "error_message": job.get("error_message"),
                "training_mode": job.get("training_mode"),
                "previous_model_version": job.get("previous_model_version"),
                "new_model_version": job.get("new_model_version"),
                "dataset_row_count": job.get("dataset_row_count", 0),
            })

        return {
            "total_count": ct_store.count_training_jobs_total(),
            "limit": limit,
            "offset": offset,
            "training_runs": training_runs,
        }
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training history")


@router.get("/training/dataset")
async def get_training_dataset_summary(training_id: int, _: bool = Depends(verify_admin)):
    """Get dataset summary for a specific training job."""
    if training_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    rows = training_datasets.get(training_id, [])
    return {
        "training_id": training_id,
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
    }


@router.get("/training/dataset/download")
async def download_training_dataset(training_id: int, _: bool = Depends(verify_admin)):
    """Download raw feedback-linked dataset used for the training trigger."""
    if training_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    rows = training_datasets.get(training_id, [])
    if not rows:
        raise HTTPException(status_code=404, detail="No dataset captured for this training run")

    headers = list(rows[0].keys())
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    buf.seek(0)

    filename = f"training_{training_id}_dataset.csv"
    response_headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=response_headers)

async def get_training_overview():
    """Get training system overview"""
    try:
        counts = ct_store.count_training_jobs_by_status()
        active_jobs = counts.get("running", 0)
        completed_jobs = counts.get("completed", 0)
        failed_jobs = counts.get("failed", 0)
        recent_rows = ct_store.list_training_jobs(limit=5, offset=0)
        
        total_predictions = ct_store.count_predictions()
        total_feedback = ct_store.count_feedback()
        eligible_feedback = ct_store.count_feedback(eligible_only=True, days=TRAINING_WINDOW_DAYS)

        return {
            "system_status": "active",
            "current_feedback_count": total_feedback,
            "training_threshold": TRAINING_THRESHOLD,
            "is_eligible_for_training": eligible_feedback >= TRAINING_THRESHOLD,
            "training_jobs": {
                "active": active_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "total": active_jobs + completed_jobs + failed_jobs + counts.get("pending", 0),
            },
            "recent_training_jobs": recent_rows,
            "data_collection": {
                "total_predictions": total_predictions,
                "total_feedback": total_feedback,
                "eligible_feedback": eligible_feedback
            }
        }
    except Exception as e:
        logger.error(f"Failed to get training overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training overview")

@router.post("/reset-all")
async def reset_all_data():
    """Reset all predictions and feedback data for clean testing"""
    try:
        cleared = ct_store.clear_all()
        prediction_count = int(cleared.get("predictions", 0))
        feedback_count = int(cleared.get("feedback", 0))
        
        # Reset prediction counter in main module
        import sys
        if 'ai_service.main' in sys.modules:
            sys.modules['ai_service.main']._prediction_counter = 0
        
        logger.info(f"Reset all data: {prediction_count} predictions, {feedback_count} feedback entries")
        
        return {
            "status": "success",
            "message": f"Reset {prediction_count} predictions and {feedback_count} feedback entries",
            "current_predictions": 0,
            "current_feedback": 0
        }
    except Exception as e:
        logger.error(f"Failed to reset all data: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset all data")

@router.get("/predictions/all")
async def get_all_predictions():
    """Get all prediction logs for debugging"""
    try:
        predictions = ct_store.fetch_predictions()
        return {
            "status": "success",
            "total_predictions": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions")

@router.get("/predictions/count")
async def get_predictions_count():
    """Get total number of predictions"""
    try:
        return {
            "status": "success",
            "total_predictions": ct_store.count_predictions()
        }
    except Exception as e:
        logger.error(f"Failed to get prediction count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prediction count")

@router.get("/feedback/count")
async def get_feedback_count():
    """Get total number of feedback entries"""
    try:
        return {
            "status": "success",
            "total_feedback": ct_store.count_feedback()
        }
    except Exception as e:
        logger.error(f"Failed to get feedback count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback count")

@router.post("/feedback/reset")
async def reset_feedback_data():
    """Manually reset feedback data"""
    try:
        feedback_count = ct_store.clear_feedback()
        logger.info(f"Manual feedback reset. Cleared {feedback_count} entries.")
        
        return {
            "status": "success",
            "message": f"Reset {feedback_count} feedback entries",
            "current_count": 0
        }
    except Exception as e:
        logger.error(f"Failed to reset feedback data: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset feedback data")

async def monitor_training_progress(training_id: int):
    """Background task to monitor training progress"""
    logger.info(f"Monitoring training progress for job {training_id}")
    
    # Monitor training job status
    max_wait_time = 300  # 5 minutes
    start_time = datetime.now()
    
    while (datetime.now() - start_time).seconds < max_wait_time:
        try:
            job = ct_store.get_training_job(training_id)
            if job and str(job.get("status", "")).lower() in ["completed", "failed"]:
                logger.info(
                    "Training job %s finished with status: %s",
                    training_id,
                    job.get("status"),
                )
                break
        except Exception as e:
            logger.warning("monitor_training_progress: failed to read job from DB: %s", e)
        
        await asyncio.sleep(5)  # Check every 5 seconds
    
    # Timeout handling
    try:
        job = ct_store.get_training_job(training_id)
        if job and str(job.get("status", "")).lower() == "running":
            ct_store.update_training_job_failed(
                training_id,
                error_message="Training timeout after 5 minutes",
            )
            if training_id in training_jobs:
                training_jobs[training_id]["status"] = "failed"
                training_jobs[training_id]["error_message"] = "Training timeout after 5 minutes"
            logger.warning("Training job %s timed out", training_id)
    except Exception as e:
        logger.warning("monitor_training_progress: failed timeout update: %s", e)
