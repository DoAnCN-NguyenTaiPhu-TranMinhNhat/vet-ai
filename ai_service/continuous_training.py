"""
Continuous Training endpoints for FastAPI service
Handles prediction logging and training trigger logic
"""

from datetime import datetime, timedelta
from io import StringIO
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from fastapi import HTTPException, BackgroundTasks, APIRouter, Depends, Query, File, Form, UploadFile
from fastapi.responses import StreamingResponse, Response
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
from psycopg2.extras import RealDictCursor, Json, register_uuid

# psycopg2 does not adapt uuid.UUID for parameters unless registered (else: can't adapt type 'UUID')
register_uuid()

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
    from ai_service.training_engine import (
        CSV_BOOTSTRAP_TEMPLATE_HEADER,
        execute_training,
        parse_bootstrap_csv,
    )
except ImportError:
    from training_engine import (  # type: ignore
        CSV_BOOTSTRAP_TEMPLATE_HEADER,
        execute_training,
        parse_bootstrap_csv,
    )

from ai_service.clinic_scope import normalize_clinic_key

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
            id UUID PRIMARY KEY,
            visit_id TEXT NULL,
            pet_id TEXT NOT NULL,
            prediction_input JSONB NOT NULL,
            prediction_output JSONB NOT NULL,
            model_version TEXT NOT NULL,
            confidence_score DOUBLE PRECISION NOT NULL,
            top_k_predictions JSONB NOT NULL,
            veterinarian_id TEXT NULL,
            clinic_id TEXT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ai_feedback (
            id BIGSERIAL PRIMARY KEY,
            prediction_id UUID NOT NULL REFERENCES ai_prediction_logs(id) ON DELETE CASCADE,
            final_diagnosis TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            ai_diagnosis TEXT NULL,
            confidence_rating INTEGER NULL,
            comments TEXT NULL,
            veterinarian_id TEXT NULL,
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
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'ai_prediction_logs'
                    ) AS t
                    """
                )
                pred_table_exists = bool(cur.fetchone()["t"])
                if pred_table_exists:
                    cur.execute(
                        """
                        SELECT data_type FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'ai_prediction_logs'
                          AND column_name = 'id'
                        """
                    )
                    id_row = cur.fetchone()
                    id_type = (id_row or {}).get("data_type") if id_row else None
                    if id_type in ("bigint", "integer", "smallint"):
                        logger.warning(
                            "ai_prediction_logs.id is %s (legacy); dropping ai_feedback + "
                            "ai_prediction_logs and recreating with UUID primary keys. "
                            "Continuous-training log rows are reset.",
                            id_type,
                        )
                        cur.execute("DROP TABLE IF EXISTS ai_feedback CASCADE")
                        cur.execute("DROP TABLE IF EXISTS ai_prediction_logs CASCADE")

                cur.execute(ddl)
                cur.execute(
                    "ALTER TABLE ai_training_jobs ADD COLUMN IF NOT EXISTS clinic_id BIGINT NULL"
                )
                cur.execute(
                    """
                    SELECT data_type FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'ai_training_jobs'
                      AND column_name = 'clinic_id'
                    """
                )
                cj = cur.fetchone()
                cj_type = (cj or {}).get("data_type") if cj else None
                if cj_type in ("bigint", "integer", "smallint"):
                    logger.warning(
                        "Widening ai_training_jobs.clinic_id from %s to TEXT for UUID clinic ids.",
                        cj_type,
                    )
                    cur.execute(
                        "ALTER TABLE ai_training_jobs ALTER COLUMN clinic_id TYPE TEXT USING clinic_id::TEXT"
                    )

                # Older DBs may still have INTEGER visit_id after IF NOT EXISTS skipped CREATE.
                cur.execute(
                    """
                    SELECT data_type FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'ai_prediction_logs'
                      AND column_name = 'visit_id'
                    """
                )
                vid = cur.fetchone()
                vtype = (vid or {}).get("data_type") if vid else None
                if vtype in ("integer", "bigint", "smallint"):
                    logger.info(
                        "Widening ai_prediction_logs.visit_id from %s to TEXT (UUID visit ids).",
                        vtype,
                    )
                    cur.execute(
                        "ALTER TABLE ai_prediction_logs ALTER COLUMN visit_id TYPE TEXT USING (visit_id::TEXT)"
                    )

                # Per-clinic policy: where feedback counts for retraining (global pool vs clinic-only pool).
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clinic_training_policy (
                        clinic_id TEXT PRIMARY KEY,
                        feedback_pool TEXT NOT NULL DEFAULT 'CLINIC_ONLY',
                        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        CONSTRAINT clinic_training_policy_pool_chk CHECK (feedback_pool IN ('GLOBAL', 'CLINIC_ONLY'))
                    )
                    """
                )
                cur.execute(
                    "ALTER TABLE ai_feedback ADD COLUMN IF NOT EXISTS training_pool TEXT DEFAULT 'CLINIC_ONLY'"
                )
                cur.execute(
                    "UPDATE ai_feedback SET training_pool = 'CLINIC_ONLY' WHERE training_pool IS NULL"
                )
                cur.execute(
                    "ALTER TABLE ai_feedback ALTER COLUMN training_pool SET DEFAULT 'CLINIC_ONLY'"
                )
                cur.execute("ALTER TABLE ai_feedback ALTER COLUMN training_pool SET NOT NULL")
                cur.execute(
                    """
                    DO $$ BEGIN
                        ALTER TABLE ai_feedback ADD CONSTRAINT ai_feedback_training_pool_chk
                        CHECK (training_pool IN ('GLOBAL', 'CLINIC_ONLY'));
                    EXCEPTION WHEN duplicate_object THEN NULL;
                    END $$;
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ai_feedback_training_pool ON ai_feedback(training_pool)"
                )
            conn.commit()
        self._initialized = True

    def insert_prediction(self, payload: Dict[str, Any]) -> uuid.UUID:
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
            -- GenAI calls /predictions/log after /predict; if Java omits clinicId, do not wipe clinic_id set by /predict.
            clinic_id = COALESCE(EXCLUDED.clinic_id, ai_prediction_logs.clinic_id)
        """
        data = dict(payload)
        if isinstance(data.get("id"), uuid.UUID):
            data["id"] = str(data["id"])
        data["prediction_input"] = Json(payload.get("prediction_input") or {})
        data["prediction_output"] = Json(payload.get("prediction_output") or {})
        data["top_k_predictions"] = Json(payload.get("top_k_predictions") or [])
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, data)
            conn.commit()
        raw_id = payload["id"]
        if isinstance(raw_id, uuid.UUID):
            return raw_id
        return uuid.UUID(str(raw_id))

    def insert_feedback(self, payload: Dict[str, Any]) -> bool:
        self._ensure_schema()
        sql = """
        INSERT INTO ai_feedback
            (prediction_id, final_diagnosis, is_correct, ai_diagnosis, confidence_rating, comments, veterinarian_id, is_training_eligible, data_quality_score, training_pool)
        VALUES
            (%(prediction_id)s, %(final_diagnosis)s, %(is_correct)s, %(ai_diagnosis)s, %(confidence_rating)s, %(comments)s, %(veterinarian_id)s, %(is_training_eligible)s, %(data_quality_score)s, %(training_pool)s)
        """
        fb = dict(payload)
        if isinstance(fb.get("prediction_id"), uuid.UUID):
            fb["prediction_id"] = str(fb["prediction_id"])
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, fb)
            conn.commit()
        return True

    def get_clinic_id_for_prediction(self, prediction_id: Any) -> Optional[str]:
        """Return clinic_id stored on the prediction log row (UUID string), if any."""
        self._ensure_schema()
        pid = str(prediction_id).strip()
        if not pid:
            return None
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT clinic_id FROM ai_prediction_logs WHERE id = %s",
                    (pid,),
                )
                row = cur.fetchone()
        if not row or row.get("clinic_id") is None:
            return None
        s = str(row["clinic_id"]).strip()
        return s or None

    def get_clinic_feedback_pool(self, clinic_id: str) -> str:
        """Policy: GLOBAL = feedback counts only toward system-wide retraining; CLINIC_ONLY = only toward this clinic."""
        self._ensure_schema()
        cid = str(clinic_id).strip()
        if not cid:
            return "CLINIC_ONLY"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT feedback_pool FROM clinic_training_policy WHERE clinic_id = %s",
                    (cid,),
                )
                row = cur.fetchone()
        if not row:
            return "CLINIC_ONLY"
        p = str(row.get("feedback_pool") or "").upper()
        return p if p in ("GLOBAL", "CLINIC_ONLY") else "CLINIC_ONLY"

    def set_clinic_feedback_pool(self, clinic_id: str, feedback_pool: str) -> None:
        self._ensure_schema()
        cid = str(clinic_id).strip()
        fp = str(feedback_pool).upper().strip()
        if fp not in ("GLOBAL", "CLINIC_ONLY"):
            raise ValueError("feedback_pool must be GLOBAL or CLINIC_ONLY")
        if not cid:
            raise ValueError("clinic_id is required")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO clinic_training_policy (clinic_id, feedback_pool, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (clinic_id) DO UPDATE SET
                        feedback_pool = EXCLUDED.feedback_pool,
                        updated_at = NOW()
                    """,
                    (cid, fp),
                )
            conn.commit()

    def count_predictions(self, clinic_id: Optional[str] = None) -> int:
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                if clinic_id is None:
                    cur.execute("SELECT COUNT(*)::BIGINT AS c FROM ai_prediction_logs")
                else:
                    cur.execute(
                        "SELECT COUNT(*)::BIGINT AS c FROM ai_prediction_logs WHERE clinic_id = %s",
                        (clinic_id,),
                    )
                return int(cur.fetchone()["c"])

    def count_feedback(
        self,
        eligible_only: bool = False,
        days: Optional[int] = None,
        clinic_id: Optional[str] = None,
    ) -> int:
        """Global scope (clinic_id None): only rows with training_pool=GLOBAL. Per-clinic: CLINIC_ONLY + matching prediction.clinic_id."""
        self._ensure_schema()
        cond = []
        params: List[Any] = []
        if clinic_id is None:
            from_clause = "ai_feedback f"
            cond.append("f.training_pool = 'GLOBAL'")
        else:
            from_clause = "ai_feedback f INNER JOIN ai_prediction_logs p ON f.prediction_id = p.id"
            cond.append("p.clinic_id = %s")
            params.append(clinic_id)
            cond.append("f.training_pool = 'CLINIC_ONLY'")
        if eligible_only:
            cond.append("f.is_training_eligible = TRUE")
        if days is not None:
            cond.append("f.created_at >= NOW() - (%s || ' days')::INTERVAL")
            params.append(int(days))
        where = "WHERE " + " AND ".join(cond)
        sql = f"SELECT COUNT(*)::BIGINT AS c FROM {from_clause} {where}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return int(cur.fetchone()["c"])

    def fetch_predictions(self, clinic_id: Optional[str] = None) -> List[Dict[str, Any]]:
        self._ensure_schema()
        base = """
        SELECT id, visit_id, pet_id, prediction_input, prediction_output, model_version, confidence_score,
               top_k_predictions, veterinarian_id, clinic_id, created_at
        FROM ai_prediction_logs
        """
        if clinic_id is None:
            sql = base + " ORDER BY created_at ASC"
            params: List[Any] = []
        else:
            sql = base + " WHERE clinic_id = %s ORDER BY created_at ASC"
            params = [clinic_id]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_feedback(
        self, eligible_only: bool = False, days: Optional[int] = None, clinic_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self._ensure_schema()
        cond = []
        params: List[Any] = []
        if clinic_id is None:
            from_sql = "ai_feedback f"
            cond.append("f.training_pool = 'GLOBAL'")
        else:
            from_sql = "ai_feedback f INNER JOIN ai_prediction_logs p ON f.prediction_id = p.id"
            cond.append("p.clinic_id = %s")
            params.append(clinic_id)
            cond.append("f.training_pool = 'CLINIC_ONLY'")
        if eligible_only:
            cond.append("f.is_training_eligible = TRUE")
        if days is not None:
            cond.append("f.created_at >= NOW() - (%s || ' days')::INTERVAL")
            params.append(int(days))
        where = "WHERE " + " AND ".join(cond)
        select_cols = """f.prediction_id AS prediction_id, f.final_diagnosis, f.is_correct, f.ai_diagnosis,
               f.confidence_rating, f.comments, f.veterinarian_id, f.is_training_eligible, f.data_quality_score,
               f.created_at AS timestamp"""
        sql = f"""
        SELECT {select_cols}
        FROM {from_sql}
        {where}
        ORDER BY f.created_at ASC
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

    def clear_feedback(self, clinic_id: Optional[str] = None) -> int:
        """After global training: delete GLOBAL pool only. After clinic training: delete CLINIC_ONLY for that clinic."""
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                if clinic_id is None:
                    cur.execute("DELETE FROM ai_feedback WHERE training_pool = 'GLOBAL'")
                else:
                    cur.execute(
                        """
                        DELETE FROM ai_feedback f
                        USING ai_prediction_logs p
                        WHERE f.prediction_id = p.id AND p.clinic_id = %s AND f.training_pool = 'CLINIC_ONLY'
                        """,
                        (clinic_id,),
                    )
                n = cur.rowcount
            conn.commit()
        return int(n)

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
        clinic_id: Optional[str] = None,
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
            clinic_id,
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
            %(clinic_id)s,
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
            "clinic_id": str(clinic_id).strip() if clinic_id is not None else None,
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
            metrics_note,
            clinic_id
        FROM ai_training_jobs
        WHERE training_id = %(training_id)s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"training_id": int(training_id)})
                row = cur.fetchone()
        return dict(row) if row else None

    def list_training_jobs(
        self, *, limit: int = 10, offset: int = 0, clinic_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List training jobs (recent first). When clinic_id is set, only jobs for that clinic."""
        self._ensure_schema()
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        where = ""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if clinic_id is not None:
            where = " WHERE clinic_id = %(clinic_id)s"
            params["clinic_id"] = clinic_id
        sql = f"""
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
            metrics_note,
            clinic_id
        FROM ai_training_jobs
        {where}
        ORDER BY training_id DESC
        LIMIT %(limit)s OFFSET %(offset)s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
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

    def count_training_jobs_total(self, clinic_id: Optional[str] = None) -> int:
        """Count total training jobs (optionally for one clinic)."""
        self._ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                if clinic_id is None:
                    cur.execute("SELECT COUNT(*)::BIGINT AS c FROM ai_training_jobs")
                else:
                    cur.execute(
                        "SELECT COUNT(*)::BIGINT AS c FROM ai_training_jobs WHERE clinic_id = %s",
                        (clinic_id,),
                    )
                return int(cur.fetchone()["c"])


ct_store = CTPostgresStore(DATABASE_URL)

# On EKS, real training runs in the same Python process as locally (execute_training). eks_hybrid (separate Batch Job)
# is only enabled when ALLOW_EKS_HYBRID_TRAINING=true — requires correct image, node group, and RBAC; off by default.
def _effective_training_mode(requested: str) -> str:
    if requested != "eks_hybrid":
        return requested
    allow = os.getenv("ALLOW_EKS_HYBRID_TRAINING", "false").lower() in ("1", "true", "yes", "y")
    if not allow:
        logger.info(
            "training_mode=eks_hybrid -> using in-process training (same as local). "
            "Set ALLOW_EKS_HYBRID_TRAINING=true only after a separate training Job is deployed."
        )
        return "local"
    return "eks_hybrid"


# Pydantic models
class PredictionLog(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: uuid.UUID
    # Visits-service uses UUID strings; legacy payloads may send int.
    visit_id: Optional[str | int] = None
    pet_id: str
    prediction_input: Dict
    prediction_output: Dict
    model_version: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    top_k_predictions: List[Dict]
    veterinarian_id: Optional[str] = None
    clinic_id: Optional[str] = None

class DoctorFeedback(BaseModel):
    prediction_id: uuid.UUID
    final_diagnosis: str
    is_correct: bool
    # AI-suggested label (used to apply negative training signal on reject)
    ai_diagnosis: Optional[str] = None
    confidence_rating: Optional[int] = Field(ge=0, le=5, default=None)
    comments: Optional[str] = None
    veterinarian_id: Optional[str] = None
    is_training_eligible: bool = True
    data_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    # Optional override; otherwise derived from clinic_training_policy + prediction clinic_id.
    training_pool: Optional[str] = Field(default=None, pattern="^(GLOBAL|CLINIC_ONLY)$")

class TrainingTriggerRequest(BaseModel):
    trigger_type: str = Field(pattern="^(scheduled|manual|automatic)$")
    trigger_reason: Optional[str] = None
    force: bool = False  # Override threshold check
    # local = execute_training() in pod (default on dev and EKS). eks_hybrid only when ALLOW_EKS_HYBRID_TRAINING=true.
    training_mode: str = Field(default="local", pattern="^(local|eks_hybrid)$")
    eks_node_group: Optional[str] = None
    # Scopes Postgres feedback/predictions, model output dir, MLflow experiment, and active pin (UUID or legacy int).
    clinic_id: Optional[str] = None

    @field_validator("clinic_id", mode="before")
    @classmethod
    def _normalize_trigger_clinic(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

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


def resolve_training_pool_for_feedback(
    prediction_clinic_id: Optional[str],
    explicit: Optional[str],
) -> str:
    """
    Mutually exclusive pools: GLOBAL counts only toward system-wide retraining; CLINIC_ONLY toward one clinic.
    """
    if explicit in ("GLOBAL", "CLINIC_ONLY"):
        return explicit
    if not prediction_clinic_id or not str(prediction_clinic_id).strip():
        return "GLOBAL"
    return ct_store.get_clinic_feedback_pool(str(prediction_clinic_id).strip())


def _resolved_clinic_id_for_prediction_row(p: PredictionLog) -> Optional[str]:
    """Top-level clinic_id, else clinic_id / clinicId inside prediction_input (Java may only embed there)."""
    if p.clinic_id is not None:
        s = str(p.clinic_id).strip()
        if s:
            return s
    inp = p.prediction_input
    if isinstance(inp, dict):
        raw = inp.get("clinic_id") or inp.get("clinicId")
        if raw is not None:
            s = str(raw).strip()
            if s:
                return s
    return None


# Database functions
async def log_prediction(prediction: PredictionLog) -> uuid.UUID:
    """Log prediction to PostgreSQL storage."""
    logger.info(f"Logging prediction for visit {prediction.visit_id}, ID: {prediction.id}")

    visit_db = None if prediction.visit_id is None else str(prediction.visit_id)
    resolved_clinic = _resolved_clinic_id_for_prediction_row(prediction)
    prediction_data = {
        "id": prediction.id,
        "visit_id": visit_db,
        "pet_id": prediction.pet_id,
        "prediction_input": prediction.prediction_input,
        "prediction_output": prediction.prediction_output,
        "model_version": prediction.model_version,
        "confidence_score": prediction.confidence_score,
        "top_k_predictions": prediction.top_k_predictions,
        "veterinarian_id": prediction.veterinarian_id,
        "clinic_id": resolved_clinic,
    }
    ct_store.insert_prediction(prediction_data)
    _predictions_logged_total.inc()
    _refresh_training_metrics()

    logger.info(f"Prediction logged successfully. ID: {prediction.id}")
    return prediction.id

async def save_feedback(feedback: DoctorFeedback) -> bool:
    """Save doctor feedback to PostgreSQL storage."""
    logger.info(f"Saving feedback for prediction {feedback.prediction_id}")

    pred_clinic = ct_store.get_clinic_id_for_prediction(feedback.prediction_id)
    pool = resolve_training_pool_for_feedback(pred_clinic, feedback.training_pool)
    logger.info(
        "Feedback training_pool=%s (prediction clinic_id=%s)",
        pool,
        pred_clinic,
    )

    feedback_data_item = {
        "prediction_id": feedback.prediction_id,
        "final_diagnosis": feedback.final_diagnosis,
        "is_correct": feedback.is_correct,
        "ai_diagnosis": feedback.ai_diagnosis,
        "confidence_rating": feedback.confidence_rating,
        "comments": feedback.comments,
        "veterinarian_id": feedback.veterinarian_id,
        "is_training_eligible": feedback.is_training_eligible,
        "data_quality_score": feedback.data_quality_score,
        "training_pool": pool,
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

async def count_eligible_feedback(days: Optional[int] = None, clinic_id: Optional[str] = None) -> int:
    """Count eligible feedback in the last N days (optionally for one clinic's prediction logs)."""
    if days is None:
        days = TRAINING_WINDOW_DAYS
    ck = normalize_clinic_key(clinic_id)
    eligible_count = ct_store.count_feedback(eligible_only=True, days=days, clinic_id=ck)
    logger.info("Current eligible feedback count: %s (clinic_id=%s)", eligible_count, ck)
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
                pred_by_id[str(pid)] = p

    for f in feedback_items:
        if not isinstance(f, dict):
            continue
        pred = pred_by_id.get(str(f.get("prediction_id")))
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

    ck = normalize_clinic_key(request.clinic_id)
    total_predictions = ct_store.count_predictions(clinic_id=ck)
    eligible_feedback_count = await count_eligible_feedback(clinic_id=ck)
    all_predictions = ct_store.fetch_predictions(clinic_id=ck)
    eligible_feedback_items = ct_store.fetch_feedback(
        eligible_only=True, days=TRAINING_WINDOW_DAYS, clinic_id=ck
    )

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
        clinic_id=ck,
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
        "clinic_id": ck,
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
        clinic_key: Optional[str] = None
        try:
            row = ct_store.get_training_job(training_id)
            if row and row.get("clinic_id") is not None:
                clinic_key = normalize_clinic_key(row["clinic_id"])
        except Exception:
            pass
        if clinic_key is None and training_id in training_jobs:
            c = training_jobs[training_id].get("clinic_id")
            if c is not None:
                clinic_key = normalize_clinic_key(c)

        effective_mode = _effective_training_mode(training_mode)
        logger.info(
            "Starting actual training for job %s (requested=%s, effective=%s, clinic_id=%s)",
            training_id,
            training_mode,
            effective_mode,
            clinic_key,
        )
        current_feedback = ct_store.fetch_feedback(
            eligible_only=True, days=TRAINING_WINDOW_DAYS, clinic_id=clinic_key
        )
        current_predictions = ct_store.fetch_predictions(clinic_id=clinic_key)
        
        if effective_mode == "eks_hybrid":
            # Optional: dedicated node group + Job (needs ALLOW_EKS_HYBRID_TRAINING=true and full infra)
            result = await run_eks_training(training_id)
        else:
            # Same sklearn/MLflow pipeline as local — runs inside FastAPI pod (EKS or docker-compose)
            result = execute_training(
                current_feedback,
                current_predictions,
                "local",
                clinic_id=clinic_key,
                training_id=training_id,
            )

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

            # Promote new model: global default or per-clinic pin when clinic_id was set on the job.
            try:
                if model_version:
                    if clinic_key is not None:
                        from ai_service.model_registry import set_clinic_active_model
                        from ai_service.main import clear_artifact_cache

                        set_clinic_active_model(clinic_key, model_version)
                        clear_artifact_cache()
                        logger.info(
                            "Clinic %s active model updated to %s after training job %s",
                            clinic_key,
                            model_version,
                            training_id,
                        )
                    else:
                        from ai_service.main import set_active_model_and_reload

                        set_active_model_and_reload(model_version)
                        logger.info(
                            "Active model updated to %s after training job %s",
                            model_version,
                            training_id,
                        )
            except Exception as e:
                logger.warning("Failed to set active model after training: %s", e)

            # Reset feedback rows after successful training to prevent immediate retraining on same batch.
            cleared = ct_store.clear_feedback(clinic_id=clinic_key)
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


async def execute_bootstrap_training(
    training_id: int,
    feedback_rows: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    clinic_key: Optional[str],
    training_mode: str,
) -> None:
    """
    Run training from in-memory CSV-derived rows; do not clear Postgres feedback on success.
    """
    try:
        effective_mode = _effective_training_mode(training_mode)
        if effective_mode != "local":
            raise ValueError(
                "CSV bootstrap training runs in-process only (training_mode=local). "
                "eks_hybrid is not supported for uploaded files."
            )
        logger.info(
            "Bootstrap CSV training job %s (clinic_id=%s, rows=%s)",
            training_id,
            clinic_key,
            len(feedback_rows),
        )
        result = execute_training(
            feedback_rows,
            prediction_rows,
            "local",
            clinic_id=clinic_key,
            training_id=training_id,
        )

        if result.get("status") == "completed":
            tm = result.get("training_metrics") or {}
            model_version = result.get("model_version")

            ct_store.update_training_job_completed(
                training_id,
                model_version=model_version,
                training_metrics=tm,
                is_deployed=True,
            )

            if training_id in training_jobs:
                training_jobs[training_id].update(
                    {
                        "status": "completed",
                        "end_time": datetime.now(),
                        "new_model_version": model_version,
                        "training_accuracy": tm.get("training_accuracy"),
                        "validation_accuracy": tm.get("validation_accuracy"),
                        "f1_score": tm.get("validation_f1"),
                        "is_deployed": True,
                        "small_sample_warning": tm.get("small_sample_warning"),
                        "metrics_note": tm.get("metrics_note"),
                    }
                )

            try:
                if model_version:
                    if clinic_key is not None:
                        from ai_service.model_registry import set_clinic_active_model
                        from ai_service.main import clear_artifact_cache

                        set_clinic_active_model(clinic_key, model_version)
                        clear_artifact_cache()
                        logger.info(
                            "Clinic %s pinned to %s after bootstrap job %s",
                            clinic_key,
                            model_version,
                            training_id,
                        )
                    else:
                        from ai_service.main import set_active_model_and_reload

                        set_active_model_and_reload(model_version)
                        logger.info(
                            "Global active model set to %s after bootstrap job %s",
                            model_version,
                            training_id,
                        )
            except Exception as e:
                logger.warning("Failed to set active model after bootstrap training: %s", e)

            logger.info("Bootstrap training %s completed successfully (Postgres feedback unchanged).", training_id)
            _refresh_training_metrics()
        else:
            error_msg = result.get("error", "Unknown error")
            ct_store.update_training_job_failed(training_id, error_message=error_msg)
            if training_id in training_jobs:
                training_jobs[training_id].update(
                    {
                        "status": "failed",
                        "error_message": error_msg,
                    }
                )
            _refresh_training_metrics()

    except Exception as e:
        logger.error("Bootstrap training execution failed for %s: %s", training_id, e)
        try:
            ct_store.update_training_job_failed(training_id, error_message=str(e))
        except Exception:
            pass
        if training_id in training_jobs:
            training_jobs[training_id].update(
                {
                    "status": "failed",
                    "error_message": str(e),
                }
            )
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
        return {"predictionId": str(prediction_id), "status": "logged"}
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

        pred_clinic = ct_store.get_clinic_id_for_prediction(feedback.prediction_id)
        pool = resolve_training_pool_for_feedback(pred_clinic, feedback.training_pool)
        ck = normalize_clinic_key(pred_clinic)
        if ck is None:
            logger.warning(
                "Feedback for prediction_id=%s: ai_prediction_logs.clinic_id is NULL. "
                "Use global training scope; per-clinic counts need clinic_id on predictions.",
                feedback.prediction_id,
            )

        eligible_global = await count_eligible_feedback(clinic_id=None)
        eligible_clinic = await count_eligible_feedback(clinic_id=ck) if ck else 0
        if pool == "GLOBAL":
            eligible_for_pool = eligible_global
        else:
            eligible_for_pool = eligible_clinic
        should_trigger = eligible_for_pool >= TRAINING_THRESHOLD
        # Hint for auto-trigger clients: which scope this feedback advanced.
        auto_scope = "global" if pool == "GLOBAL" else "clinic"

        return {
            "status": "saved",
            "training_pool": pool,
            "auto_trigger_scope": auto_scope,
            "eligible_feedback_count": eligible_for_pool,
            "eligible_feedback_count_global": eligible_global,
            "eligible_feedback_count_clinic": eligible_clinic if ck else None,
            "training_threshold": TRAINING_THRESHOLD,
            "should_trigger_training": should_trigger,
            "clinic_id": ck,
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
        ck = normalize_clinic_key(request.clinic_id)
        # Check threshold unless forced
        if not request.force:
            eligible_count = await count_eligible_feedback(clinic_id=ck)
            if eligible_count < TRAINING_THRESHOLD:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient data: {eligible_count} < {TRAINING_THRESHOLD}"
                )
        else:
            # Even when forced, training requires at least one eligible feedback+prediction pair
            eligible_count = await count_eligible_feedback(clinic_id=ck)
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


@router.get(
    "/training/bootstrap-csv/template",
    summary="CSV template (header row) for bootstrap training",
)
async def bootstrap_csv_template_endpoint():
    """Public template — same columns as vet-ml golden dataset."""
    return Response(
        content=CSV_BOOTSTRAP_TEMPLATE_HEADER.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="vet_bootstrap_training_template.csv"',
        },
    )


@router.post(
    "/training/bootstrap-csv",
    status_code=201,
    summary="Cold-start train from uploaded CSV (admin)",
)
async def bootstrap_csv_training_endpoint(
    file: UploadFile = File(..., description="UTF-8 CSV; schema see /training/bootstrap-csv/template"),
    clinic_id: Optional[str] = Form(
        None,
        description="Omit or empty for global model; UUID string to pin model for one clinic after train.",
    ),
    training_mode: str = Form("local"),
    _: bool = Depends(verify_admin),
):
    """
    Parse CSV in memory (does not insert into ai_feedback / ai_prediction_logs).
    On success, promotes global or per-clinic active model like a normal job; Postgres feedback is not cleared.
    """
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            fb_rows, pred_rows = parse_bootstrap_csv(raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        ck = normalize_clinic_key(clinic_id)
        tm = (training_mode or "local").strip() or "local"

        dataset_rows = build_training_dataset_snapshot(fb_rows, pred_rows)
        training_id = ct_store.create_training_job(
            status="running",
            total_predictions=ct_store.count_predictions(clinic_id=ck),
            eligible_feedback_count=len(fb_rows),
            previous_model_version="v2.0",
            trigger_type="bootstrap_csv",
            training_mode=tm,
            eks_node_group=None,
            dataset_row_count=len(dataset_rows),
            clinic_id=ck,
        )

        training_datasets[training_id] = dataset_rows
        training_jobs[training_id] = {
            "training_id": training_id,
            "status": "running",
            "start_time": datetime.now(),
            "end_time": None,
            "total_predictions": ct_store.count_predictions(clinic_id=ck),
            "eligible_feedback_count": len(fb_rows),
            "previous_model_version": "v2.0",
            "new_model_version": None,
            "training_accuracy": None,
            "validation_accuracy": None,
            "f1_score": None,
            "is_deployed": False,
            "error_message": None,
            "trigger_type": "bootstrap_csv",
            "training_mode": tm,
            "dataset_row_count": len(dataset_rows),
            "small_sample_warning": None,
            "metrics_note": None,
            "clinic_id": ck,
        }

        asyncio.create_task(
            execute_bootstrap_training(training_id, fb_rows, pred_rows, ck, tm)
        )
        _refresh_training_metrics()

        return {
            "training_id": training_id,
            "status": "triggered",
            "trigger_type": "bootstrap_csv",
            "row_count": len(fb_rows),
            "clinic_id": ck,
            "training_scope": "global" if ck is None else "clinic",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("bootstrap-csv failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start bootstrap CSV training")


@router.get("/training/eligibility")
async def check_training_eligibility(clinic_id: Optional[str] = Query(None)):
    """Check if system is eligible for retraining (global or scoped to one clinic)."""
    try:
        ck = normalize_clinic_key(clinic_id)
        eligible_count = await count_eligible_feedback(clinic_id=ck)
        is_eligible = eligible_count >= TRAINING_THRESHOLD
        
        policy = None
        if ck is not None:
            try:
                policy = ct_store.get_clinic_feedback_pool(ck)
            except Exception:
                policy = None
        return {
            "eligible_feedback_count": eligible_count,
            "training_threshold": TRAINING_THRESHOLD,
            "is_eligible_for_training": is_eligible,
            "training_window_days": TRAINING_WINDOW_DAYS,
            "clinic_id": ck,
            "training_scope": "global" if ck is None else "clinic",
            "clinic_feedback_pool_policy": policy,
            "next_check_date": datetime.now() + timedelta(days=1),
        }
    except Exception as e:
        logger.error(f"Failed to check eligibility: {e}")
        raise HTTPException(status_code=500, detail="Failed to check eligibility")

@router.get("/training/history")
async def get_training_history(
    limit: int = 10, offset: int = 0, clinic_id: Optional[str] = Query(None)
):
    """Get training job history from actual training jobs (optional per-clinic filter)."""
    try:
        ck = normalize_clinic_key(clinic_id)
        rows = ct_store.list_training_jobs(limit=limit, offset=offset, clinic_id=ck)
        training_runs = []
        for job in rows:
            cid = job.get("clinic_id")
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
                "clinic_id": cid,
                "training_scope": "clinic" if cid not in (None, "") else "global",
            })

        return {
            "total_count": ct_store.count_training_jobs_total(clinic_id=ck),
            "limit": limit,
            "offset": offset,
            "clinic_id": ck,
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
