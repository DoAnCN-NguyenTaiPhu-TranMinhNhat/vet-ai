#!/usr/bin/env python3
"""
EKS Training Script - Runs inside Kubernetes pod on training node group
This script is designed to be executed in a Kubernetes job on EKS
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data_from_api() -> tuple:
    """Load training data from the FastAPI service"""
    try:
        # Get API endpoint from environment
        api_base_url = os.getenv('API_BASE_URL', 'http://vet-ai-service:8000')
        
        # Fetch feedback data
        feedback_response = requests.get(f"{api_base_url}/continuous-training/feedback/all")
        feedback_response.raise_for_status()
        feedback_data = feedback_response.json()
        
        # Fetch prediction logs
        predictions_response = requests.get(f"{api_base_url}/continuous-training/predictions/all")
        predictions_response.raise_for_status()
        prediction_logs = predictions_response.json()
        
        logger.info(f"Loaded {len(feedback_data)} feedback and {len(prediction_logs)} predictions from API")
        return feedback_data, prediction_logs
        
    except Exception as e:
        logger.error(f"Failed to load data from API: {e}")
        raise

def load_training_data_from_files() -> tuple:
    """Load training data from mounted files (for testing)"""
    try:
        data_dir = os.getenv('TRAINING_DATA_DIR', '/data')
        
        with open(os.path.join(data_dir, 'feedback.json'), 'r') as f:
            feedback_data = json.load(f)
        
        with open(os.path.join(data_dir, 'predictions.json'), 'r') as f:
            prediction_logs = json.load(f)
        
        logger.info(f"Loaded {len(feedback_data)} feedback and {len(prediction_logs)} predictions from files")
        return feedback_data, prediction_logs
        
    except Exception as e:
        logger.error(f"Failed to load data from files: {e}")
        raise


def load_training_data_from_postgres() -> tuple:
    """Load training data directly from PostgreSQL (Pha A)."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn or not str(dsn).strip():
        raise RuntimeError("DATABASE_URL is not configured for training worker")

    days = int(os.getenv("TRAINING_WINDOW_DAYS", "30"))
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    final_diagnosis,
                    is_correct,
                    ai_diagnosis,
                    confidence_rating,
                    comments,
                    veterinarian_id,
                    is_training_eligible,
                    data_quality_score,
                    created_at AS timestamp
                FROM ai_feedback
                WHERE is_training_eligible = TRUE
                  AND created_at >= NOW() - (%s || ' days')::INTERVAL
                ORDER BY created_at ASC
                """,
                (days,),
            )
            feedback_data = cur.fetchall()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    visit_id,
                    pet_id,
                    prediction_input,
                    prediction_output,
                    model_version,
                    confidence_score,
                    top_k_predictions,
                    veterinarian_id,
                    clinic_id,
                    created_at
                FROM ai_prediction_logs
                ORDER BY created_at ASC
                """
            )
            prediction_logs = cur.fetchall()

        logger.info(
            "Loaded %d eligible feedback and %d predictions from PostgreSQL",
            len(feedback_data),
            len(prediction_logs),
        )
        # RealDictCursor already returns dict-like rows
        return [dict(r) for r in feedback_data], [dict(r) for r in prediction_logs]
    finally:
        conn.close()


def _update_training_job_completed(
    conn,
    training_id: int,
    *,
    model_version: str,
    training_metrics: Dict[str, Any],
) -> None:
    tm = training_metrics or {}
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ai_training_jobs
            SET
                status = 'completed',
                end_time = NOW(),
                new_model_version = %(new_model_version)s,
                training_accuracy = %(training_accuracy)s,
                validation_accuracy = %(validation_accuracy)s,
                f1_score = %(f1_score)s,
                is_deployed = FALSE,
                error_message = NULL,
                small_sample_warning = %(small_sample_warning)s,
                metrics_note = %(metrics_note)s,
                updated_at = NOW()
            WHERE training_id = %(training_id)s
            """,
            {
                "training_id": int(training_id),
                "new_model_version": model_version,
                "training_accuracy": tm.get("training_accuracy"),
                "validation_accuracy": tm.get("validation_accuracy"),
                "f1_score": tm.get("validation_f1"),
                "small_sample_warning": tm.get("small_sample_warning"),
                "metrics_note": tm.get("metrics_note"),
            },
        )
        if cur.rowcount == 0:
            logger.warning("ai_training_jobs row not found (training_id=%s)", training_id)
    conn.commit()


def _update_training_job_failed(conn, training_id: int, *, error_message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ai_training_jobs
            SET
                status = 'failed',
                end_time = NOW(),
                is_deployed = FALSE,
                error_message = %(error_message)s,
                updated_at = NOW()
            WHERE training_id = %(training_id)s
            """,
            {"training_id": int(training_id), "error_message": str(error_message)[:4000]},
        )
        if cur.rowcount == 0:
            logger.warning("ai_training_jobs row not found (training_id=%s)", training_id)
    conn.commit()


def _clear_feedback_postgres(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ai_feedback")
    conn.commit()

def main():
    """Main training execution function"""
    training_id = os.getenv('TRAINING_ID', 'unknown')
    logger.info(f"Starting EKS training job {training_id}")
    
    try:
        # Add ai_service to path
        sys.path.insert(0, '/app/ai_service')
        from training_engine import execute_training
        
        # Load training data
        data_source = os.getenv('DATA_SOURCE', 'postgres')  # 'postgres' | 'api' | 'files'

        if data_source == 'api':
            feedback_data, prediction_logs = load_training_data_from_api()
        elif data_source == 'files':
            feedback_data, prediction_logs = load_training_data_from_files()
        else:
            feedback_data, prediction_logs = load_training_data_from_postgres()
        
        # Validate data
        if not feedback_data or not prediction_logs:
            raise ValueError("No training data available")
        
        # Execute training
        logger.info(f"Executing training with {len(feedback_data)} samples")
        result = execute_training(feedback_data, prediction_logs, "eks_hybrid")
        
        # Persist status/matrics into PostgreSQL (Pha A)
        dsn = os.getenv("DATABASE_URL")
        if dsn and str(dsn).strip() and str(training_id).strip().isdigit():
            conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
            try:
                training_id_int = int(training_id)
                if result.get("status") == "completed":
                    tm = result.get("training_metrics") or {}
                    model_version = result.get("model_version")
                    _update_training_job_completed(
                        conn,
                        training_id_int,
                        model_version=model_version,
                        training_metrics=tm,
                    )
                    _clear_feedback_postgres(conn)
                else:
                    _update_training_job_failed(
                        conn,
                        training_id_int,
                        error_message=str(result.get("error", "Unknown error")),
                    )
            finally:
                conn.close()

        # Output result for log parsing / debug
        result_json = json.dumps(result, default=str)
        print(f"TRAINING_RESULT:{result_json}")
        
        # Log summary
        if result['status'] == 'completed':
            logger.info(f"Training completed successfully!")
            logger.info(f"Model version: {result['model_version']}")
            metrics = result['training_metrics']
            logger.info(f"Validation accuracy: {metrics.get('validation_accuracy', 0):.3f}")
            logger.info(f"Validation F1: {metrics.get('validation_f1', 0):.3f}")
        else:
            logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Training execution failed: {e}")
        dsn = os.getenv("DATABASE_URL")
        if dsn and str(dsn).strip() and str(training_id).strip().isdigit():
            try:
                conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
                try:
                    _update_training_job_failed(
                        conn,
                        int(training_id),
                        error_message=str(e),
                    )
                finally:
                    conn.close()
            except Exception as db_err:
                logger.warning("Failed to persist training failure to DB: %s", db_err)
        error_result = {
            'status': 'failed',
            'error': str(e),
            'training_id': training_id
        }
        print(f"TRAINING_RESULT:{json.dumps(error_result, default=str)}")
        return error_result

if __name__ == "__main__":
    main()
