"""
MLOps API Endpoints for Veterinary AI System
Provides REST endpoints for drift detection, monitoring, and MLOps management
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import logging
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ai_service.app.api.deps import require_admin
from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
from ai_service.app.domain.services.clinic_catalog_service import get_clinics_for_mlops
from ai_service.app.infrastructure.storage.model_store import (
    get_active_model,
    get_active_model_for_clinic,
    get_clinic_pinned_model,
    list_model_versions,
    set_active_model,
)
from ai_service.app.mlops.manager import MLOpsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mlops", tags=["MLOps"])
mlops_manager = MLOpsManager()


class PredictionRequest(BaseModel):
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    actual: Optional[str] = Field(None, description="Actual class (if available)")
    features: Optional[Dict[str, Any]] = Field(None, description="Input features")


class ReferenceDataRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Training data as list of dictionaries")


class DriftCheckRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Current data for drift detection")


class ConfigUpdateRequest(BaseModel):
    drift_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    accuracy_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_training_samples: Optional[int] = Field(None, gt=0)
    max_error_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    check_interval_hours: Optional[int] = Field(None, gt=0)


class ActiveModelRequest(BaseModel):
    model_version: str = Field(..., description="Model version to activate for inference")


class ClinicTrainingPolicyBody(BaseModel):
    feedback_pool: str = Field(..., pattern="^(GLOBAL|CLINIC_ONLY)$")


@router.get(
    "/clinics",
    summary="Clinics for MLOps admin (customers-service + cache, or MLOPS_CLINICS_JSON / defaults)",
)
async def list_clinics_for_mlops():
    clinics, source = get_clinics_for_mlops()
    return {"status": "success", "clinics": clinics, "source": source}


@router.get("/clinics/{clinic_id}/training-policy", summary="Get feedback pool policy for a clinic")
async def get_clinic_training_policy(clinic_id: str):
    from ai_service.app.domain.services.training_service import get_training_runtime_handles
    ct_store, _ = get_training_runtime_handles()

    ck = normalize_clinic_key(clinic_id)
    if ck is None:
        raise HTTPException(status_code=400, detail="Invalid clinic_id")
    try:
        pool = ct_store.get_clinic_feedback_pool(ck)
        return {"status": "success", "clinic_id": ck, "feedback_pool": pool}
    except Exception as exc:
        logger.error("Failed to get clinic training policy: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/clinics/{clinic_id}/training-policy", summary="Set feedback pool policy for a clinic")
async def set_clinic_training_policy(
    clinic_id: str,
    body: ClinicTrainingPolicyBody,
    _: bool = Depends(require_admin),
):
    from ai_service.app.domain.services.training_service import get_training_runtime_handles
    ct_store, _ = get_training_runtime_handles()

    ck = normalize_clinic_key(clinic_id)
    if ck is None:
        raise HTTPException(status_code=400, detail="Invalid clinic_id")
    try:
        ct_store.set_clinic_feedback_pool(ck, body.feedback_pool)
        return {"status": "success", "clinic_id": ck, "feedback_pool": body.feedback_pool.upper()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to set clinic training policy: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/initialize", summary="Initialize MLOps with reference data")
async def initialize_mlops(request: ReferenceDataRequest):
    try:
        df = pd.DataFrame(request.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        result = mlops_manager.initialize_reference_data(df)
        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Initialization failed"))
        return {"status": "success", "message": "MLOps initialized successfully", "details": result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error initializing MLOps: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/prediction", summary="Process prediction through MLOps")
async def process_prediction(request: PredictionRequest):
    try:
        result = mlops_manager.process_prediction(request.model_dump())
        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        return {
            "status": "success",
            "message": "Prediction processed successfully",
            "training_eligible": result.get("training_eligible", False),
            "timestamp": result.get("timestamp"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error processing prediction: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/drift-check", summary="Check for data drift")
async def check_data_drift(request: DriftCheckRequest):
    try:
        df = pd.DataFrame(request.data)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        result = mlops_manager.check_data_drift(df)
        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Drift check failed"))
        return {"status": "success", "message": "Drift check completed", "drift_results": result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error checking drift: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status", summary="Get comprehensive MLOps status")
async def get_mlops_status():
    try:
        return {"status": "success", "mlops_status": mlops_manager.get_mlops_status()}
    except Exception as exc:
        logger.error("Error getting MLOps status: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/model-metrics", summary="Get model performance metrics")
async def get_model_metrics():
    try:
        return {"status": "success", "metrics": mlops_manager.model_monitor.get_model_metrics()}
    except Exception as exc:
        logger.error("Error getting model metrics: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health", summary="Run comprehensive health check")
async def run_health_check():
    try:
        return {"status": "success", "health_check": mlops_manager.run_health_check()}
    except Exception as exc:
        logger.error("Error running health check: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/models", summary="List model versions + active (global or per-clinic)")
async def list_models(clinic_id: Optional[str] = Query(None)):
    ck = normalize_clinic_key(clinic_id)
    versions = list_model_versions(ck)
    if ck is None:
        active = get_active_model()
        return {
            "status": "success",
            "scope": "global",
            "active": active.model_version if active else None,
            "active_source": "state_file" if active else None,
            "versions": versions,
        }

    pinned = get_clinic_pinned_model(ck)
    effective = get_active_model_for_clinic(ck)
    global_active = get_active_model()
    return {
        "status": "success",
        "scope": "clinic",
        "clinic_id": ck,
        "pinned": pinned.model_version if pinned else None,
        "effective": effective.model_version if effective else None,
        "effective_source": "clinic_pin" if pinned is not None else "global_default",
        "global_active": global_active.model_version if global_active else None,
        "versions": versions,
    }


@router.put("/models/active", summary="Set active model version for inference")
async def set_active_model_version(request: ActiveModelRequest, _: bool = Depends(require_admin)):
    try:
        active = set_active_model(request.model_version)
        return {"status": "success", "active": active.model_version, "model_dir": active.model_dir}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to set active model: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/training/last", summary="Get last training job status")
async def get_last_training_job(clinic_id: Optional[str] = Query(None)):
    try:
        from ai_service.app.domain.services.training_service import get_training_runtime_handles
        ct_store, training_jobs = get_training_runtime_handles()

        ck = normalize_clinic_key(clinic_id)
        last_row = None
        last_successful = None
        try:
            recent_rows = ct_store.list_training_jobs(limit=50, offset=0, clinic_id=ck)
            if recent_rows:
                last_row = recent_rows[0]
            successful_jobs = [
                j for j in (recent_rows or []) if str(j.get("status", "")).lower() == "completed" and bool(j.get("is_deployed"))
            ]
            last_successful = max(successful_jobs, key=lambda x: x.get("training_id", 0)) if successful_jobs else None
        except Exception:
            if training_jobs:
                mem_values = list(training_jobs.values())
                if ck is not None:
                    mem_values = [j for j in mem_values if str(j.get("clinic_id") or "").strip() == str(ck).strip()]
                if mem_values:
                    last_row = max(mem_values, key=lambda x: x.get("training_id", 0))
                    successful_jobs = [
                        j for j in mem_values if str(j.get("status", "")).lower() == "completed" and bool(j.get("is_deployed"))
                    ]
                    if successful_jobs:
                        last_successful = max(successful_jobs, key=lambda x: x.get("training_id", 0))

        active_model_version = (get_active_model_for_clinic(ck) if ck else get_active_model())
        return {
            "status": "success",
            "scope": "clinic" if ck else "global",
            "clinic_id": ck,
            "last_training": last_row,
            "last_successful_training": last_successful,
            "active_model": active_model_version.model_version if active_model_version else None,
            "serving_model_source": "last_successful_training"
            if last_successful and active_model_version and active_model_version.model_version == last_successful.get("new_model_version")
            else "active_model_registry",
        }
    except Exception as exc:
        logger.error("Failed to read training jobs: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/mlflow/latest-vs-active", summary="Compare latest MLflow run with active model")
async def mlflow_latest_vs_active(clinic_id: Optional[str] = Query(None)):
    try:
        from ai_service.app.infrastructure.external.mlflow_client import compare_latest_run_to_active

        return compare_latest_run_to_active(clinic_id=clinic_id)
    except Exception as exc:
        logger.error("mlflow latest-vs-active failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/drift/summary", summary="Get drift summary + simple alert rule")
async def get_drift_summary(days: int = 7, clinic_id: Optional[str] = Query(None)):
    try:
        ck = normalize_clinic_key(clinic_id)
        summary = mlops_manager.drift_detector.get_drift_summary(days=days)
        drift_score = mlops_manager.training_eligibility.get("drift_score", 0.0)
        alert = (
            {"severity": "warning", "message": "Data drift detected (drift_score > 0.5)", "drift_score": drift_score}
            if drift_score > 0.5
            else None
        )
        return {
            "status": "success",
            "scope": "clinic" if ck else "global",
            "clinic_id": ck,
            "summary": summary,
            "alert": alert,
            "note": "Drift preset is workspace-global; clinic_id is echoed for UI scope only.",
        }
    except Exception as exc:
        logger.error("Failed to get drift summary: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/retraining-eligibility", summary="Check if model should be retrained")
async def check_retraining_eligibility():
    try:
        return {"status": "success", "retraining_recommendation": mlops_manager.should_retrain_model()}
    except Exception as exc:
        logger.error("Error checking retraining eligibility: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/performance-trend", summary="Get performance trend over time")
async def get_performance_trend(hours: int = 24):
    try:
        if hours <= 0 or hours > 168:
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        return {"status": "success", "performance_trend": mlops_manager.model_monitor.get_performance_trend(hours=hours)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting performance trend: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/alerts", summary="Get active alerts")
async def get_alerts():
    try:
        alerts = mlops_manager.model_monitor.check_model_health().get("alerts", [])
        return {"status": "success", "active_alerts": alerts, "total_alerts": len(alerts)}
    except Exception as exc:
        logger.error("Error getting alerts: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/config", summary="Update MLOps configuration")
async def update_config(request: ConfigUpdateRequest):
    try:
        current_config = mlops_manager.config.copy()
        current_config.update(request.model_dump(exclude_unset=True))
        mlops_manager.config = current_config
        return {"status": "success", "message": "Configuration updated successfully", "updated_config": current_config}
    except Exception as exc:
        logger.error("Error updating config: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/config", summary="Get current MLOps configuration")
async def get_config():
    try:
        return {"status": "success", "config": mlops_manager.config}
    except Exception as exc:
        logger.error("Error getting config: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/artifact-storage", summary="S3 model artifact backup status")
async def get_artifact_storage_status():
    try:
        from ai_service.app.infrastructure.external.s3_client import s3_bucket, s3_prefix_base

        bucket = s3_bucket()
        return {"status": "success", "enabled": bool(bucket), "bucket": bucket, "prefix_base": s3_prefix_base()}
    except Exception as exc:
        logger.error("artifact-storage status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/export", summary="Export MLOps data")
async def export_mlops_data(background_tasks: BackgroundTasks):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mlops_export_{timestamp}.json"
        filepath = f"/tmp/{filename}"
        background_tasks.add_task(lambda: mlops_manager.export_mlops_data(filepath))
        return {"status": "success", "message": "Export started in background", "filename": filename, "filepath": filepath}
    except Exception as exc:
        logger.error("Error exporting MLOps data: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/reset", summary="Reset MLOps system")
async def reset_mlops():
    try:
        global mlops_manager
        mlops_manager = MLOpsManager()
        return {"status": "success", "message": "MLOps system reset successfully"}
    except Exception as exc:
        logger.error("Error resetting MLOps: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/info", summary="Get MLOps system information")
async def get_system_info():
    try:
        info = {
            "system": "Veterinary AI MLOps",
            "version": "1.0.0",
            "components": [
                "Data Drift Detection (Evidently AI)",
                "Model Performance Monitoring",
                "Training Eligibility Checker",
                "Health Monitoring",
            ],
            "initialized": mlops_manager.reference_data is not None,
            "total_predictions": mlops_manager.model_monitor.model_metrics["total_predictions"],
            "last_health_check": mlops_manager.last_health_check.isoformat() if mlops_manager.last_health_check else None,
        }
        return {"status": "success", "system_info": info}
    except Exception as exc:
        logger.error("Error getting system info: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

