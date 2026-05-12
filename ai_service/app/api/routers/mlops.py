"""
MLOps API Endpoints for Veterinary AI System
Provides REST endpoints for drift detection, monitoring, and MLOps management
"""

import logging
import os
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ai_service.app.api.deps import require_admin
from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
from ai_service.app.domain.services.clinic_catalog_service import get_clinics_for_mlops
from ai_service.app.domain.schemas.training import TrainingTriggerRequest
from ai_service.app.infrastructure.external.mlair_client import (
    apply_mlair_promote_webhook_to_vet_ai,
    clinic_key_from_mlair_project_id,
    list_mlair_models_for_mlops_ui,
    mlair_project_for_clinic,
    mlair_tenant_id,
    try_sync_vetai_pin_to_mlair_production,
)
from ai_service.app.infrastructure.storage.model_store import (
    display_label_for_model_version,
    get_active_model,
    get_active_model_for_clinic,
    get_clinic_pinned_model,
    list_models_for_clinic_project_view,
    list_user_visible_model_versions,
    set_active_model,
    storage_scope_for_version,
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


class MLAirTrainingInvokeBody(BaseModel):
    """MLAir (or any automation) calls this to start the same PyTorch training as POST /training/trigger."""

    project_id: str = Field(..., min_length=1, description="MLAir project id: global project or clinic_<clinic_id>")
    tenant_id: Optional[str] = Field(
        default=None,
        description="If set, must equal Vet-AI MLAIR_TENANT_ID (usually default).",
    )
    training_mode: str = Field(default="local", pattern="^(local|eks_hybrid)$")
    force: bool = Field(
        default=True,
        description="When true, bypass TRAINING_THRESHOLD if at least one eligible sample exists (same as admin trigger).",
    )
    finetune_base_model_version: Optional[str] = None


class MLAirDatasetTrainInvokeBody(BaseModel):
    dataset_version_id: str = Field(..., min_length=1, description="MLAir dataset_versions.id (UUID)")
    project_id: str = Field(..., min_length=1, description="MLAir project id (global or clinic_<id>)")
    tenant_id: Optional[str] = Field(default=None)
    training_mode: str = Field(default="local", pattern="^(local|eks_hybrid)$")
    clinic_id: Optional[str] = Field(
        default=None,
        description="Optional explicit clinic_id override; otherwise derived from project_id prefix clinic_.",
    )
    trigger_reason: Optional[str] = Field(default="mlair_dataset_train_invoke")
    mlair_run_id: Optional[str] = Field(
        default=None,
        description="Pipeline run_id so tracking data is posted to the same run instead of creating a new one.",
    )


class MLAirPhaseInvokeBody(BaseModel):
    """Shared request body for multi-phase pipeline task endpoints."""
    dataset_version_id: Optional[str] = Field(default=None, description="Required for data_prep phase only")
    project_id: str = Field(..., min_length=1, description="MLAir project id")
    tenant_id: Optional[str] = Field(default=None)
    training_mode: str = Field(default="local", pattern="^(local|eks_hybrid)$")
    clinic_id: Optional[str] = Field(default=None)
    mlair_run_id: Optional[str] = Field(default=None, description="Pipeline run_id (used as session key)")


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
    mlair_models = list_mlair_models_for_mlops_ui(ck)
    if ck is None:
        versions = list_user_visible_model_versions(None)
        models = [
            {"version": v, "storageScope": "global", "label": v, "displayLabel": v} for v in versions
        ]
        active = get_active_model()
        return {
            "status": "success",
            "scope": "global",
            "active": active.model_version if active else None,
            "active_source": "state_file" if active else None,
            "versions": versions,
            "models": models,
            "mlair": mlair_models,
        }

    # Clinic UI: on-disk models under clinics/<slug>/ first, then global root (label ``… - global``).
    models = list_models_for_clinic_project_view(ck)
    versions = [str(m["version"]) for m in models]
    pinned = get_clinic_pinned_model(ck)
    effective = get_active_model_for_clinic(ck)
    global_active = get_active_model()

    def _label_for_version(mv: Optional[str]) -> Optional[str]:
        if not mv:
            return None
        sc = storage_scope_for_version(ck, mv)
        return display_label_for_model_version(ck, mv, sc)

    pv = pinned.model_version if pinned else None
    ev = effective.model_version if effective else None
    gv = global_active.model_version if global_active else None
    return {
        "status": "success",
        "scope": "clinic",
        "clinic_id": ck,
        "pinned": pv,
        "pinned_label": _label_for_version(pv),
        "effective": ev,
        "effective_label": _label_for_version(ev),
        "effective_source": "clinic_pin" if pinned is not None else "global_default",
        "global_active": gv,
        "global_active_label": _label_for_version(gv),
        "versions": versions,
        "models": models,
        "mlair": mlair_models,
    }


@router.put("/models/active", summary="Set active model version for inference")
async def set_active_model_version(request: ActiveModelRequest, _: bool = Depends(require_admin)):
    try:
        active = set_active_model(request.model_version)
        try_sync_vetai_pin_to_mlair_production(None, active.model_version)
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


@router.get(
    "/mlair/status",
    summary="Check MLAir API reachability and bearer token scope (GET /v1/auth/whoami)",
)
async def mlair_status(_: bool = Depends(require_admin)):
    try:
        from ai_service.app.infrastructure.external.mlair_client import mlair_whoami

        return mlair_whoami()
    except Exception as exc:
        logger.error("mlair status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/mlair/promote-webhook",
    summary="Inbound MLAir model promote → pin Vet-AI active (set MLAIR_MODEL_PROMOTE_WEBHOOK_URL on mlair-api)",
)
async def mlair_promote_webhook(
    body: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
):
    """
    When MLAir promotes a version to ``production``, call this URL so Vet-AI (and Spring callers of
    the same pin) stay aligned. JSON body: ``project_id``, ``version`` (MLAir integer); optional ``tenant_id``, ``model_id``.
    """
    expected = (os.getenv("MLAIR_PROMOTE_WEBHOOK_INBOUND_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="MLAIR_PROMOTE_WEBHOOK_INBOUND_TOKEN is not set on vet-ai",
        )
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="invalid webhook authorization")
    out = apply_mlair_promote_webhook_to_vet_ai(body)
    if out.get("status") != "ok":
        raise HTTPException(status_code=400, detail=out)
    return out


@router.post(
    "/mlair/training-invoke",
    status_code=201,
    summary="Inbound hook: MLAir pipeline → real Vet-AI training (Bearer MLAIR_TRAINING_INVOKE_TOKEN)",
)
async def mlair_training_invoke(
    body: MLAirTrainingInvokeBody,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    """
    Starts the in-process / EKS training job (``execute_training``), not MLAir's demo executor tasks.
    Configure MLAir (plugin or external HTTP step) to POST here after dataset snapshot / promote, etc.
    """
    expected = (os.getenv("MLAIR_TRAINING_INVOKE_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="MLAIR_TRAINING_INVOKE_TOKEN is not set on vet-ai",
        )
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="invalid webhook authorization")

    tid = (body.tenant_id or "").strip()
    if tid and tid != mlair_tenant_id():
        raise HTTPException(status_code=400, detail="tenant_id does not match MLAIR_TENANT_ID")

    pid = body.project_id.strip()
    gid = mlair_project_for_clinic(None)
    pref = (os.getenv("MLAIR_PROJECT_CLINIC_PREFIX") or "clinic_").strip() or "clinic_"
    if pid != gid and not pid.startswith(pref):
        raise HTTPException(
            status_code=400,
            detail=f"project_id must be the global MLAir project ({gid!r}) or {pref}<clinic_id>",
        )

    clinic_id = clinic_key_from_mlair_project_id(pid)
    from ai_service.app.api.routers.training import run_training_trigger_flow

    req = TrainingTriggerRequest(
        trigger_type="manual",
        trigger_reason="mlair_training_invoke",
        force=body.force,
        training_mode=body.training_mode,
        clinic_id=clinic_id,
        finetune_base_model_version=body.finetune_base_model_version,
    )
    out = await run_training_trigger_flow(req, background_tasks)
    return {**out, "clinic_id": clinic_id, "mlair_project_id": pid}


@router.post(
    "/mlair/dataset-train-invoke",
    status_code=201,
    summary="Inbound hook: MLAir dataset_version → Vet-AI bootstrap CSV training (Bearer MLAIR_TRAINING_INVOKE_TOKEN)",
)
async def mlair_dataset_train_invoke(
    body: MLAirDatasetTrainInvokeBody,
    authorization: Optional[str] = Header(default=None),
):
    expected = (os.getenv("MLAIR_TRAINING_INVOKE_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="MLAIR_TRAINING_INVOKE_TOKEN is not set on vet-ai")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="invalid webhook authorization")

    tid = (body.tenant_id or "").strip()
    if tid and tid != mlair_tenant_id():
        raise HTTPException(status_code=400, detail="tenant_id does not match MLAIR_TENANT_ID")

    pid = body.project_id.strip()
    gid = mlair_project_for_clinic(None)
    pref = (os.getenv("MLAIR_PROJECT_CLINIC_PREFIX") or "clinic_").strip() or "clinic_"
    if pid != gid and not pid.startswith(pref):
        raise HTTPException(
            status_code=400,
            detail=f"project_id must be the global MLAir project ({gid!r}) or {pref}<clinic_id>",
        )

    ck = normalize_clinic_key(body.clinic_id) if body.clinic_id else clinic_key_from_mlair_project_id(pid)

    from ai_service.app.infrastructure.external.mlair_client import download_mlair_dataset_version_csv_bytes
    from ai_service.app.api.routers.training import (
        build_training_dataset_snapshot,
        ct_store,
        execute_bootstrap_training,
        training_datasets,
        training_jobs,
        _refresh_training_metrics,
        _snapshot_active_model_version,
    )
    from ai_service.app.domain.services.training_service import parse_bootstrap_csv

    raw = download_mlair_dataset_version_csv_bytes(
        tenant_id=mlair_tenant_id(),
        project_id=pid,
        dataset_version_id=body.dataset_version_id,
    )
    if not raw:
        raise HTTPException(status_code=400, detail="downloaded dataset CSV is empty")

    try:
        fb_rows, pred_rows = parse_bootstrap_csv(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tm = (body.training_mode or "local").strip() or "local"
    if tm != "local":
        raise HTTPException(
            status_code=400,
            detail="CSV bootstrap training is local-only (training_mode=local) in this stack",
        )

    dataset_rows = build_training_dataset_snapshot(fb_rows, pred_rows)
    prev_bootstrap = _snapshot_active_model_version(ck)
    training_id = ct_store.create_training_job(
        status="running",
        total_predictions=ct_store.count_predictions(clinic_id=ck),
        eligible_feedback_count=len(fb_rows),
        previous_model_version=prev_bootstrap,
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
        "previous_model_version": prev_bootstrap,
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
        "trigger_reason": str(body.trigger_reason or "").strip() or "mlair_dataset_train_invoke",
    }

    asyncio.create_task(execute_bootstrap_training(training_id, fb_rows, pred_rows, ck, tm, mlair_run_id=body.mlair_run_id))
    _refresh_training_metrics()
    return {
        "training_id": training_id,
        "status": "triggered",
        "trigger_type": "bootstrap_csv",
        "row_count": len(fb_rows),
        "clinic_id": ck,
        "mlair_project_id": pid,
        "mlair_dataset_version_id": body.dataset_version_id,
    }


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


# ── Multi-phase pipeline endpoints ─────────────────────────────────────


def _validate_mlair_phase_auth(authorization: Optional[str]) -> None:
    expected = (os.getenv("MLAIR_TRAINING_INVOKE_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="MLAIR_TRAINING_INVOKE_TOKEN is not set on vet-ai")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="invalid webhook authorization")


def _validate_mlair_phase_project(body: MLAirPhaseInvokeBody) -> Optional[str]:
    tid = (body.tenant_id or "").strip()
    if tid and tid != mlair_tenant_id():
        raise HTTPException(status_code=400, detail="tenant_id does not match MLAIR_TENANT_ID")
    pid = body.project_id.strip()
    gid = mlair_project_for_clinic(None)
    pref = (os.getenv("MLAIR_PROJECT_CLINIC_PREFIX") or "clinic_").strip() or "clinic_"
    if pid != gid and not pid.startswith(pref):
        raise HTTPException(
            status_code=400,
            detail=f"project_id must be the global MLAir project ({gid!r}) or {pref}<clinic_id>",
        )
    return normalize_clinic_key(body.clinic_id) if body.clinic_id else clinic_key_from_mlair_project_id(pid)


@router.get(
    "/mlair/phase-status",
    summary="Poll phase job status for multi-phase pipeline tasks",
)
async def mlair_phase_status(
    session_id: str = Query(..., min_length=1),
    phase: str = Query(..., min_length=1),
):
    from ai_service.app.api.routers.training import phase_jobs

    pjk = f"{phase}:{session_id}"
    job = phase_jobs.get(pjk)
    if not job:
        raise HTTPException(status_code=404, detail=f"Phase job not found: {pjk}")
    return {
        "status": job.get("status", "running"),
        "progress_pct": job.get("progress_pct", 0),
        "current_phase": job.get("current_phase", ""),
        "phase_metrics": job.get("phase_metrics", {}),
        "error_message": job.get("error_message"),
        "result": job.get("result"),
    }


@router.post(
    "/mlair/data-prep-invoke",
    status_code=201,
    summary="Phase 1: Data collection + preprocessing (multi-phase pipeline)",
)
async def mlair_data_prep_invoke(
    body: MLAirPhaseInvokeBody,
    authorization: Optional[str] = Header(default=None),
):
    _validate_mlair_phase_auth(authorization)
    ck = _validate_mlair_phase_project(body)

    if not body.dataset_version_id:
        raise HTTPException(status_code=400, detail="dataset_version_id is required for data_prep phase")
    if not body.mlair_run_id:
        raise HTTPException(status_code=400, detail="mlair_run_id is required as session key")

    session_id = body.mlair_run_id
    tm = (body.training_mode or "local").strip() or "local"
    if tm != "local":
        raise HTTPException(status_code=400, detail="CSV bootstrap training is local-only (training_mode=local)")

    from ai_service.app.infrastructure.external.mlair_client import download_mlair_dataset_version_csv_bytes
    from ai_service.app.api.routers.training import (
        build_training_dataset_snapshot, ct_store, phase_jobs, run_phase_data_prep,
        _snapshot_active_model_version, _refresh_training_metrics,
    )
    from ai_service.app.domain.services.training_service import parse_bootstrap_csv

    raw = download_mlair_dataset_version_csv_bytes(
        tenant_id=mlair_tenant_id(),
        project_id=body.project_id.strip(),
        dataset_version_id=body.dataset_version_id,
    )
    if not raw:
        raise HTTPException(status_code=400, detail="downloaded dataset CSV is empty")

    try:
        fb_rows, pred_rows = parse_bootstrap_csv(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prev_model = _snapshot_active_model_version(ck)
    training_id = ct_store.create_training_job(
        status="running",
        total_predictions=ct_store.count_predictions(clinic_id=ck),
        eligible_feedback_count=len(fb_rows),
        previous_model_version=prev_model,
        trigger_type="bootstrap_csv",
        training_mode=tm,
        eks_node_group=None,
        dataset_row_count=len(fb_rows),
        clinic_id=ck,
    )

    finetune_base_model_dir: Optional[str] = None
    try:
        from ai_service.app.api.routers.training import resolve_model_dir
        if prev_model and str(prev_model).strip():
            cand = resolve_model_dir(str(prev_model).strip(), ck)
            if os.path.isdir(cand):
                finetune_base_model_dir = cand
    except Exception:
        pass

    pjk = f"data_prep:{session_id}"
    phase_jobs[pjk] = {"status": "running", "progress_pct": 0, "current_phase": "initializing"}

    asyncio.create_task(run_phase_data_prep(
        session_id, fb_rows, pred_rows, ck, tm, training_id,
        finetune_base_model_dir=finetune_base_model_dir,
    ))
    _refresh_training_metrics()

    return {
        "phase_job_id": pjk,
        "session_id": session_id,
        "training_id": training_id,
        "status": "triggered",
        "phase": "data_prep",
    }


@router.post(
    "/mlair/model-train-invoke",
    status_code=201,
    summary="Phase 2: Model fitting + calibration (multi-phase pipeline)",
)
async def mlair_model_train_invoke(
    body: MLAirPhaseInvokeBody,
    authorization: Optional[str] = Header(default=None),
):
    _validate_mlair_phase_auth(authorization)
    _validate_mlair_phase_project(body)

    if not body.mlair_run_id:
        raise HTTPException(status_code=400, detail="mlair_run_id is required as session key")

    session_id = body.mlair_run_id

    from ai_service.app.domain.services.training_session import session_exists
    if not session_exists(session_id):
        raise HTTPException(status_code=400, detail=f"Session not found: {session_id}. data_prep must run first.")

    from ai_service.app.api.routers.training import phase_jobs, run_phase_model_train

    pjk = f"model_train:{session_id}"
    phase_jobs[pjk] = {"status": "running", "progress_pct": 0, "current_phase": "model_fit"}

    asyncio.create_task(run_phase_model_train(session_id))

    return {
        "phase_job_id": pjk,
        "session_id": session_id,
        "status": "triggered",
        "phase": "model_train",
    }


@router.post(
    "/mlair/validation-invoke",
    status_code=201,
    summary="Phase 3: Regression gate + feedback improvement gate (multi-phase pipeline)",
)
async def mlair_validation_invoke(
    body: MLAirPhaseInvokeBody,
    authorization: Optional[str] = Header(default=None),
):
    _validate_mlair_phase_auth(authorization)
    _validate_mlair_phase_project(body)

    if not body.mlair_run_id:
        raise HTTPException(status_code=400, detail="mlair_run_id is required as session key")

    session_id = body.mlair_run_id

    from ai_service.app.domain.services.training_session import session_exists
    if not session_exists(session_id):
        raise HTTPException(status_code=400, detail=f"Session not found: {session_id}. model_train must run first.")

    from ai_service.app.api.routers.training import phase_jobs, run_phase_validation

    pjk = f"validation:{session_id}"
    phase_jobs[pjk] = {"status": "running", "progress_pct": 0, "current_phase": "cv_scoring"}

    asyncio.create_task(run_phase_validation(session_id))

    return {
        "phase_job_id": pjk,
        "session_id": session_id,
        "status": "triggered",
        "phase": "validation",
    }


@router.post(
    "/mlair/persist-invoke",
    status_code=201,
    summary="Phase 4: Model save + S3 + MLflow + MLAir sync (multi-phase pipeline)",
)
async def mlair_persist_invoke(
    body: MLAirPhaseInvokeBody,
    authorization: Optional[str] = Header(default=None),
):
    _validate_mlair_phase_auth(authorization)
    _validate_mlair_phase_project(body)

    if not body.mlair_run_id:
        raise HTTPException(status_code=400, detail="mlair_run_id is required as session key")

    session_id = body.mlair_run_id

    from ai_service.app.domain.services.training_session import session_exists
    if not session_exists(session_id):
        raise HTTPException(status_code=400, detail=f"Session not found: {session_id}. validation must run first.")

    from ai_service.app.api.routers.training import phase_jobs, run_phase_persist

    pjk = f"persist:{session_id}"
    phase_jobs[pjk] = {"status": "running", "progress_pct": 0, "current_phase": "model_save"}

    asyncio.create_task(run_phase_persist(session_id, mlair_run_id=body.mlair_run_id))

    return {
        "phase_job_id": pjk,
        "session_id": session_id,
        "status": "triggered",
        "phase": "persist",
    }

