from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ai_service.app.api.deps import require_admin
from ai_service.app.infrastructure.external import mlair_client

router = APIRouter(prefix="/mlair", tags=["MLAir Integration"])


class MlairTriggerRequest(BaseModel):
    pipeline_id: Optional[str] = Field(default=None, description="Optional MLAir pipeline override")
    idempotency_key: Optional[str] = Field(default=None, description="Optional idempotency key")
    clinic_id: Optional[str] = Field(default=None, description="Optional clinic id for tenant/project scope mapping")
    training_mode: Optional[str] = Field(default=None, description="Optional MLAir training mode (quick|standard|full)")
    override_config: Optional[dict] = Field(default=None, description="Optional per-run readiness override config")
    context: Optional[dict] = Field(
        default=None,
        description="Optional MLAir run plugin_context (e.g. mlair_model_id, mlair_new_version_stage)",
    )


class MlairModelDatasetTriggerRequest(BaseModel):
    """Body for MLAir ``POST .../runs/trigger`` (model + dataset; pipeline resolved server-side)."""

    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version_id: Optional[str] = Field(default=None, description="Optional; default latest in MLAir")
    pipeline_id_override: Optional[str] = Field(default=None, description="Advanced: force pipeline id")
    idempotency_key: Optional[str] = Field(default=None, description="Optional stable key")
    clinic_id: Optional[str] = Field(default=None, description="Optional clinic for tenant/project scope")
    training_mode: Optional[str] = Field(default=None, description="Optional training_mode sent to MLAir")
    context: Optional[dict] = Field(default=None, description="Merged into MLAir plugin_context")


def _normalize_mlair_status(raw: Optional[str]) -> str:
    key = str(raw or "").strip().lower()
    if key in {"completed", "success", "succeeded"}:
        return "SUCCESS"
    if key in {"failed", "error"}:
        return "FAILED"
    if key in {"running"}:
        return "RUNNING"
    return "PENDING"


def _job_idempotency_key(job: Dict[str, Any]) -> str:
    explicit = str(job.get("idempotency_key") or "").strip()
    if explicit:
        return explicit
    tid = int(job.get("training_id") or 0)
    trigger_type = str(job.get("trigger_type") or "").strip().lower()
    if trigger_type == "bootstrap_csv":
        return f"vet-ai-bootstrap-job-{tid}"
    return f"vet-ai-training-job-{tid}"


def _job_context(job: Dict[str, Any]) -> Dict[str, str]:
    clinic_id = str(job.get("clinic_id") or "").strip() or "global"
    out: Dict[str, str] = {
        "source_app": "vet-ai",
        "source_task_type": str(job.get("trigger_type") or "continuous_training"),
        "source_training_id": str(job.get("training_id") or ""),
        "source_clinic_id": clinic_id,
    }
    if clinic_id and clinic_id != "global":
        out["clinic_id"] = clinic_id
    return out


def _job_params(job: Dict[str, Any], mapped_status: str) -> Dict[str, str]:
    out: Dict[str, str] = {
        "source_app": "vet-ai",
        "source_status": mapped_status,
    }
    field_map = {
        "source_training_id": "training_id",
        "source_trigger_type": "trigger_type",
        "source_training_mode": "training_mode",
        "source_previous_model_version": "previous_model_version",
        "source_new_model_version": "new_model_version",
        "source_clinic_id": "clinic_id",
    }
    for out_key, src_key in field_map.items():
        value = job.get(src_key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out[out_key] = text
    return out


@router.get("/status")
async def mlair_status():
    return {"status": "success", "mlair": mlair_client.config_summary()}


@router.get(
    "/pipelines",
    summary="List MLAir pipelines for a tenant/project (from runs + pipeline_versions)",
)
async def list_mlair_pipelines(
    clinic_id: Optional[str] = Query(
        None,
        description="Omit for global MLAIR_PROJECT_ID; set for clinic_<slug> project mapping.",
    ),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin),
):
    try:
        data = mlair_client.list_project_pipelines(clinic_id=clinic_id, limit=limit, offset=offset)
        return {"status": "success", "mlair": data}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/pipelines/{pipeline_id}/versions",
    summary="List MLAir pipeline DAG versions for a pipeline_id in a tenant/project",
)
async def list_mlair_pipeline_versions(
    pipeline_id: str,
    clinic_id: Optional[str] = Query(None, description="Omit for global project"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin),
):
    try:
        data = mlair_client.list_project_pipeline_versions(
            clinic_id=clinic_id, pipeline_id=pipeline_id, limit=limit, offset=offset
        )
        return {"status": "success", "mlair": data}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/alignment",
    summary="Compare Vet-AI active model vs MLAir production for a clinic (read-only)",
)
async def mlair_serving_alignment(
    clinic_id: Optional[str] = Query(
        None,
        description=(
            "Canonical clinic UUID (e.g. 78343a5e-047b-5edb-9975-678bf3f815c6) or exact catalog name "
            "(e.g. demo0 if MLOPS_CLINICS_JSON / customers lists that name). Omit for global."
        ),
    ),
    _: bool = Depends(require_admin),
):
    """
    MLAir project id is ``clinic_<slug>`` of the **resolved** id (UUID → ``clinic_78343a5e-047b-5edb-9975-678bf3f815c6``).
    Scoped model name defaults to ``vet-<slug>`` of the same id (e.g. ``vet-78343a5e-047b-5edb-9975-678bf3f815c6``).
    """
    try:
        report = mlair_client.inspect_serving_alignment_with_mlair(clinic_id)
        return {"status": "success", "report": report}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/runs/trigger-by-model")
async def trigger_mlair_run_by_model_dataset(
    request: MlairModelDatasetTriggerRequest,
    _: bool = Depends(require_admin),
):
    """Proxy to MLAir model-centric run trigger (same as enabling VETAI_MLAIR_MIRROR_USE_RUNS_TRIGGER for mirrors)."""
    key = request.idempotency_key or f"vet-ai-md-{int(datetime.now(tz=timezone.utc).timestamp())}"
    try:
        data = mlair_client.trigger_run_by_model_dataset(
            model_id=request.model_id,
            dataset_id=request.dataset_id,
            idempotency_key=key,
            clinic_id=request.clinic_id,
            dataset_version_id=request.dataset_version_id,
            pipeline_id_override=request.pipeline_id_override,
            training_mode=request.training_mode,
            context=request.context if isinstance(request.context, dict) else None,
        )
        return {"status": "success", "mlair_response": data, "idempotency_key": key}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/runs/training")
async def trigger_mlair_training(request: MlairTriggerRequest, _: bool = Depends(require_admin)):
    key = request.idempotency_key or f"vet-ai-training-{int(datetime.now(tz=timezone.utc).timestamp())}"
    try:
        data = mlair_client.trigger_training_run(
            idempotency_key=key,
            pipeline_id=request.pipeline_id,
            clinic_id=request.clinic_id,
            training_mode=request.training_mode,
            override_config=request.override_config,
            context=request.context if isinstance(request.context, dict) else None,
        )
        return {"status": "success", "mlair_response": data, "idempotency_key": key}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/runs/{run_id}")
async def get_mlair_run(run_id: str):
    try:
        data = mlair_client.get_run(run_id)
        return {"status": "success", "mlair_run": data}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/models/sync")
async def sync_mlair_models(_: bool = Depends(require_admin)):
    try:
        data = mlair_client.sync_all_models_to_mlair()
        return {"status": "success", "sync": data}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post(
    "/models/dedupe-versions",
    summary="Remove duplicate MLAir model versions (same canonical file:// artifact)",
)
async def mlair_dedupe_model_versions(
    clinic_id: Optional[str] = Query(
        None,
        description="Clinic id for one project; omit with all_scopes=true; omit both for global project only",
    ),
    all_scopes: bool = Query(
        False,
        description="When true, dedupe global + every clinic from the Vet-AI clinic catalog",
    ),
    dry_run: bool = Query(
        True,
        description="Preview only. Use dry_run=false to DELETE duplicate version rows in MLAir",
    ),
    _: bool = Depends(require_admin),
):
    if clinic_id and all_scopes:
        raise HTTPException(status_code=400, detail="Pass either clinic_id or all_scopes=true, not both")
    try:
        data = mlair_client.dedupe_mlair_registered_model_versions(
            clinic_id_for_scope=clinic_id,
            all_catalog_clinics=all_scopes,
            dry_run=dry_run,
        )
        return {"status": "success", "dedupe": data}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/runs/backfill")
async def backfill_mlair_runs_from_db(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    clinic_id: Optional[str] = Query(default=None),
    _: bool = Depends(require_admin),
):
    """
    Backfill vet-ai training history from DB into MLAir runs.
    Safe to run multiple times because idempotency_key is stable per training job.
    """
    try:
        from ai_service.app.api.routers.training import ct_store
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot load training store: {exc}")

    jobs = ct_store.list_training_jobs(limit=limit, offset=offset, clinic_id=clinic_id)
    synced = 0
    failed = 0
    failures: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for job in jobs:
        try:
            key = _job_idempotency_key(job)
            cid = str(job.get("clinic_id") or "").strip() or None
            status = _normalize_mlair_status(job.get("status"))
            mlair_client.mirror_training_job_to_mlair(
                idempotency_key=key,
                clinic_id=cid,
                training_mode=None,
                override_config=None,
                context=_job_context(job),
            )
            mlair_client.sync_training_outcome_to_mlair(
                idempotency_key=key,
                status=status,
                clinic_id=cid,
                reason=str(job.get("error_message") or "").strip() or None,
                run_params=_job_params(job, status),
            )
            items.append(
                {
                    "training_id": int(job.get("training_id") or 0),
                    "idempotency_key": key,
                    "clinic_id": cid or "global",
                    "status": status,
                }
            )
            synced += 1
        except Exception as exc:
            failed += 1
            failures.append(
                {
                    "training_id": int(job.get("training_id") or 0),
                    "error": str(exc),
                }
            )

    return {
        "status": "success",
        "summary": {
            "requested_limit": limit,
            "offset": offset,
            "clinic_id": clinic_id,
            "jobs_loaded": len(jobs),
            "synced": synced,
            "failed": failed,
        },
        "items": items,
        "failures": failures,
    }
