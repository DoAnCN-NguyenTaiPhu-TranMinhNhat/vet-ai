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
    return {
        "source_app": "vet-ai",
        "source_task_type": str(job.get("trigger_type") or "continuous_training"),
        "source_training_id": str(job.get("training_id") or ""),
        "source_clinic_id": clinic_id,
    }


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


@router.post("/runs/training")
async def trigger_mlair_training(request: MlairTriggerRequest, _: bool = Depends(require_admin)):
    key = request.idempotency_key or f"vet-ai-training-{int(datetime.now(tz=timezone.utc).timestamp())}"
    try:
        data = mlair_client.trigger_training_run(
            idempotency_key=key,
            pipeline_id=request.pipeline_id,
            clinic_id=request.clinic_id,
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
            mlair_client.trigger_training_run(
                idempotency_key=key,
                clinic_id=cid,
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
