"""
MLAir model registry proxy (admin).

Paths mirror ``ml-air`` ``api/app/api/routes/v1.py`` model registry section.
``clinic_id`` maps to the same tenant/project as :func:`mlair_client._resolve_scope`.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ai_service.app.api.deps import require_admin
from ai_service.app.infrastructure.external import mlair_client

router = APIRouter(prefix="/mlair/registry", tags=["MLAir Registry"])


class PipelineMappingBody(BaseModel):
    pipeline_id: str = Field(min_length=1)


class PromoteBody(BaseModel):
    version: int = Field(ge=1)
    stage: str = Field(default="production")


class TriggerPolicyBody(BaseModel):
    trigger_mode: str = Field(default="manual")
    debounce_minutes: int = Field(default=10, ge=0)
    schedule_cron: Optional[str] = None


class ApprovalBody(BaseModel):
    approval_status: str = Field(min_length=1)
    reason: Optional[str] = None


class CreateModelBody(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = None


class CreateModelVersionBody(BaseModel):
    artifact_uri: str = Field(min_length=1)
    run_id: Optional[str] = None
    stage: str = Field(default="staging")


def _wrap(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.get("/models")
async def registry_list_models(
    clinic_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {"status": "success", "mlair": mlair_client.registry_list_models(clinic_id=clinic_id, limit=limit, offset=offset)}
    except Exception as exc:
        raise _wrap(exc) from exc


@router.post("/models")
async def registry_create_model_post(
    body: CreateModelBody,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_create_model(
                clinic_id=clinic_id, name=body.name, description=body.description
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/models/{model_id}")
async def registry_get_model(
    model_id: str,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {"status": "success", "mlair": mlair_client.registry_get_model(clinic_id=clinic_id, model_id=model_id)}
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/models/{model_id}/status")
async def registry_get_model_status(
    model_id: str,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_get_model_status(clinic_id=clinic_id, model_id=model_id),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/models/{model_id}/resolved-pipeline")
async def registry_get_resolved_pipeline(
    model_id: str,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_get_resolved_pipeline(clinic_id=clinic_id, model_id=model_id),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.put("/models/{model_id}/pipeline-mapping")
async def registry_put_pipeline_mapping(
    model_id: str,
    body: PipelineMappingBody,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_put_pipeline_mapping(
                clinic_id=clinic_id,
                model_id=model_id,
                pipeline_id=body.pipeline_id,
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/models/{model_id}/trigger-policy")
async def registry_get_trigger_policy(
    model_id: str,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_get_trigger_policy(clinic_id=clinic_id, model_id=model_id),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.put("/models/{model_id}/trigger-policy")
async def registry_put_trigger_policy(
    model_id: str,
    body: TriggerPolicyBody,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_put_trigger_policy(
                clinic_id=clinic_id,
                model_id=model_id,
                trigger_mode=body.trigger_mode,
                debounce_minutes=body.debounce_minutes,
                schedule_cron=body.schedule_cron,
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/models/{model_id}/versions")
async def registry_list_versions(
    model_id: str,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {"status": "success", "mlair": mlair_client.registry_list_model_versions(clinic_id=clinic_id, model_id=model_id)}
    except Exception as exc:
        raise _wrap(exc) from exc


@router.post("/models/{model_id}/versions")
async def registry_create_version_post(
    model_id: str,
    body: CreateModelVersionBody,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_create_model_version(
                clinic_id=clinic_id,
                model_id=model_id,
                artifact_uri=body.artifact_uri,
                run_id=body.run_id,
                stage=body.stage,
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/models/{model_id}/next-artifact-uri")
async def registry_next_artifact_uri(
    model_id: str,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_preview_next_artifact_uri(clinic_id=clinic_id, model_id=model_id),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.post("/models/{model_id}/promote")
async def registry_promote(
    model_id: str,
    body: PromoteBody,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_promote_model_version(
                clinic_id=clinic_id,
                model_id=model_id,
                version=body.version,
                stage=body.stage,
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/models/{model_id}/versions/{version}/approval")
async def registry_get_approval(
    model_id: str,
    version: int,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_get_version_approval(
                clinic_id=clinic_id, model_id=model_id, version=version
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.put("/models/{model_id}/versions/{version}/approval")
async def registry_put_approval(
    model_id: str,
    version: int,
    body: ApprovalBody,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_put_version_approval(
                clinic_id=clinic_id,
                model_id=model_id,
                version=version,
                approval_status=body.approval_status,
                reason=body.reason,
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.delete("/models/{model_id}/versions/{version}")
async def registry_delete_version(
    model_id: str,
    version: int,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "mlair": mlair_client.registry_delete_model_version(
                clinic_id=clinic_id, model_id=model_id, version=version
            ),
        }
    except Exception as exc:
        raise _wrap(exc) from exc


@router.delete("/models/{model_id}")
async def registry_delete_model(
    model_id: str,
    clinic_id: Optional[str] = Query(None),
    _: bool = Depends(require_admin),
) -> dict[str, Any]:
    try:
        return {"status": "success", "mlair": mlair_client.registry_delete_model(clinic_id=clinic_id, model_id=model_id)}
    except Exception as exc:
        raise _wrap(exc) from exc


@router.get("/plugins")
async def registry_list_plugins(_: bool = Depends(require_admin)) -> dict[str, Any]:
    try:
        return {"status": "success", "mlair": mlair_client.registry_list_plugins()}
    except Exception as exc:
        raise _wrap(exc) from exc
