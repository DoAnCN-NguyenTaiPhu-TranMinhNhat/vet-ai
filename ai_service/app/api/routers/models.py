from typing import Any

from fastapi import APIRouter

from ai_service.app.api.routers import predict as predict_router
from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
from ai_service.app.infrastructure.external.mlair_client import try_sync_vetai_pin_to_mlair_production
from ai_service.app.infrastructure.storage.model_store import list_user_visible_model_versions, set_clinic_active_model

router = APIRouter(tags=["Models"])


@router.get("/model/info")
def model_info_endpoint() -> dict[str, Any]:
    return predict_router.model_info()


@router.get("/models/versions", include_in_schema=False)
def model_versions() -> dict[str, Any]:
    return {"active": predict_router.MODEL_VERSION, "versions": list_user_visible_model_versions(None)}


@router.post("/models/active", include_in_schema=False)
def set_active_model_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    model_version = str(payload.get("model_version") or "").strip()
    if not model_version:
        return {"status": "error", "error": "model_version is required"}
    try:
        predict_router.set_active_model_and_reload(model_version)
        try_sync_vetai_pin_to_mlair_production(None, predict_router.MODEL_VERSION)
        return {
            "status": "success",
            "active": predict_router.MODEL_VERSION,
            "model_dir": predict_router.MODEL_DIR,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@router.post("/models/clinic/{clinic_id}/active", include_in_schema=False)
def set_clinic_active_endpoint(clinic_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    model_version = str(payload.get("model_version") or "").strip()
    if not model_version:
        return {"status": "error", "error": "model_version is required"}
    try:
        set_clinic_active_model(clinic_id, model_version)
        predict_router.clear_artifact_cache()
        ck = normalize_clinic_key(clinic_id)
        try_sync_vetai_pin_to_mlair_production(ck, model_version)
        return {"status": "success", "clinic_id": clinic_id, "model_version": model_version}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
