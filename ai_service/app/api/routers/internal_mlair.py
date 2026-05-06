"""
Inbound MLAir callbacks (executor → Vet-AI).

Configure MLAir with the same ``MLAIR_MODEL_PROMOTE_WEBHOOK_BEARER_TOKEN`` as this service
so ``Authorization: Bearer …`` matches.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
from ai_service.app.infrastructure.storage import model_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/mlair", tags=["internal-mlair"])


async def verify_mlair_promote_webhook(authorization: str | None = Header(None)) -> bool:
    expected = (os.getenv("MLAIR_MODEL_PROMOTE_WEBHOOK_BEARER_TOKEN") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="MLAIR_MODEL_PROMOTE_WEBHOOK_BEARER_TOKEN is not set on Vet-AI",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization")
    token = authorization[7:].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


class ModelPromotionWebhookIn(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    project_id: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    version: int = Field(..., ge=0)
    artifact_uri: str = Field(..., min_length=1)
    idempotency_key: str | None = None


def _version_label_for_promotion(model_id: str, version: int) -> str:
    mid = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(model_id).strip())[:48]
    return f"v_mlair_{mid}_n{int(version)}"


def _clinic_key_from_project_id(project_id: str) -> str | None:
    p = str(project_id).strip()
    if p.startswith("clinic_"):
        rest = p[len("clinic_") :].strip()
        return rest or None
    return None


def _promotion_mode() -> str:
    """
    - ``materialize`` (default): create ``v_mlair_*`` symlink + set active (legacy).
    - ``reuse_only``: do not create files; set active only if ``artifact_uri`` realpath matches
      an existing ``v*`` folder under ``MODEL_ROOT`` (training / manual deploy).
    """
    return (os.getenv("VETAI_MLAIR_PROMOTION_MODE") or "materialize").strip().lower()


@router.post("/model-promotion")
async def mlair_model_promotion_webhook(
    payload: ModelPromotionWebhookIn,
    _ok: bool = Depends(verify_mlair_promote_webhook),
) -> dict:
    """
    MLAir promote webhook: optionally materialize ``file://`` weights, then set Vet-AI active.

    When ``VETAI_MLAIR_PROMOTION_MODE=reuse_only``, no new directories or symlinks are created;
    the artifact path must already match an on-disk version folder (see ``find_version_label_for_artifact_realpath``).
    MLAir does not need code changes for either mode — only this env and/or webhook URL configuration.
    """
    uri = str(payload.artifact_uri).strip()
    if not uri.startswith("file://"):
        raise HTTPException(
            status_code=400,
            detail="Only file:// artifact_uri is supported for Vet-AI promotion",
        )

    raw_clinic = _clinic_key_from_project_id(payload.project_id)
    clinic_key = normalize_clinic_key(raw_clinic) if raw_clinic else None
    mode = _promotion_mode()

    if mode == "reuse_only":
        src_path = os.path.realpath(urlparse(uri).path)
        if not os.path.isdir(src_path):
            raise HTTPException(status_code=400, detail=f"artifact_not_a_directory:{src_path}")
        version_label = model_store.find_version_label_for_artifact_realpath(src_path, clinic_key)
        if not version_label:
            raise HTTPException(
                status_code=409,
                detail=(
                    "VETAI_MLAIR_PROMOTION_MODE=reuse_only: no existing version folder under MODEL_ROOT "
                    f"matches this artifact ({src_path}). Use training/UI to deploy into a v* folder, "
                    "or set VETAI_MLAIR_PROMOTION_MODE=materialize."
                ),
            )
        logger.info(
            "mlair_promotion_reuse_only model_id=%s mlair_version=%s clinic=%s vetai_version=%s path=%s",
            payload.model_id,
            payload.version,
            clinic_key,
            version_label,
            src_path,
        )
    elif mode == "materialize":
        version_label = _version_label_for_promotion(payload.model_id, payload.version)
        try:
            version_label = model_store.materialize_mlair_artifact_as_version(
                artifact_uri=uri,
                version_label=version_label,
                clinic_key=clinic_key,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Invalid VETAI_MLAIR_PROMOTION_MODE={mode!r}; use materialize or reuse_only",
        )

    if clinic_key is not None:
        model_store.set_clinic_active_model(clinic_key, version_label)
        try:
            from ai_service.app.api.routers.predict import clear_artifact_cache

            clear_artifact_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("clear_artifact_cache after clinic promote failed: %s", exc)
        logger.info(
            "mlair_promotion_applied clinic=%s version=%s model_id=%s",
            clinic_key,
            version_label,
            payload.model_id,
        )
    else:
        from ai_service.app.api.routers.predict import set_active_model_and_reload

        set_active_model_and_reload(version_label)
        logger.info(
            "mlair_promotion_applied global version=%s model_id=%s",
            version_label,
            payload.model_id,
        )

    return {
        "status": "ok",
        "model_version": version_label,
        "project_id": payload.project_id,
        "idempotency_key": payload.idempotency_key,
    }
