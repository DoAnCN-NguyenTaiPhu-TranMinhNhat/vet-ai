import logging
import os
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from prometheus_client import Gauge, Info
from scipy import sparse

from ai_service.app.core.metrics import observe_inference, timing
from ai_service.app.domain.services.clinic_catalog_service import resolve_clinic_identifier
from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
from ai_service.app.domain.schemas.predict import PredictRequest, parse_symptoms
from ai_service.app.infrastructure.storage.model_store import (
    ActiveModel,
    active_model_for_predict,
    detect_default_model,
    display_label_for_model_version,
    find_primary_model_pkl,
    get_active_model_for_clinic,
    get_clinic_pinned_model,
    list_user_visible_model_versions,
    storage_scope_for_version,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])

_active = detect_default_model()
if _active is None:
    MODEL_VERSION = os.getenv("MODEL_VERSION") or "v2.0"
    MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join("./ai_service/models", MODEL_VERSION)
else:
    MODEL_VERSION = _active.model_version
    MODEL_DIR = _active.model_dir
    logger.info("Using active model: %s (%s)", MODEL_VERSION, MODEL_DIR)

_model: Any = None
_tab_preprocess: Any = None
_symptoms_mlb: Any = None
_calibrated: Any = None
_artifact_cache: dict[str, tuple[Any, Any, Any, Any]] = {}

_active_model_info = Info("vetai_active_model", "Active model version for inference")
_active_model_reload_total = Gauge("vetai_active_model_reload_total", "Number of active model reloads")


def _cache_key(model_dir: str) -> str:
    return os.path.normpath(os.path.abspath(model_dir))


def clear_artifact_cache() -> None:
    _artifact_cache.clear()


def load_artifacts_for_dir(model_dir: str, model_version: str) -> tuple[Any, Any, Any, Any]:
    key = _cache_key(model_dir)
    if key not in _artifact_cache:
        cal_path = os.path.join(model_dir, "calibrated_classifier.pkl")
        calibrated = joblib.load(cal_path) if os.path.exists(cal_path) else None
        mp = find_primary_model_pkl(model_dir)
        if mp is None:
            raise FileNotFoundError(f"No model pickle under {model_dir} (expected model.pkl or .model.pkl)")
        _artifact_cache[key] = (
            joblib.load(str(mp)),
            joblib.load(os.path.join(model_dir, "tab_preprocess.pkl")),
            joblib.load(os.path.join(model_dir, "symptoms_mlb.pkl")),
            calibrated,
        )
        logger.info("Cached model artifacts: version=%s dir=%s", model_version, key)
    return _artifact_cache[key]


def load_artifacts() -> None:
    global _model, _tab_preprocess, _symptoms_mlb, _calibrated
    _model, _tab_preprocess, _symptoms_mlb, _calibrated = load_artifacts_for_dir(MODEL_DIR, MODEL_VERSION)
    _active_model_info.info({"model_version": str(MODEL_VERSION)})


def set_active_model_and_reload(model_version: str) -> None:
    from ai_service.app.infrastructure.storage.model_store import set_active_model

    global MODEL_VERSION, MODEL_DIR, _model, _tab_preprocess, _symptoms_mlb, _calibrated
    active = set_active_model(model_version)
    MODEL_VERSION = active.model_version
    MODEL_DIR = active.model_dir
    clear_artifact_cache()
    _model = None
    _tab_preprocess = None
    _symptoms_mlb = None
    _calibrated = None
    load_artifacts()
    _active_model_reload_total.inc()


def model_info() -> dict[str, Any]:
    if _model is None:
        return {"loaded": False, "model_dir": MODEL_DIR, "model_version": MODEL_VERSION}
    infer = _calibrated if _calibrated is not None else _model
    classes = getattr(infer, "classes_", None)
    return {
        "loaded": True,
        "model_dir": MODEL_DIR,
        "model_version": MODEL_VERSION,
        "model_type": type(infer).__name__,
        "classes": classes.tolist() if isinstance(classes, np.ndarray) else classes,
    }


def _clinic_id_for_model_router(cid_raw: Any) -> str | None:
    if cid_raw is None:
        return None
    resolved = resolve_clinic_identifier(str(cid_raw).strip())
    return normalize_clinic_key(resolved)


def _clinic_id_for_prediction_log(cid_raw: Any) -> str | None:
    if cid_raw is None:
        return None
    value = str(cid_raw).strip()
    return value or None


@router.get("/predict/models", summary="List model versions usable for POST /predict (optional clinic)")
async def list_predict_models(
    clinic_id: str | int | None = Query(default=None, alias="clinicId"),
) -> dict[str, Any]:
    """
    Versions are those returned by ``list_user_visible_model_versions`` for the clinic key
    (merged global + clinic folder when a clinic is set).

    Each row includes ``label`` (clinic scope: global-origin versions show ``… - global``),
    ``storageScope``, and ``isActiveDefault`` so UIs can badge the active model.
    """
    ck = _clinic_id_for_model_router(clinic_id)
    fallback = detect_default_model() or ActiveModel(str(MODEL_VERSION), MODEL_DIR)
    effective = get_active_model_for_clinic(ck) or fallback
    versions = list_user_visible_model_versions(ck)
    if not versions:
        fb = detect_default_model()
        if fb and str(fb.model_version or "").strip() and os.path.isdir(fb.model_dir):
            versions = [str(fb.model_version).strip()]
    eff_ver = (effective.model_version or "").strip()

    if ck:
        pinned = get_clinic_pinned_model(ck)
        active_source = "clinic_pin" if pinned is not None else "global_fallback"
    else:
        active_source = "global"

    models = []
    for v in versions:
        scope = storage_scope_for_version(ck, v)
        lbl = display_label_for_model_version(ck, v, scope)
        models.append(
            {
                "version": v,
                "storageScope": scope,
                "label": lbl,
                "displayLabel": lbl,
                "isActiveDefault": v == eff_ver,
            }
        )
    eff_scope = storage_scope_for_version(ck, eff_ver) if eff_ver else "global"
    default_lbl = (
        display_label_for_model_version(ck, eff_ver, eff_scope)
        if eff_ver
        else (effective.model_version or None)
    )
    return {
        "clinicId": str(clinic_id).strip() if clinic_id is not None and str(clinic_id).strip() else None,
        "clinicKey": ck,
        "defaultModelVersion": effective.model_version,
        "defaultModelLabel": default_lbl,
        "defaultModelScope": storage_scope_for_version(ck, effective.model_version),
        "activeSource": active_source,
        "models": models,
    }


@router.post("/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    if _model is None:
        load_artifacts()

    cid = _clinic_id_for_model_router(req.clinic_id)
    clinic_log = _clinic_id_for_prediction_log(req.clinic_id)
    payload = req.model_dump()
    if req.clinic_id is not None:
        payload["clinic_id"] = req.clinic_id

    fallback = ActiveModel(str(MODEL_VERSION), MODEL_DIR)
    try:
        active, explicit_override = active_model_for_predict(
            cid,
            req.model_version,
            fallback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload["inference_model_version"] = active.model_version
    payload["inference_model_explicit_override"] = explicit_override
    payload["inference_model_scope"] = storage_scope_for_version(cid, active.model_version)

    model, tab_preprocess, symptoms_mlb, calibrated = load_artifacts_for_dir(
        active.model_dir, active.model_version
    )
    infer = calibrated if calibrated is not None else model

    tabular = {
        "animal_type": payload["animal_type"],
        "gender": payload["gender"],
        "age_months": payload["age_months"],
        "weight_kg": payload["weight_kg"],
        "temperature": payload["temperature"],
        "heart_rate": payload["heart_rate"],
        "current_season": payload["current_season"],
        "vaccination_status": payload["vaccination_status"],
        "medical_history": payload.get("medical_history") or "Unknown",
        "symptom_duration": payload["symptom_duration"],
    }
    x_tab = pd.DataFrame([tabular])
    x_tab_p = tab_preprocess.transform(x_tab)
    x_sym = symptoms_mlb.transform([parse_symptoms(payload["symptoms_list"])])
    x_final = sparse.hstack([x_tab_p, sparse.csr_matrix(x_sym)]).tocsr()

    confidence = None
    top_k = None
    with timing() as timer:
        pred = infer.predict(x_final)[0]
        if hasattr(infer, "predict_proba"):
            proba = infer.predict_proba(x_final)[0]
            classes = infer.classes_
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            top_idx = np.argsort(proba)[::-1][: min(3, len(classes))]
            top_k = [{"label": str(classes[i]), "prob": float(proba[i])} for i in top_idx]

    try:
        observe_inference(clinic_log, ok=True, latency_seconds=timer.elapsed, confidence=confidence)
    except Exception:
        logger.exception("Failed to record inference metrics")

    prediction_id = uuid.uuid4()
    pet_key = (payload.get("pet_id") or "").strip() or "00000000-0000-0000-0000-000000000000"
    visit_key = payload.get("visit_id")

    try:
        from ai_service.app.domain.services.training_service import get_prediction_log_model
        PredictionLog = get_prediction_log_model()
        from ai_service.app.domain.services.training_service import log_prediction_entry

        prediction_log = PredictionLog(
            id=prediction_id,
            visit_id=visit_key,
            pet_id=pet_key,
            prediction_input=payload,
            prediction_output={
                "diagnosis": str(pred),
                "confidence": confidence,
                "top_k": top_k,
                "model_version": active.model_version,
                "model_scope": storage_scope_for_version(cid, active.model_version),
                "explicit_model_override": explicit_override,
            },
            model_version=active.model_version,
            confidence_score=confidence if confidence is not None else 0.0,
            top_k_predictions=top_k or [],
            veterinarian_id=None,
            clinic_id=clinic_log,
        )
        await log_prediction_entry(prediction_log)
        logger.info("Prediction logged successfully with ID: %s", prediction_id)
    except Exception as exc:
        logger.warning("Failed to log prediction: %s", exc)

    return {
        "diagnosis": str(pred),
        "confidence": confidence,
        "top_k": top_k,
        "modelVersion": active.model_version,
        "modelScope": storage_scope_for_version(cid, active.model_version),
        "explicitModelVersion": explicit_override,
        "predictions": top_k or [],
        "predictionId": str(prediction_id),
        "clinicId": clinic_log,
    }
