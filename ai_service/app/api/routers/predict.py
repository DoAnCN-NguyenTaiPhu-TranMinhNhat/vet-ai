import logging
import os
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter
from prometheus_client import Gauge, Info
from scipy import sparse

from ai_service.app.core.metrics import observe_inference, timing
from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
from ai_service.app.domain.schemas.predict import PredictRequest, parse_symptoms
from ai_service.app.infrastructure.storage.model_store import (
    ActiveModel,
    detect_default_model,
    get_active_model_for_clinic,
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
_artifact_cache: dict[str, tuple[Any, Any, Any]] = {}

_active_model_info = Info("vetai_active_model", "Active model version for inference")
_active_model_reload_total = Gauge("vetai_active_model_reload_total", "Number of active model reloads")


def _cache_key(model_dir: str) -> str:
    return os.path.normpath(os.path.abspath(model_dir))


def clear_artifact_cache() -> None:
    _artifact_cache.clear()


def load_artifacts_for_dir(model_dir: str, model_version: str) -> tuple[Any, Any, Any]:
    key = _cache_key(model_dir)
    if key not in _artifact_cache:
        _artifact_cache[key] = (
            joblib.load(os.path.join(model_dir, "model.pkl")),
            joblib.load(os.path.join(model_dir, "tab_preprocess.pkl")),
            joblib.load(os.path.join(model_dir, "symptoms_mlb.pkl")),
        )
        logger.info("Cached model artifacts: version=%s dir=%s", model_version, key)
    return _artifact_cache[key]


def load_artifacts() -> None:
    global _model, _tab_preprocess, _symptoms_mlb
    _model, _tab_preprocess, _symptoms_mlb = load_artifacts_for_dir(MODEL_DIR, MODEL_VERSION)
    _active_model_info.info({"model_version": str(MODEL_VERSION)})


def set_active_model_and_reload(model_version: str) -> None:
    from ai_service.app.infrastructure.storage.model_store import set_active_model

    global MODEL_VERSION, MODEL_DIR, _model, _tab_preprocess, _symptoms_mlb
    active = set_active_model(model_version)
    MODEL_VERSION = active.model_version
    MODEL_DIR = active.model_dir
    clear_artifact_cache()
    _model = None
    _tab_preprocess = None
    _symptoms_mlb = None
    load_artifacts()
    _active_model_reload_total.inc()


def model_info() -> dict[str, Any]:
    if _model is None:
        return {"loaded": False, "model_dir": MODEL_DIR, "model_version": MODEL_VERSION}
    classes = getattr(_model, "classes_", None)
    return {
        "loaded": True,
        "model_dir": MODEL_DIR,
        "model_version": MODEL_VERSION,
        "model_type": type(_model).__name__,
        "classes": classes.tolist() if isinstance(classes, np.ndarray) else classes,
    }


def _clinic_id_for_model_router(cid_raw: Any) -> str | None:
    return normalize_clinic_key(cid_raw)


def _clinic_id_for_prediction_log(cid_raw: Any) -> str | None:
    if cid_raw is None:
        return None
    value = str(cid_raw).strip()
    return value or None


@router.post("/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    if _model is None:
        load_artifacts()

    cid = _clinic_id_for_model_router(req.clinic_id)
    clinic_log = _clinic_id_for_prediction_log(req.clinic_id)
    payload = req.model_dump()
    if req.clinic_id is not None:
        payload["clinic_id"] = req.clinic_id

    active = get_active_model_for_clinic(cid) or ActiveModel(str(MODEL_VERSION), MODEL_DIR)
    model, tab_preprocess, symptoms_mlb = load_artifacts_for_dir(active.model_dir, active.model_version)

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
        pred = model.predict(x_final)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_final)[0]
            classes = model.classes_
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
            prediction_output={"diagnosis": str(pred), "confidence": confidence, "top_k": top_k},
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
        "predictions": top_k or [],
        "predictionId": str(prediction_id),
        "clinicId": clinic_log,
    }
