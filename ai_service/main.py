import os
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import Request, Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from scipy import sparse
import contextvars
import time
import uuid

from ai_service.model_registry import detect_default_model, list_model_versions, set_active_model
from prometheus_client import Gauge, Info
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Request ID (X-Request-Id) correlation for logs
_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class _RequestIdLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = _request_id_ctx.get()
        setattr(record, "request_id", rid or "-")
        return True


logging.getLogger().addFilter(_RequestIdLogFilter())

# Import continuous training endpoints
try:
    from ai_service.continuous_training import router as ct_router
except ImportError:
    # Fallback for direct execution/testing
    try:
        from continuous_training import router as ct_router
    except ImportError:
        print("Warning: continuous_training module not found")
        ct_router = None

# Import MLOps endpoints
try:
    from ai_service.mlops_api import router as mlops_router
except ImportError:
    # Fallback for direct execution/testing
    try:
        from mlops_api import router as mlops_router
    except ImportError:
        print("Warning: mlops_api module not found")
        mlops_router = None

# Import MLOps v2 endpoints (Champion-Challenger)
try:
    from ai_service.mlops_api_v2 import router as mlops_v2_router
except ImportError:
    # Fallback for direct execution/testing
    try:
        from mlops_api_v2 import router as mlops_v2_router
    except ImportError:
        print("Warning: mlops_api_v2 module not found")
        mlops_v2_router = None


def parse_symptoms(s: Any) -> list[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    items = [x.strip().lower() for x in s.split(",")]
    return [x for x in items if x]


class PredictRequest(BaseModel):
    animal_type: str
    gender: str
    age_months: int
    weight_kg: float
    temperature: float
    heart_rate: int
    current_season: str
    vaccination_status: str
    medical_history: str | None = "Unknown"
    symptoms_list: str
    symptom_duration: int


app = FastAPI(
    title="Veterinary Diagnosis AI",
    version="2.0.0",
    description="Veterinary AI Diagnosis System with Champion-Challenger MLOps"
)

_UI_DIR = Path(__file__).parent / "ui"
_UI_STATIC_DIR = _UI_DIR / "static"
if _UI_STATIC_DIR.exists():
    app.mount("/mlops-ui/static", StaticFiles(directory=str(_UI_STATIC_DIR)), name="mlops-ui-static")


@app.get("/mlops-ui", include_in_schema=False)
def mlops_ui() -> HTMLResponse:
    index = _UI_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h3>UI not found</h3>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))

# Ensure every request has X-Request-Id and expose it back.
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    token = _request_id_ctx.set(rid)
    start = time.time()
    try:
        response: Response = await call_next(request)
    finally:
        _request_id_ctx.reset(token)
    response.headers["X-Request-Id"] = rid
    logger.info("request completed", extra={"path": request.url.path, "method": request.method, "ms": int((time.time() - start) * 1000)})
    return response

# Expose Prometheus metrics for AI service
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

# Include continuous training endpoints if available
if ct_router:
    app.include_router(ct_router)

# Include MLOps endpoints if available
if mlops_router:
    app.include_router(mlops_router)

# Include MLOps v2 endpoints (Champion-Challenger) if available
if mlops_v2_router:
    app.include_router(mlops_v2_router)


def get_latest_model_version() -> str:
    """Get the latest model version from models directory"""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(models_dir):
            return "v2.0"  # fallback
        
        versions = []
        for item in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, item)):
                if item.startswith('v') and (item[1:].replace('.', '').replace('_', '').isdigit()):
                    versions.append(item)
        
        if versions:
            # Sort by version - prefer timestamped versions, then semantic versions
            timestamped_versions = [v for v in versions if '_' in v]
            semantic_versions = [v for v in versions if '_' not in v]
            
            if timestamped_versions:
                return max(timestamped_versions)
            elif semantic_versions:
                return max(semantic_versions, key=lambda x: float(x[1:]))
        
        return "v2.0"  # fallback
    except Exception:
        return "v2.0"  # fallback


_DEFAULT_MODEL_DIR = "./ai_service/models/v2"
_active = detect_default_model()
if _active is None:
    MODEL_VERSION = os.getenv("MODEL_VERSION") or "v2.0"
    MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join("./ai_service/models", MODEL_VERSION)
else:
    MODEL_VERSION = _active.model_version
    MODEL_DIR = _active.model_dir
    logger.info("Using active model: %s (%s)", MODEL_VERSION, MODEL_DIR)

_model = None
_tab_preprocess = None
_symptoms_mlb = None

_active_model_info = Info("vetai_active_model", "Active model version for inference")
_active_model_reload_total = Gauge("vetai_active_model_reload_total", "Number of active model reloads")


def load_artifacts() -> None:
    global _model, _tab_preprocess, _symptoms_mlb

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    tab_preprocess_path = os.path.join(MODEL_DIR, "tab_preprocess.pkl")
    symptoms_mlb_path = os.path.join(MODEL_DIR, "symptoms_mlb.pkl")

    _model = joblib.load(model_path)
    _tab_preprocess = joblib.load(tab_preprocess_path)
    _symptoms_mlb = joblib.load(symptoms_mlb_path)
    _active_model_info.info({"model_version": str(MODEL_VERSION)})


def set_active_model_and_reload(model_version: str) -> None:
    global MODEL_VERSION, MODEL_DIR, _model, _tab_preprocess, _symptoms_mlb
    active = set_active_model(model_version)
    MODEL_VERSION = active.model_version
    MODEL_DIR = active.model_dir
    # Clear cached artifacts then reload
    _model = None
    _tab_preprocess = None
    _symptoms_mlb = None
    load_artifacts()
    _active_model_reload_total.inc()


@app.on_event("startup")
def _startup() -> None:
    load_artifacts()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/readyz", include_in_schema=False)
def readyz() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/livez", include_in_schema=False)
def livez() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    if _model is None:
        return {
            "loaded": False,
            "model_dir": MODEL_DIR,
            "model_version": MODEL_VERSION,
        }

    classes = getattr(_model, "classes_", None)
    return {
        "loaded": True,
        "model_dir": MODEL_DIR,
        "model_version": MODEL_VERSION,
        "model_type": type(_model).__name__,
        "classes": classes.tolist() if isinstance(classes, np.ndarray) else classes,
    }


@app.get("/models/versions", include_in_schema=False)
def model_versions() -> dict[str, Any]:
    versions = list_model_versions()
    return {"active": MODEL_VERSION, "versions": versions}


@app.post("/models/active", include_in_schema=False)
def set_active_model_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    mv = str(payload.get("model_version") or "").strip()
    if not mv:
        return {"status": "error", "error": "model_version is required"}
    try:
        set_active_model_and_reload(mv)
        return {"status": "success", "active": MODEL_VERSION, "model_dir": MODEL_DIR}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Global counter for prediction IDs
_prediction_counter = 0

@app.post("/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    global _prediction_counter
    
    if _model is None:
        load_artifacts()

    payload = req.model_dump()

    # Prepare tabular row
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

    X_tab = pd.DataFrame([tabular])

    # Transform tabular
    X_tab_p = _tab_preprocess.transform(X_tab)

    # Transform symptoms
    X_sym = _symptoms_mlb.transform([parse_symptoms(payload["symptoms_list"])])

    X_final = sparse.hstack([X_tab_p, sparse.csr_matrix(X_sym)]).tocsr()

    pred = _model.predict(X_final)[0]

    confidence = None
    top_k = None

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X_final)[0]
        classes = _model.classes_
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])

        k = min(3, len(classes))
        top_idx = np.argsort(proba)[::-1][:k]
        top_k = [
            {"label": str(classes[i]), "prob": float(proba[i])}
            for i in top_idx
        ]

    # Increment prediction counter
    _prediction_counter += 1
    prediction_id = _prediction_counter

    # Log prediction for continuous training
    try:
        from ai_service.continuous_training import PredictionLog, log_prediction
        from datetime import datetime
        
        prediction_log = PredictionLog(
            id=prediction_id,  # Simple sequential integer
            visit_id=prediction_id,  # Same as prediction ID for simplicity
            pet_id=prediction_id,  # Same as prediction ID for simplicity
            prediction_input=payload,
            prediction_output={
                "diagnosis": str(pred),
                "confidence": confidence,
                "top_k": top_k
            },
            model_version=MODEL_VERSION,
            confidence_score=confidence,
            top_k_predictions=top_k or [],
            veterinarian_id=1,  # Default, should come from auth context
            clinic_id=1,
            timestamp=datetime.now()
        )
        
        await log_prediction(prediction_log)
        logger.info(f"Prediction logged successfully with ID: {prediction_id}")
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")

    return {
        "diagnosis": str(pred),
        "confidence": confidence,
        "top_k": top_k,
        "modelVersion": MODEL_VERSION,
        "predictions": top_k or [],
        "predictionId": prediction_id  # Return ID for feedback reference (camelCase for Java compatibility)
    }
