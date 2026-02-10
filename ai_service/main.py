import os
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from scipy import sparse

# Configure logging
logger = logging.getLogger(__name__)

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
MODEL_VERSION = os.getenv("MODEL_VERSION")
MODEL_DIR = os.getenv("MODEL_DIR")

# Auto-detect latest model if not specified
if MODEL_DIR is None or not str(MODEL_DIR).strip():
    if MODEL_VERSION is not None and str(MODEL_VERSION).strip():
        MODEL_DIR = os.path.join("./ai_service/models", str(MODEL_VERSION).strip())
    else:
        # Auto-detect latest version
        latest_version = get_latest_model_version()
        MODEL_DIR = os.path.join("./ai_service/models", latest_version)
        MODEL_VERSION = latest_version
        logger.info(f"Auto-detected latest model version: {latest_version}")

_model = None
_tab_preprocess = None
_symptoms_mlb = None


def load_artifacts() -> None:
    global _model, _tab_preprocess, _symptoms_mlb

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    tab_preprocess_path = os.path.join(MODEL_DIR, "tab_preprocess.pkl")
    symptoms_mlb_path = os.path.join(MODEL_DIR, "symptoms_mlb.pkl")

    _model = joblib.load(model_path)
    _tab_preprocess = joblib.load(tab_preprocess_path)
    _symptoms_mlb = joblib.load(symptoms_mlb_path)


@app.on_event("startup")
def _startup() -> None:
    load_artifacts()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
