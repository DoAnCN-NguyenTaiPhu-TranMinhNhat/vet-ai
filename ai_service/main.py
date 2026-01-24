import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from scipy import sparse

# Import continuous training endpoints
try:
    from .continuous_training import router as ct_router
except ImportError:
    # Fallback for direct execution/testing
    from continuous_training import router as ct_router

# Import MLOps endpoints
try:
    from .mlops_api import router as mlops_router
except ImportError:
    # Fallback for direct execution/testing
    from mlops_api import router as mlops_router


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


app = FastAPI(title="Veterinary Diagnosis AI", version="0.1.0")

# Include continuous training endpoints
app.include_router(ct_router)

# Include MLOps endpoints
app.include_router(mlops_router)


_DEFAULT_MODEL_DIR = "./ai_service/models/v2"
MODEL_VERSION = os.getenv("MODEL_VERSION")
MODEL_DIR = os.getenv("MODEL_DIR")
if MODEL_DIR is None or not str(MODEL_DIR).strip():
    if MODEL_VERSION is not None and str(MODEL_VERSION).strip():
        MODEL_DIR = os.path.join("./ai_service/models", str(MODEL_VERSION).strip())
    else:
        MODEL_DIR = _DEFAULT_MODEL_DIR

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


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
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

    return {
        "diagnosis": str(pred),
        "confidence": confidence,
        "top_k": top_k,
    }
