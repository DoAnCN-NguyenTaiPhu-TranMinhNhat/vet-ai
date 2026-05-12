"""
Intermediate-state persistence for multi-phase training pipelines.

Each MLAir pipeline run gets a session directory under SESSION_ROOT (default
``/tmp/training_sessions/<session_id>/``).  Phase functions save/load numpy
arrays and sklearn objects with joblib, and plain dicts with JSON.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

SESSION_ROOT = os.getenv("TRAINING_SESSION_ROOT", "/tmp/training_sessions")


def _session_dir(session_id: str) -> Path:
    return Path(SESSION_ROOT) / session_id


def create_session(session_id: str) -> Path:
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_exists(session_id: str) -> bool:
    return _session_dir(session_id).is_dir()


def save_object(session_id: str, key: str, obj: Any) -> None:
    d = create_session(session_id)
    joblib.dump(obj, d / f"{key}.joblib")


def load_object(session_id: str, key: str) -> Any:
    p = _session_dir(session_id) / f"{key}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Session object not found: {p}")
    return joblib.load(p)


def save_json(session_id: str, key: str, data: Any) -> None:
    d = create_session(session_id)
    with open(d / f"{key}.json", "w") as f:
        json.dump(data, f, default=str)


def load_json(session_id: str, key: str) -> Any:
    p = _session_dir(session_id) / f"{key}.json"
    if not p.exists():
        raise FileNotFoundError(f"Session JSON not found: {p}")
    with open(p) as f:
        return json.load(f)


def cleanup_session(session_id: str) -> None:
    d = _session_dir(session_id)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
        logger.info("Cleaned up training session %s", session_id)


# ── Convenience wrappers for training phases ────────────────────────────


def save_data_prep_results(
    session_id: str,
    *,
    X_processed: Any,
    y: Any,
    sample_weights: Any,
    sample_timestamps: Any,
    preprocessing_info: Dict[str, Any],
    split_seed: int,
    trainer: Any,
    eligible_feedback_data: List[Dict],
    prediction_logs: List[Dict],
    finetune: bool,
    finetune_base_model_dir: Optional[str],
    clinic_key: Optional[str],
    training_id: int,
    training_mode: str,
    dataset_window_days: int,
) -> None:
    save_object(session_id, "X_processed", X_processed)
    save_object(session_id, "y", y)
    save_object(session_id, "sample_weights", sample_weights)
    save_object(session_id, "sample_timestamps", sample_timestamps)
    save_object(session_id, "trainer", trainer)
    save_json(session_id, "preprocessing_info", preprocessing_info)
    save_json(session_id, "eligible_feedback_data", eligible_feedback_data)
    save_json(session_id, "prediction_logs", prediction_logs)
    save_json(session_id, "session_meta", {
        "split_seed": split_seed,
        "finetune": finetune,
        "finetune_base_model_dir": finetune_base_model_dir,
        "clinic_key": clinic_key,
        "training_id": training_id,
        "training_mode": training_mode,
        "dataset_window_days": dataset_window_days,
    })


def load_data_prep_results(session_id: str) -> Dict[str, Any]:
    meta = load_json(session_id, "session_meta")
    return {
        "X_processed": load_object(session_id, "X_processed"),
        "y": load_object(session_id, "y"),
        "sample_weights": load_object(session_id, "sample_weights"),
        "sample_timestamps": load_object(session_id, "sample_timestamps"),
        "preprocessing_info": load_json(session_id, "preprocessing_info"),
        "trainer": load_object(session_id, "trainer"),
        "eligible_feedback_data": load_json(session_id, "eligible_feedback_data"),
        "prediction_logs": load_json(session_id, "prediction_logs"),
        **meta,
    }


def save_model_train_results(
    session_id: str,
    *,
    model: Any,
    training_metrics: Dict[str, Any],
) -> None:
    save_object(session_id, "model", model)
    save_json(session_id, "training_metrics", training_metrics)


def load_model_train_results(session_id: str) -> Dict[str, Any]:
    return {
        "model": load_object(session_id, "model"),
        "training_metrics": load_json(session_id, "training_metrics"),
    }


def save_validation_results(
    session_id: str,
    *,
    training_metrics: Dict[str, Any],
) -> None:
    save_json(session_id, "training_metrics", training_metrics)
