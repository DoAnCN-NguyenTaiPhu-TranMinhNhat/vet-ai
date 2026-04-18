from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from mlflow.tracking import MlflowClient

from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key
from ai_service.app.infrastructure.storage.model_store import get_active_model, get_active_model_for_clinic

logger = logging.getLogger(__name__)

TAG_MODEL_VERSION = "vetai_model_version"
TAG_CLINIC = "vetai_clinic_id"
LEGACY_TAG_CLINIC = "clinic_id"


def mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").strip()


def experiment_name_for_scope(clinic_key: Optional[str]) -> str:
    base = os.getenv("MLFLOW_EXPERIMENT_NAME", "vet-ai-continuous-training").strip()
    ck = normalize_clinic_key(clinic_key)
    if ck is None:
        return base
    return f"{base}-clinic-{clinic_dir_slug(ck)}"


def _run_model_version(tags: Dict[str, str]) -> Optional[str]:
    mv = (tags.get(TAG_MODEL_VERSION) or tags.get("model_version") or "").strip()
    return mv or None


def _pick_latest_run(runs: List[Any]) -> Any:
    return max(runs, key=lambda r: int(r.info.start_time or 0))


def compare_latest_run_to_active(clinic_id: Optional[str] = None) -> Dict[str, Any]:
    ck = normalize_clinic_key(clinic_id)
    uri = mlflow_tracking_uri()
    exp_name = experiment_name_for_scope(ck)

    if ck is None:
        active = get_active_model()
    else:
        active = get_active_model_for_clinic(ck)
    active_version = active.model_version if active else None

    out: Dict[str, Any] = {
        "status": "ok",
        "mlflow_tracking_uri": uri,
        "experiment_name": exp_name,
        "scope": "clinic" if ck else "global",
        "clinic_id": ck,
        "active_model_version": active_version,
        "latest_run": None,
        "versions_match": None,
        "note": None,
    }

    try:
        client = MlflowClient(tracking_uri=uri)
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            out["status"] = "no_experiment"
            out["note"] = f"Experiment '{exp_name}' does not exist in MLflow yet (no training run has logged to it)."
            return out

        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
        if not runs:
            out["status"] = "no_runs"
            out["note"] = "No runs found in this experiment."
            return out

        latest = _pick_latest_run(runs)
        tags = dict(latest.data.tags or {})
        run_mv = _run_model_version(tags)

        tag_keys = (
            TAG_MODEL_VERSION,
            TAG_CLINIC,
            "vetai_training_id",
            "vetai_dataset_window_days",
            "vetai_model_source",
            LEGACY_TAG_CLINIC,
            "training_id",
        )
        out["latest_run"] = {
            "run_id": latest.info.run_id,
            "status": latest.info.status,
            "start_time_ms": latest.info.start_time,
            "artifact_uri": latest.info.artifact_uri,
            "tags": {k: tags[k] for k in tag_keys if k in tags},
            "inferred_model_version": run_mv,
        }

        if run_mv and active_version:
            out["versions_match"] = run_mv == active_version
        elif not run_mv:
            out["versions_match"] = None
            out["note"] = "Latest run is missing tag 'vetai_model_version' (likely from older runs before tag standardization)."
        else:
            out["versions_match"] = None
            out["note"] = "No active model is set in the registry for this scope."

        return out
    except Exception as e:
        logger.warning("MLflow alignment check failed: %s", e)
        out["status"] = "mlflow_error"
        out["note"] = str(e)
        return out
