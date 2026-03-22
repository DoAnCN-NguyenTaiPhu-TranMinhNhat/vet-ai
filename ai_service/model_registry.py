import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveModel:
    model_version: str
    model_dir: str


_STATE_DIR = os.getenv("VETAI_STATE_DIR") or os.path.join(os.path.dirname(__file__), "state")
_ACTIVE_MODEL_FILE = os.path.join(_STATE_DIR, "active_model.json")


def _models_root() -> str:
    # Preferred: explicit root dir
    root = os.getenv("MODEL_ROOT_DIR") or os.getenv("VETAI_MODELS_ROOT")
    if root and str(root).strip():
        return str(root).strip()

    # Backward-compat: if MODEL_DIR points to a version directory, use its parent as root
    env_dir = os.getenv("MODEL_DIR")
    if env_dir and str(env_dir).strip():
        p = str(env_dir).strip().rstrip("/")
        # If it looks like a version folder containing model.pkl, treat parent as root
        try:
            if os.path.exists(os.path.join(p, "model.pkl")):
                return os.path.dirname(p)
        except Exception:
            pass
        return p

    # Default: ./ai_service/models by repo layout; in container this is /app/ai_service/models
    return os.path.join(os.path.dirname(__file__), "models")


def list_model_versions() -> List[str]:
    root = _models_root()
    if not os.path.isdir(root):
        return []
    versions: List[str] = []
    for item in os.listdir(root):
        path = os.path.join(root, item)
        if not os.path.isdir(path):
            continue
        if item.startswith("v"):
            versions.append(item)
    # newest-first: timestamped versions first, then lexical fallback
    timestamped = sorted([v for v in versions if "_" in v], reverse=True)
    semantic = sorted([v for v in versions if "_" not in v], reverse=True)
    return timestamped + semantic


def resolve_model_dir(model_version: str) -> str:
    return os.path.join(_models_root(), model_version)


def get_active_model() -> Optional[ActiveModel]:
    try:
        with open(_ACTIVE_MODEL_FILE, "r") as f:
            data = json.load(f)
        mv = str(data.get("model_version") or "").strip()
        if not mv:
            return None
        md = resolve_model_dir(mv)
        return ActiveModel(model_version=mv, model_dir=md)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Failed to read active model state: %s", e)
        return None


def set_active_model(model_version: str) -> ActiveModel:
    mv = str(model_version).strip()
    if not mv:
        raise ValueError("model_version is required")
    model_dir = resolve_model_dir(mv)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    os.makedirs(_STATE_DIR, exist_ok=True)
    tmp = _ACTIVE_MODEL_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"model_version": mv, "model_dir": model_dir}, f)
    os.replace(tmp, _ACTIVE_MODEL_FILE)

    logger.info("Active model set to %s (%s)", mv, model_dir)
    return ActiveModel(model_version=mv, model_dir=model_dir)


def detect_default_model() -> Optional[ActiveModel]:
    # Priority: persisted active model -> env MODEL_DIR/MODEL_VERSION -> latest on disk
    active = get_active_model()
    if active is not None:
        return active

    env_dir = os.getenv("MODEL_DIR")
    env_ver = os.getenv("MODEL_VERSION")
    if env_dir and str(env_dir).strip():
        # best-effort infer version name from path
        mv = str(env_ver).strip() if env_ver and str(env_ver).strip() else os.path.basename(str(env_dir).rstrip("/"))
        return ActiveModel(model_version=mv, model_dir=str(env_dir))

    versions = list_model_versions()
    if not versions:
        return None
    mv = versions[0]
    return ActiveModel(model_version=mv, model_dir=resolve_model_dir(mv))

