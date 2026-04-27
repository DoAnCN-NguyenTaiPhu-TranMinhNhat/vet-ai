import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveModel:
    model_version: str
    model_dir: str


_STATE_DIR = os.getenv("VETAI_STATE_DIR") or os.path.join(os.path.dirname(__file__), "state")
_ACTIVE_MODEL_FILE = os.path.join(_STATE_DIR, "active_model.json")


def _models_root() -> str:
    root = os.getenv("MODEL_ROOT_DIR") or os.getenv("VETAI_MODELS_ROOT")
    if root and str(root).strip():
        return str(root).strip()

    env_dir = os.getenv("MODEL_DIR")
    if env_dir and str(env_dir).strip():
        p = str(env_dir).strip().rstrip("/")
        try:
            if os.path.exists(os.path.join(p, "model.pkl")):
                return os.path.dirname(p)
        except Exception:
            pass
        return p
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")


def _collect_versions_under(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    out: List[str] = []
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path) and item.startswith("v"):
            out.append(item)
    return out


def list_model_versions(clinic_key: Optional[Union[str, int]] = None) -> List[str]:
    root = _models_root()
    seen = set()
    ordered: List[str] = []

    def add_from(versions: List[str]) -> None:
        for v in versions:
            if v not in seen:
                seen.add(v)
                ordered.append(v)

    add_from(_collect_versions_under(root))
    ck = normalize_clinic_key(clinic_key)
    if ck:
        sub = os.path.join(root, "clinics", clinic_dir_slug(ck))
        add_from(_collect_versions_under(sub))

    timestamped = sorted([v for v in ordered if "_" in v], reverse=True)
    semantic = sorted([v for v in ordered if "_" not in v], reverse=True)
    rest = [v for v in ordered if v not in timestamped and v not in semantic]
    merged = timestamped + semantic + rest
    seen_order: List[str] = []
    for v in merged:
        if v not in seen_order:
            seen_order.append(v)
    return seen_order


def list_all_model_versions() -> List[str]:
    """
    Return all discoverable model versions from global root and all clinic subfolders.
    """
    root = _models_root()
    seen = set()
    out: List[str] = []

    def _add(items: List[str]) -> None:
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)

    _add(_collect_versions_under(root))
    clinics_root = os.path.join(root, "clinics")
    if os.path.isdir(clinics_root):
        for clinic_slug in os.listdir(clinics_root):
            clinic_path = os.path.join(clinics_root, clinic_slug)
            if not os.path.isdir(clinic_path):
                continue
            _add(_collect_versions_under(clinic_path))

    timestamped = sorted([v for v in out if "_" in v], reverse=True)
    semantic = sorted([v for v in out if "_" not in v], reverse=True)
    rest = [v for v in out if v not in timestamped and v not in semantic]
    return timestamped + semantic + rest


def list_model_versions_with_scope() -> List[Dict[str, Optional[str]]]:
    """
    Return model versions with scope metadata.
    - clinic_key=None means global model root.
    - clinic_key=<id> means clinic-scoped model under models/clinics/<slug>/.
    """
    root = _models_root()
    rows: List[Dict[str, Optional[str]]] = []

    for version in _collect_versions_under(root):
        rows.append(
            {
                "version": version,
                "clinic_key": None,
                "model_dir": resolve_model_dir(version, None),
            }
        )

    clinics_root = os.path.join(root, "clinics")
    if os.path.isdir(clinics_root):
        for clinic_slug in os.listdir(clinics_root):
            clinic_path = os.path.join(clinics_root, clinic_slug)
            if not os.path.isdir(clinic_path):
                continue
            for version in _collect_versions_under(clinic_path):
                rows.append(
                    {
                        "version": version,
                        "clinic_key": clinic_slug,
                        "model_dir": os.path.join(clinic_path, version),
                    }
                )

    # newest-looking versions first while keeping scope grouping stable enough
    rows.sort(key=lambda r: str(r.get("version") or ""), reverse=True)
    return rows


def resolve_model_dir(model_version: str, clinic_key: Optional[Union[str, int]] = None) -> str:
    root = _models_root()
    ck = normalize_clinic_key(clinic_key)
    if ck:
        sub = os.path.join(root, "clinics", clinic_dir_slug(ck), model_version)
        if os.path.isdir(sub):
            return sub
    return os.path.join(root, model_version)


def get_active_model() -> Optional[ActiveModel]:
    try:
        with open(_ACTIVE_MODEL_FILE, "r") as f:
            data = json.load(f)
        mv = str(data.get("model_version") or "").strip()
        if not mv:
            return None
        md = resolve_model_dir(mv, None)
        return ActiveModel(model_version=mv, model_dir=md)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Failed to read active model state: %s", e)
        return None


def _verify_inference_artifacts(model_dir: str) -> None:
    import joblib

    required = ("model.pkl", "tab_preprocess.pkl", "symptoms_mlb.pkl")
    for name in required:
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            raise ValueError(f"Missing artifact {name} under {model_dir}")
    try:
        joblib.load(os.path.join(model_dir, "model.pkl"))
    except Exception as e:
        raise ValueError(f"model.pkl cannot be loaded from {model_dir}: {e}") from e


def set_active_model(model_version: str) -> ActiveModel:
    mv = str(model_version).strip()
    if not mv:
        raise ValueError("model_version is required")

    try:
        from ai_service.app.infrastructure.external.s3_client import ensure_model_directory_from_s3

        ok, err = ensure_model_directory_from_s3(mv, None)
        if not ok:
            raise FileNotFoundError(err or f"Model {mv} not found locally and could not be restored from S3")
    except ImportError:
        pass

    model_dir = resolve_model_dir(mv, None)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    _verify_inference_artifacts(model_dir)

    os.makedirs(_STATE_DIR, exist_ok=True)
    tmp = _ACTIVE_MODEL_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"model_version": mv, "model_dir": model_dir}, f)
    os.replace(tmp, _ACTIVE_MODEL_FILE)

    logger.info("Active model set to %s (%s)", mv, model_dir)
    return ActiveModel(model_version=mv, model_dir=model_dir)


def detect_default_model() -> Optional[ActiveModel]:
    active = get_active_model()
    if active is not None:
        return active

    env_dir = os.getenv("MODEL_DIR")
    env_ver = os.getenv("MODEL_VERSION")
    if env_dir and str(env_dir).strip():
        mv = str(env_ver).strip() if env_ver and str(env_ver).strip() else os.path.basename(str(env_dir).rstrip("/"))
        return ActiveModel(model_version=mv, model_dir=str(env_dir))

    versions = list_model_versions(None)
    if not versions:
        return None
    mv = versions[0]
    return ActiveModel(model_version=mv, model_dir=resolve_model_dir(mv, None))


def _clinic_active_path(clinic_key: str) -> str:
    return os.path.join(_STATE_DIR, "clinics", clinic_dir_slug(clinic_key), "active_model.json")


def get_active_model_for_clinic(clinic_id: Optional[Union[str, int]]) -> Optional[ActiveModel]:
    ck = normalize_clinic_key(clinic_id)
    if ck is None:
        return detect_default_model()

    path = _clinic_active_path(ck)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        mv = str(data.get("model_version") or "").strip()
        md = str(data.get("model_dir") or "").strip()
        if md and os.path.isdir(md):
            ver = mv or os.path.basename(md.rstrip("/"))
            return ActiveModel(model_version=ver, model_dir=md)
        if mv:
            d = resolve_model_dir(mv, ck)
            if os.path.isdir(d):
                return ActiveModel(model_version=mv, model_dir=d)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Clinic %s active model invalid: %s", ck, e)
    return detect_default_model()


def get_clinic_pinned_model(clinic_id: Union[str, int]) -> Optional[ActiveModel]:
    ck = normalize_clinic_key(clinic_id)
    if ck is None:
        return None
    path = _clinic_active_path(ck)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        mv = str(data.get("model_version") or "").strip()
        md = str(data.get("model_dir") or "").strip()
        if md and os.path.isdir(md):
            ver = mv or os.path.basename(md.rstrip("/"))
            return ActiveModel(model_version=ver, model_dir=md)
        if mv:
            d = resolve_model_dir(mv, ck)
            if os.path.isdir(d):
                return ActiveModel(model_version=mv, model_dir=d)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Clinic %s pinned model unreadable: %s", ck, e)
    return None


def set_clinic_active_model(clinic_id: Union[str, int], model_version: str) -> ActiveModel:
    ck = normalize_clinic_key(clinic_id)
    if ck is None:
        raise ValueError("clinic_id is required")
    mv = str(model_version).strip()
    if not mv:
        raise ValueError("model_version is required")

    try:
        from ai_service.app.infrastructure.external.s3_client import ensure_model_directory_from_s3

        ok, err = ensure_model_directory_from_s3(mv, ck)
        if not ok:
            raise FileNotFoundError(err or f"Model {mv} not found for clinic and could not be restored from S3")
    except ImportError:
        pass

    model_dir = resolve_model_dir(mv, ck)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    _verify_inference_artifacts(model_dir)

    out = _clinic_active_path(ck)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"model_version": mv, "model_dir": model_dir}, f)
    os.replace(tmp, out)
    logger.info("Clinic %s active model set to %s (%s)", ck, mv, model_dir)
    return ActiveModel(model_version=mv, model_dir=model_dir)
