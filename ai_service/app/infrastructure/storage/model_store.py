import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key

logger = logging.getLogger(__name__)


def find_primary_model_pkl(version_dir: Union[str, Path]) -> Optional[Path]:
    """
    Resolve the serialized classifier file inside a version directory.

    Supports:

    - ``model.pkl`` (normal)
    - ``.model.pkl`` (Unix dotfile / \"hidden\" name)
    - Any casing of ``model.pkl`` on case-sensitive volumes (e.g. ``Model.pkl``)
    """
    p = Path(version_dir)
    if not p.is_dir():
        return None
    for name in ("model.pkl", ".model.pkl"):
        c = p / name
        if c.is_file():
            return c
    try:
        for child in p.iterdir():
            if child.is_file() and child.name.lower() == "model.pkl":
                return child
    except OSError:
        pass
    return None


def list_user_visible_model_versions(clinic_key: Optional[Union[str, int]] = None) -> List[str]:
    """Same as :func:`list_model_versions` (all discovered ``v*`` dirs are user-selectable)."""
    return list_model_versions(clinic_key)


def list_user_visible_model_versions_clinic_storage_only(clinic_key: Optional[Union[str, int]] = None) -> List[str]:
    return list_model_versions_clinic_storage_only(clinic_key)


def _ordered_version_names(versions: List[str]) -> List[str]:
    timestamped = sorted([v for v in versions if "_" in v], reverse=True)
    semantic = sorted([v for v in versions if "_" not in v], reverse=True)
    rest = [v for v in versions if v not in timestamped and v not in semantic]
    merged = timestamped + semantic + rest
    out: List[str] = []
    seen: set[str] = set()
    for v in merged:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def list_models_for_clinic_project_view(clinic_key: Union[str, int]) -> List[Dict[str, Union[str, None]]]:
    """
    On-disk models for MLOps when a clinic is selected: clinic folder first, then global root.

    Each row has the real ``version`` folder name (for activate/train APIs), ``storageScope``,
    and ``label`` (global rows use ``"{version} - global"`` so the UI can distinguish them).
    """
    ck = normalize_clinic_key(clinic_key)
    if ck is None:
        return []
    root = _models_root()
    clinic_sub = os.path.join(root, "clinics", clinic_dir_slug(ck))
    clinic_versions = _ordered_version_names(_collect_versions_under(clinic_sub))
    global_versions = _ordered_version_names(_collect_versions_under(root))
    rows: List[Dict[str, Union[str, None]]] = []
    for v in clinic_versions:
        rows.append({"version": v, "storageScope": "clinic", "label": v, "displayLabel": v})
    for v in global_versions:
        lab = v if str(v).strip().endswith(" - global") else f"{v} - global"
        rows.append({"version": v, "storageScope": "global", "label": lab, "displayLabel": lab})
    return rows


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
        p = os.path.normpath(str(env_dir).strip().rstrip("/"))
        try:
            if find_primary_model_pkl(p) is not None:
                return os.path.dirname(str(p))
            # Compose often sets MODEL_DIR=/app/ai_service/models/v2 even before model.pkl exists.
            # Treat .../models/<v*> as a version directory so listing scans sibling v* under .../models.
            parent = os.path.dirname(p)
            base = os.path.basename(p)
            if (
                base.startswith("v")
                and os.path.isdir(parent)
                and os.path.basename(parent.rstrip(os.sep)) == "models"
            ):
                return parent
        except Exception:
            pass
        return p
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")


def _collect_versions_under(dir_path: str) -> List[str]:
    """
    Version folders under ``dir_path``: legacy names ``v*``, or any subdirectory that already
    contains ``model.pkl`` (supports non-v version names). Skips the reserved ``clinics`` folder
    when listing the global models root.
    """
    if not os.path.isdir(dir_path):
        return []
    root_norm = os.path.normpath(_models_root())
    parent_norm = os.path.normpath(dir_path)
    out: List[str] = []
    for item in os.listdir(dir_path):
        if item in (".", ".."):
            continue
        if item == "clinics" and parent_norm == root_norm:
            continue
        path = os.path.join(dir_path, item)
        if not os.path.isdir(path):
            continue
        if find_primary_model_pkl(path) is not None:
            out.append(item)
            continue
        if item.startswith("v"):
            out.append(item)
    return out


def list_model_versions(clinic_key: Optional[Union[str, int]] = None) -> List[str]:
    """
    When ``clinic_key`` is set, returns **global root versions plus** that clinic's subdirectory
    (deduped). Used for training warm-start fallback so a clinic without local artifacts can still
    see global models. For UI that must show **only** artifacts stored under the clinic folder, use
    :func:`list_model_versions_clinic_storage_only`.
    """
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


def list_model_versions_clinic_storage_only(clinic_key: Optional[Union[str, int]] = None) -> List[str]:
    """
    Versions that exist **only** under the given scope on disk — no global+clinic merge.

    - ``clinic_key`` None: same as global root (``models/`` top-level ``v*`` dirs).
    - ``clinic_key`` set: only ``models/clinics/<slug>/v*`` (excludes global root).
    """
    root = _models_root()
    ck = normalize_clinic_key(clinic_key)
    if ck:
        sub = os.path.join(root, "clinics", clinic_dir_slug(ck))
        raw = _collect_versions_under(sub)
    else:
        raw = _collect_versions_under(root)
    timestamped = sorted([v for v in raw if "_" in v], reverse=True)
    semantic = sorted([v for v in raw if "_" not in v], reverse=True)
    rest = [v for v in raw if v not in timestamped and v not in semantic]
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
        # Prefer clinic-local storage only when inference artifacts are complete.
        # Empty/incomplete clinic folders can appear during failed syncs and should
        # not shadow valid global versions.
        if os.path.isdir(sub):
            tab = os.path.join(sub, "tab_preprocess.pkl")
            sym = os.path.join(sub, "symptoms_mlb.pkl")
            if find_primary_model_pkl(sub) is not None and os.path.isfile(tab) and os.path.isfile(sym):
                return sub
            logger.warning(
                "Clinic model folder incomplete; fallback to global version=%s clinic=%s path=%s",
                model_version,
                ck,
                sub,
            )
    return os.path.join(root, model_version)


def find_version_label_for_artifact_realpath(
    artifact_realpath: str,
    clinic_key: Optional[Union[str, int]] = None,
) -> Optional[str]:
    """
    Return an existing version folder name (under ``MODEL_ROOT``) whose resolved directory
    equals ``artifact_realpath`` (after :func:`os.path.realpath`).
    """
    target = os.path.realpath(str(artifact_realpath or "").strip())
    if not os.path.isdir(target):
        return None
    ck = normalize_clinic_key(clinic_key)
    matches: List[str] = []
    for v in list_model_versions(ck):
        try:
            d = os.path.realpath(resolve_model_dir(v, ck))
        except OSError:
            continue
        if d == target:
            matches.append(v)
    if not matches:
        return None
    return sorted(matches, reverse=True)[0]


def display_label_for_model_version(
    clinic_key: Optional[Union[str, int]],
    model_version: str,
    storage_scope: str,
) -> str:
    """Human-friendly label in clinic UIs: global-origin rows get the `` - global`` suffix."""
    ck = normalize_clinic_key(clinic_key)
    mv = str(model_version or "").strip()
    if ck is not None and str(storage_scope or "").strip().lower() == "global":
        if mv.endswith(" - global"):
            return mv
        return f"{mv} - global"
    return mv


def storage_scope_for_version(clinic_key: Optional[Union[str, int]], model_version: str) -> str:
    """
    Whether ``resolve_model_dir`` picked a clinic subfolder or the global models root.
    """
    ck = normalize_clinic_key(clinic_key)
    d = os.path.abspath(resolve_model_dir(model_version, ck))
    if not ck:
        return "global"
    clinic_root = os.path.abspath(os.path.join(_models_root(), "clinics", clinic_dir_slug(ck)))
    prefix = clinic_root + os.sep
    return "clinic" if d.startswith(prefix) else "global"


def active_model_for_predict(
    clinic_key: Optional[Union[str, int]],
    model_version_override: Optional[str],
    fallback: ActiveModel,
) -> tuple[ActiveModel, bool]:
    """
    Resolve which on-disk model to run for ``POST /predict``.

    When ``model_version_override`` is set, it must appear in
    :func:`list_user_visible_model_versions` for the same ``clinic_key`` (merged clinic + global
    when a clinic is set), and artifacts under the resolved directory must validate.

    Returns ``(active_model, used_explicit_override)``.
    """
    raw = (model_version_override or "").strip()
    if not raw:
        chosen = get_active_model_for_clinic(clinic_key) or fallback
        return (chosen, False)

    ck = normalize_clinic_key(clinic_key)
    allowed = set(list_user_visible_model_versions(ck))
    if raw not in allowed:
        raise ValueError(
            f"model_version {raw!r} is not available for this clinic scope. "
            f"Use GET /predict/models?clinicId=... to list allowed versions."
        )

    model_dir = resolve_model_dir(raw, ck)
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory missing for version {raw!r}: {model_dir}")
    _verify_inference_artifacts(model_dir)
    return (ActiveModel(model_version=raw, model_dir=model_dir), True)


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

    mp = find_primary_model_pkl(model_dir)
    if mp is None or not mp.is_file():
        raise ValueError(f"Missing model pickle (model.pkl / .model.pkl) under {model_dir}")
    for name in ("tab_preprocess.pkl", "symptoms_mlb.pkl"):
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            raise ValueError(f"Missing artifact {name} under {model_dir}")
    try:
        joblib.load(str(mp))
    except Exception as e:
        raise ValueError(f"Model pickle cannot be loaded from {mp}: {e}") from e


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

    versions = list_user_visible_model_versions(None)
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
