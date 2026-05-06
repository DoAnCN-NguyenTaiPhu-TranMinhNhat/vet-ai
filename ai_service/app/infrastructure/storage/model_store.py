import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key

logger = logging.getLogger(__name__)

# Symlinks created under MODEL_ROOT to map MLAir ``file://`` trees into a ``v*`` name.
# They are not "models" for operators: hide from version pickers; dedupe by realpath.
_MLAIR_ALIAS_PREFIX = "v_mlair_"


def is_mlair_materialization_alias(version: str) -> bool:
    v = str(version or "").strip()
    return bool(v) and v.startswith(_MLAIR_ALIAS_PREFIX)


def list_user_visible_model_versions(clinic_key: Optional[Union[str, int]] = None) -> List[str]:
    """Like :func:`list_model_versions` but omits internal MLAir materialization symlinks."""
    return [v for v in list_model_versions(clinic_key) if not is_mlair_materialization_alias(v)]


def list_user_visible_model_versions_clinic_storage_only(clinic_key: Optional[Union[str, int]] = None) -> List[str]:
    return [
        v
        for v in list_model_versions_clinic_storage_only(clinic_key)
        if not is_mlair_materialization_alias(v)
    ]


@dataclass(frozen=True)
class ActiveModel:
    model_version: str
    model_dir: str


_STATE_DIR = os.getenv("VETAI_STATE_DIR") or os.path.join(os.path.dirname(__file__), "state")
_ACTIVE_MODEL_FILE = os.path.join(_STATE_DIR, "active_model.json")


def _sync_mlair_stages_after_active_change() -> None:
    """
    After Vet-AI active/pinned model changes, re-sync MLAir so production stage matches
    the same artifact as Vet-AI active (see mlair_client.sync_all_models_to_mlair).
    """
    if os.getenv("VETAI_MLAIR_SYNC_ON_ACTIVE_CHANGE", "true").lower() not in ("1", "true", "yes", "y"):
        return
    def _run_sync() -> None:
        try:
            from ai_service.app.infrastructure.external import mlair_client as _mc

            if not _mc.config_summary().get("enabled"):
                return
            _mc.sync_all_models_to_mlair()
            logger.info("MLAir model stages refreshed after Vet-AI active model change")
        except Exception as exc:
            logger.warning("MLAir refresh after active model change failed (non-fatal): %s", exc)

    # Non-blocking: active-model APIs should return fast even when MLAir sync is slow.
    threading.Thread(target=_run_sync, daemon=True, name="vetai-mlair-sync-after-active").start()


def materialize_mlair_artifact_as_version(
    *,
    artifact_uri: str,
    version_label: str,
    clinic_key: Optional[Union[str, int]] = None,
) -> str:
    """
    Map a ``file://`` MLAir artifact directory into ``MODEL_ROOT`` as a ``v*`` entry (symlink or copy).

    If the same weights already exist under any ``v*`` folder for this scope (same realpath),
    returns that folder name and **does not** create another ``v_mlair_*`` symlink.

    ``v_mlair_*`` names are internal aliases for Vet-AI only; they are not shown in user-facing
    model lists (see :func:`list_user_visible_model_versions`).
    """
    uri = str(artifact_uri or "").strip()
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("materialize_mlair_artifact_requires_file_scheme")
    src = os.path.realpath(parsed.path)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"mlair_artifact_not_a_directory:{src}")

    ck = normalize_clinic_key(clinic_key)
    existing = find_version_label_for_artifact_realpath(src, ck)
    if existing:
        logger.info(
            "materialize_mlair_reuse_existing realpath=%s version=%s (skip new symlink)",
            src,
            existing,
        )
        return existing

    label = str(version_label).strip()
    if not label or not label.startswith("v"):
        raise ValueError("version_label_must_start_with_v")

    root = _models_root()
    if ck:
        dest_parent = os.path.join(root, "clinics", clinic_dir_slug(ck))
    else:
        dest_parent = root
    os.makedirs(dest_parent, exist_ok=True)
    dest = os.path.join(dest_parent, label)

    if os.path.lexists(dest):
        if os.path.islink(dest) and os.path.realpath(dest) == src:
            return label
        if os.path.isdir(dest) and not os.path.islink(dest):
            shutil.rmtree(dest)
        elif os.path.lexists(dest):
            os.unlink(dest)
    try:
        os.symlink(src, dest, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dest)
    return label


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
        if not item.startswith("v"):
            continue
        # Internal MLAir alias links may point outside local disk and can be slow to stat;
        # user-visible model pickers do not need them.
        if is_mlair_materialization_alias(item):
            continue
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
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

    timestamped = sorted([v for v in out if "_" in v and not is_mlair_materialization_alias(v)], reverse=True)
    semantic = sorted([v for v in out if "_" not in v and not is_mlair_materialization_alias(v)], reverse=True)
    rest = [
        v
        for v in out
        if v not in timestamped and v not in semantic and not is_mlair_materialization_alias(v)
    ]
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
        if is_mlair_materialization_alias(version):
            continue
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
                if is_mlair_materialization_alias(version):
                    continue
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
            required = ("model.pkl", "tab_preprocess.pkl", "symptoms_mlb.pkl")
            if all(os.path.isfile(os.path.join(sub, name)) for name in required):
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

    Used when MLAir promote should only **pin active** to weights already present in the store
    (no new ``v_mlair_*`` symlinks).
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
    non_alias = [m for m in matches if not is_mlair_materialization_alias(m)]
    pool = non_alias if non_alias else matches
    return sorted(pool, reverse=True)[0]


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
    _sync_mlair_stages_after_active_change()
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
    _sync_mlair_stages_after_active_change()
    return ActiveModel(model_version=mv, model_dir=model_dir)
