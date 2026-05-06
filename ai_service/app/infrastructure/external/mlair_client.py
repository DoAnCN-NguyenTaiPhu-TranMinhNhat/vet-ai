from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import quote

from ai_service.app.domain.services.clinic_catalog_service import get_clinics_for_mlops
from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key
from ai_service.app.infrastructure.storage import model_store

logger = logging.getLogger(__name__)


def _mlair_file_uri_canonical(uri: str) -> str:
    """Normalize file:// artifact URIs so symlink dir, trailing slash, and spelling variants dedupe."""
    u = str(uri or "").strip()
    if not u.lower().startswith("file://"):
        return u
    raw = u[len("file://") :].split("?", 1)[0].strip()
    if not raw:
        return u
    try:
        rp = os.path.realpath(raw)
    except OSError:
        rp = os.path.normpath(raw)
    rp = rp.rstrip("/") or "/"
    return f"file://{rp}"


# Default DAG for MLAir executor (same shape as ml-air/scripts/day6_integration_check.py).
_DEFAULT_TRAINING_PIPELINE_CONFIG: Dict[str, Any] = {
    "tasks": [
        # NOTE: ml-air enforces plugin-contract validation for pipeline versions.
        # If a task has no `plugin`, pipeline version creation fails with status=BLOCKED.
        {"id": "extract", "plugin": "app_etl_adapter"},
        {"id": "transform", "depends_on": ["extract"], "plugin": "app_etl_adapter"},
        {"id": "train", "depends_on": ["transform"], "plugin": "app_train_adapter"},
    ]
}


def _env_truthy(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default) or "").lower() in ("1", "true", "yes", "y")


def _effective_training_pipeline_id_for_seed() -> str:
    """
    MLAir lists pipelines from runs plus any ``pipeline_id`` present in ``pipeline_versions``.
    If ``MLAIR_PIPELINE_ID`` is unset, optionally use a stable default so sync/trigger can seed
    ``POST .../pipelines/{id}/versions`` and model–pipeline mapping without extra manual setup.
    """
    explicit = str(os.getenv("MLAIR_PIPELINE_ID", "") or "").strip()
    if explicit:
        return explicit
    if not _env_truthy("VETAI_MLAIR_AUTO_ENSURE_TRAINING_PIPELINE_ID", "true"):
        return ""
    return str(os.getenv("VETAI_MLAIR_DEFAULT_PIPELINE_ID", "vet-ai-training") or "").strip()


def _mirror_model_dataset_map_raw() -> Dict[str, Any]:
    raw = os.getenv("VETAI_MLAIR_MIRROR_MODEL_DATASET_MAP_JSON", "").strip()
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Invalid VETAI_MLAIR_MIRROR_MODEL_DATASET_MAP_JSON; ignoring")
        return {}


def _resolve_mirror_model_dataset_ids(clinic_id: Optional[str]) -> tuple[str, str, Optional[str]]:
    """
    Resolve (model_id, dataset_id, dataset_version_id?) for MLAir POST .../runs/trigger mirroring.

    Per-clinic overrides: JSON map keyed by normalized clinic id (same keys as MLAIR_CLINIC_TENANT_MAP_JSON).
    Fallback: VETAI_MLAIR_MIRROR_MODEL_ID, VETAI_MLAIR_MIRROR_DATASET_ID, VETAI_MLAIR_MIRROR_DATASET_VERSION_ID.
    """
    from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key

    mmap = _mirror_model_dataset_map_raw()
    ck = normalize_clinic_key(clinic_id) if clinic_id else None
    if ck and isinstance(mmap.get(ck), dict):
        row = mmap[ck]
        mid = str(row.get("model_id") or "").strip()
        did = str(row.get("dataset_id") or "").strip()
        dvid = str(row.get("dataset_version_id") or "").strip() or None
        if mid and did:
            return mid, did, dvid
    mid = str(os.getenv("VETAI_MLAIR_MIRROR_MODEL_ID", "") or "").strip()
    did = str(os.getenv("VETAI_MLAIR_MIRROR_DATASET_ID", "") or "").strip()
    dvid = str(os.getenv("VETAI_MLAIR_MIRROR_DATASET_VERSION_ID", "") or "").strip() or None
    return mid, did, dvid


def _tenant_project_http_path(cfg: Dict[str, Any], clinic_id: Optional[str], path_suffix: str) -> str:
    scope = _resolve_scope(cfg, clinic_id)
    tid = quote(str(scope["tenant_id"]).strip(), safe="")
    pid = quote(str(scope["project_id"]).strip(), safe="")
    base = str(cfg.get("base_url") or "").rstrip("/")
    suf = path_suffix if path_suffix.startswith("/") else f"/{path_suffix}"
    return f"{base}/v1/tenants/{tid}/projects/{pid}{suf}"


def _autodiscover_mirror_model_dataset_ids(clinic_id: Optional[str]) -> tuple[str, str, Optional[str]]:
    """
    If env ids are unset, use the first model and first dataset returned by MLAir for this tenant/project
    (typical after ``sync_all_models_to_mlair`` + dataset upload).
    """
    cfg = _cfg()
    if not cfg.get("enabled"):
        return "", "", None
    timeout = min(float(cfg.get("timeout") or 10.0), 20.0)
    try:
        mbody = _request_json(
            "GET",
            _tenant_project_http_path(cfg, clinic_id, "/models?limit=20&offset=0"),
            body=None,
            headers=_headers(cfg),
            timeout=timeout,
        )
        items = mbody.get("items") if isinstance(mbody.get("items"), list) else []
        mid = ""
        for it in items:
            if isinstance(it, dict) and str(it.get("model_id") or "").strip():
                mid = str(it["model_id"]).strip()
                break
        dbody = _request_json(
            "GET",
            _tenant_project_http_path(cfg, clinic_id, "/datasets?limit=20&offset=0"),
            body=None,
            headers=_headers(cfg),
            timeout=timeout,
        )
        ditems = dbody.get("items") if isinstance(dbody.get("items"), list) else []
        did = ""
        for it in ditems:
            if isinstance(it, dict) and str(it.get("dataset_id") or "").strip():
                did = str(it["dataset_id"]).strip()
                break
        if mid and did:
            logger.info(
                "MLAir mirror autodiscovered model_id=%s dataset_id=%s (clinic_id=%s)",
                mid,
                did,
                clinic_id,
            )
        return mid, did, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLAir mirror autodiscover failed: %s", exc)
        return "", "", None


def mirror_training_job_to_mlair(
    *,
    idempotency_key: str,
    clinic_id: Optional[str],
    training_mode: Optional[str],
    override_config: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Mirror a Vet-AI training job into MLAir as a new run.

    - Default: ``POST .../pipelines/{id}/run`` via :func:`trigger_training_run` (requires ``MLAIR_PIPELINE_ID``).
    - When ``VETAI_MLAIR_MIRROR_USE_RUNS_TRIGGER=true``: ``POST .../runs/trigger`` with model + dataset.
      Ids come from env / per-clinic JSON, or from autodiscover (``VETAI_MLAIR_MIRROR_AUTODISCOVER``, default on).
      If still unresolved, falls back to :func:`trigger_training_run` with a warning.
      Readiness ``override_config`` is left to MLAir for the model+dataset path.
    """
    if _env_truthy("VETAI_MLAIR_MIRROR_USE_RUNS_TRIGGER", "false"):
        mid, did, dvid = _resolve_mirror_model_dataset_ids(clinic_id)
        if (not mid or not did) and _env_truthy("VETAI_MLAIR_MIRROR_AUTODISCOVER", "true"):
            am, ad, adv = _autodiscover_mirror_model_dataset_ids(clinic_id)
            mid = mid or am
            did = did or ad
            dvid = dvid or adv
        if not mid or not did:
            logger.warning(
                "MLAir mirror: VETAI_MLAIR_MIRROR_USE_RUNS_TRIGGER set but no model/dataset "
                "(set VETAI_MLAIR_MIRROR_MODEL_ID / DATASET_ID or seed MLAir); using pipeline trigger_training_run"
            )
            return trigger_training_run(
                idempotency_key=idempotency_key,
                clinic_id=clinic_id,
                training_mode=training_mode,
                override_config=override_config,
                context=context,
            )
        p_ov = _effective_training_pipeline_id_for_seed() or None
        return trigger_run_by_model_dataset(
            model_id=mid,
            dataset_id=did,
            idempotency_key=idempotency_key,
            clinic_id=clinic_id,
            dataset_version_id=dvid,
            pipeline_id_override=p_ov,
            context=context,
            training_mode=training_mode,
            override_config=override_config,
        )
    return trigger_training_run(
        idempotency_key=idempotency_key,
        clinic_id=clinic_id,
        training_mode=training_mode,
        override_config=override_config,
        context=context,
    )


def _cfg() -> Dict[str, Any]:
    clinic_tenant_map = os.getenv("MLAIR_CLINIC_TENANT_MAP_JSON", "").strip()
    clinic_model_alias_map = os.getenv("MLAIR_CLINIC_MODEL_ALIAS_MAP_JSON", "").strip()
    return {
        "enabled": os.getenv("MLAIR_ENABLED", "false").lower() == "true",
        "base_url": os.getenv("MLAIR_API_BASE_URL", "http://localhost:8080").rstrip("/"),
        "tenant_id": os.getenv("MLAIR_TENANT_ID", "default"),
        "project_id": os.getenv("MLAIR_PROJECT_ID", "default_project"),
        # MLAIR_PIPELINE_ID or, when VETAI_MLAIR_AUTO_ENSURE_TRAINING_PIPELINE_ID=true, VETAI_MLAIR_DEFAULT_PIPELINE_ID.
        "pipeline_id": _effective_training_pipeline_id_for_seed(),
        "token": os.getenv("MLAIR_AUTH_TOKEN", ""),
        "timeout": float(os.getenv("MLAIR_TIMEOUT_SECONDS", "10")),
        "clinic_tenant_map": json.loads(clinic_tenant_map) if clinic_tenant_map else {},
        "clinic_model_alias_map": json.loads(clinic_model_alias_map) if clinic_model_alias_map else {},
        "model_scope_per_clinic": os.getenv("MLAIR_MODEL_SCOPE_PER_CLINIC", "true").lower() == "true",
    }


def _headers(cfg: Dict[str, Any]) -> Dict[str, str]:
    out = {"Content-Type": "application/json"}
    if cfg["token"]:
        out["Authorization"] = f"Bearer {cfg['token']}"
    return out


def _request_json(method: str, url: str, *, body: Dict[str, Any] | None, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"MLAir HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach MLAir: {exc.reason}") from exc


def _http_delete(url: str, cfg: Dict[str, Any]) -> int:
    req = urllib.request.Request(url, method="DELETE", headers=_headers(cfg))
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("timeout") or 30.0)) as resp:  # noqa: S310
            resp.read()
            return int(getattr(resp, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"MLAir HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach MLAir: {exc.reason}") from exc


def _request_delete_json(url: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="DELETE", headers=_headers(cfg))
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("timeout") or 30.0)) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"MLAir HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach MLAir: {exc.reason}") from exc


def _note_pipeline_map_target(
    targets: set[tuple[Optional[str], str]],
    clinic_id_for_scope: Optional[str],
    model_id: str,
) -> None:
    mid = str(model_id).strip()
    if not mid:
        return
    ck: Optional[str] = None
    if clinic_id_for_scope:
        ck = normalize_clinic_key(str(clinic_id_for_scope).strip())
    targets.add((ck, mid))


def _put_model_version_approval(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    model_id: str,
    version: int,
    *,
    approval_status: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    mid = quote(str(model_id).strip(), safe="")
    base = str(cfg.get("base_url") or "").rstrip("/")
    url = f"{base}/v1/tenants/{tid}/projects/{pid}/models/{mid}/versions/{int(version)}/approval"
    body: Dict[str, Any] = {"approval_status": str(approval_status).strip()}
    if reason is not None and str(reason).strip():
        body["reason"] = str(reason).strip()
    return _request_json("PUT", url, body=body, headers=_headers(cfg), timeout=float(cfg.get("timeout") or 10.0))


def _maybe_auto_approve_version_for_promote(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    model_id: str,
    version: int,
) -> bool:
    """
    MLAir promote to production rejects unless approval_status=approved (unless ML_AIR_SKIP_APPROVAL_FOR_PROMOTE).
    New model_versions are always created pending_manual_approval, so Vet-AI sync must approve before promote.
    """
    if not _env_truthy("VETAI_MLAIR_AUTO_APPROVE_ON_SYNC", "true"):
        return False
    try:
        _put_model_version_approval(
            cfg,
            tenant_id,
            project_id,
            model_id,
            int(version),
            approval_status="approved",
            reason="vet-ai sync (auto-approve for production promote)",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLAir auto-approve skipped/failed tenant=%s project=%s model=%s v=%s: %s",
            tenant_id,
            project_id,
            model_id,
            version,
            exc,
        )
        return False


def config_summary() -> Dict[str, Any]:
    cfg = _cfg()
    use_runs_trigger = _env_truthy("VETAI_MLAIR_MIRROR_USE_RUNS_TRIGGER", "false")
    autodisc = _env_truthy("VETAI_MLAIR_MIRROR_AUTODISCOVER", "true")
    mid, did, _ = _resolve_mirror_model_dataset_ids(None)
    explicit_pipe = str(os.getenv("MLAIR_PIPELINE_ID", "") or "").strip()
    eff_pipe = _effective_training_pipeline_id_for_seed()
    return {
        "enabled": cfg["enabled"],
        "base_url": cfg["base_url"],
        "tenant_id": cfg["tenant_id"],
        "project_id": cfg["project_id"],
        "pipeline_id": cfg["pipeline_id"],
        "mlair_pipeline_id_explicit": explicit_pipe,
        "pipeline_id_configured": bool(eff_pipe),
        "auto_ensure_training_pipeline_id": _env_truthy("VETAI_MLAIR_AUTO_ENSURE_TRAINING_PIPELINE_ID", "true"),
        "default_training_pipeline_id": str(os.getenv("VETAI_MLAIR_DEFAULT_PIPELINE_ID", "vet-ai-training") or "").strip(),
        "has_token": bool(cfg["token"]),
        "model_scope_per_clinic": cfg["model_scope_per_clinic"],
        "auto_approve_on_sync": _env_truthy("VETAI_MLAIR_AUTO_APPROVE_ON_SYNC", "true"),
        "map_pipeline_on_sync": _env_truthy("VETAI_MLAIR_MAP_PIPELINE_ON_SYNC", "true"),
        "mirror_use_runs_trigger": use_runs_trigger,
        "mirror_autodiscover": autodisc,
        "mirror_model_dataset_configured_global": bool(mid and did),
    }


def _resolve_scope(cfg: Dict[str, Any], clinic_id: Optional[str]) -> Dict[str, str]:
    clinic_key = str(clinic_id).strip() if clinic_id else ""
    tenant_id = cfg["tenant_id"]
    project_id = cfg["project_id"]
    if clinic_key:
        tenant_id = str(cfg["clinic_tenant_map"].get(clinic_key) or tenant_id)
        # Scope is always clinic-driven to avoid env hard-coded project mapping.
        project_id = f"clinic_{clinic_dir_slug(clinic_key)}"
    return {"tenant_id": tenant_id, "project_id": project_id}


def _clinic_model_suffix(cfg: Dict[str, Any], clinic_key: str) -> str:
    alias = str(cfg.get("clinic_model_alias_map", {}).get(clinic_key) or "").strip()
    if alias:
        return clinic_dir_slug(alias)
    return clinic_dir_slug(clinic_key)


def _scoped_model_name(base_name: str, clinic_key: Optional[str]) -> str:
    # Naming convention:
    # - global scope: vet-global
    # - clinic scope: vet-<clinic_id>
    # Keep default "vet" prefix to make filtering deterministic in MLAir UI.
    prefix = "vet"
    if base_name.strip():
        normalized = clinic_dir_slug(base_name.strip())
        if normalized and normalized != "vet-ai":
            prefix = normalized
    if clinic_key:
        return f"{prefix}-{clinic_dir_slug(clinic_key)}"
    return f"{prefix}-global"


def register_model_version_for_scope(
    *,
    tenant_id: str,
    project_id: str,
    model_id: str,
    artifact_uri: str,
    run_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new MLAir model_versions row under an existing models row (same tenant/project as the training run).
    Used after Vet-AI finishes training so the clinic-scoped MLAir model reflects the new artifact.
    """
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    mid = quote(str(model_id).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/models/{mid}/versions"
    stg = (stage or os.getenv("VETAI_MLAIR_NEW_VERSION_STAGE", "staging") or "staging").strip()
    body: Dict[str, Any] = {"artifact_uri": str(artifact_uri).strip(), "stage": stg}
    if run_id and str(run_id).strip():
        body["run_id"] = str(run_id).strip()
    hdr = dict(_headers(cfg))
    hdr["Content-Type"] = "application/json"
    return _request_json("POST", url, body=body, headers=hdr, timeout=float(cfg["timeout"]))


def trigger_training_run(
    *,
    idempotency_key: str,
    pipeline_id: str | None = None,
    clinic_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    training_mode: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    pid = str(pipeline_id or cfg["pipeline_id"] or "").strip()
    if not pid:
        raise RuntimeError(
            "No pipeline id: set MLAIR_PIPELINE_ID, or enable VETAI_MLAIR_AUTO_ENSURE_TRAINING_PIPELINE_ID (default true) "
            "with VETAI_MLAIR_DEFAULT_PIPELINE_ID, and run POST /mlair/models/sync to create pipeline_versions."
        )
    scope = _resolve_scope(cfg, clinic_id)
    use_gating_run = os.getenv("MLAIR_USE_GATING_RUN_ENDPOINT", "true").lower() == "true"
    route = "run" if use_gating_run else "runs"
    tid = quote(str(scope["tenant_id"]).strip(), safe="")
    prj = quote(str(scope["project_id"]).strip(), safe="")
    pl = quote(str(pid).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{prj}/pipelines/{pl}/{route}"
    body: Dict[str, Any] = {
        "pipeline_id": pid,
        "idempotency_key": idempotency_key,
        # Critical for clinic scopes: without this, MLAir can create runs with
        # no config_snapshot/pipeline_version_id and scheduler falls back to task:1.
        "use_latest_pipeline_version": True,
    }
    if clinic_id:
        body["context"] = {"clinic_id": clinic_id}
    if context:
        merged = dict(body.get("context") or {})
        merged.update(context)
        body["context"] = merged
    if training_mode:
        body["training_mode"] = str(training_mode).strip().lower()
    if override_config and isinstance(override_config, dict):
        body["override_config"] = override_config
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def trigger_run_by_model_dataset(
    *,
    model_id: str,
    dataset_id: str,
    idempotency_key: str,
    clinic_id: Optional[str] = None,
    dataset_version_id: Optional[str] = None,
    pipeline_id_override: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    training_mode: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Model-centric trigger: ``POST /v1/tenants/{tid}/projects/{pid}/runs/trigger`` (MLAir ≥ model pipeline mapping).

    Does not require ``MLAIR_PIPELINE_ID`` when MLAir resolves the default pipeline from model–dataset mapping.
    Optional ``pipeline_id_override`` (or env ``MLAIR_PIPELINE_ID``) can still be sent when the server expects it.
    """
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    scope = _resolve_scope(cfg, clinic_id)
    tid = quote(str(scope["tenant_id"]).strip(), safe="")
    pid = quote(str(scope["project_id"]).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/runs/trigger"
    body: Dict[str, Any] = {
        "model_id": str(model_id).strip(),
        "dataset_id": str(dataset_id).strip(),
        "idempotency_key": str(idempotency_key).strip(),
    }
    if dataset_version_id and str(dataset_version_id).strip():
        body["dataset_version_id"] = str(dataset_version_id).strip()
    p_ov = str(pipeline_id_override or cfg.get("pipeline_id") or "").strip()
    if p_ov:
        body["pipeline_id_override"] = p_ov
    if clinic_id:
        body["context"] = {"clinic_id": clinic_id}
    if context:
        merged = dict(body.get("context") or {})
        merged.update(context)
        body["context"] = merged
    if training_mode:
        body["training_mode"] = str(training_mode).strip().lower()
    if override_config and isinstance(override_config, dict):
        body["override_config"] = override_config
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def list_project_pipelines(
    *,
    clinic_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Proxy MLAir ``GET .../pipelines`` (pipelines appear after runs or ``pipeline_versions`` rows)."""
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")
    lim = max(1, min(int(limit), 200))
    off = max(0, int(offset))
    url = _tenant_project_http_path(cfg, clinic_id, f"/pipelines?limit={lim}&offset={off}")
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg.get("timeout") or 10.0))


def list_project_pipeline_versions(
    *,
    clinic_id: Optional[str] = None,
    pipeline_id: str,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Proxy MLAir ``GET .../pipelines/{pipeline_id}/versions``."""
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")
    pl = quote(str(pipeline_id).strip(), safe="")
    lim = max(1, min(int(limit), 200))
    off = max(0, int(offset))
    url = _tenant_project_http_path(cfg, clinic_id, f"/pipelines/{pl}/versions?limit={lim}&offset={off}")
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg.get("timeout") or 10.0))


def get_run(run_id: str) -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    url = f"{cfg['base_url']}/v1/tenants/{cfg['tenant_id']}/projects/{cfg['project_id']}/runs/{run_id}"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _list_models(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    *,
    limit: int = 200,
    offset: int = 0,
) -> Dict[str, Any]:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    url = (
        f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/models"
        f"?limit={int(max(1, min(limit, 500)))}&offset={int(max(0, offset))}"
    )
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _find_model_id_for_name(cfg: Dict[str, Any], tenant_id: str, project_id: str, scope_model_name: str) -> str:
    listed = _list_models(cfg, tenant_id, project_id)
    items = listed.get("items") if isinstance(listed.get("items"), list) else []
    want = str(scope_model_name).strip()
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or "").strip() == want:
            mid = str(item.get("model_id") or item.get("id") or "").strip()
            if mid:
                return mid
    return ""


def _mlair_stage_rank(stage: Any) -> int:
    s = str(stage or "").strip().lower()
    if s == "production":
        return 100
    if s == "staging":
        return 50
    if s == "archived":
        return 1
    return 10


def _delete_model_version_at(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    model_id: str,
    version_int: int,
) -> None:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    mid = quote(str(model_id).strip(), safe="")
    vid = int(version_int)
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/models/{mid}/versions/{vid}"
    _http_delete(url, cfg)


def dedupe_mlair_registered_model_versions(
    *,
    clinic_id_for_scope: Optional[str] = None,
    all_catalog_clinics: bool = False,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Delete duplicate MLAir **model version** rows that share the same canonical ``file://`` artifact.

    Per duplicate group, keeps the row with best ``stage`` (production > staging > other), then highest
    ``version`` integer. Requires MLAir ``DELETE .../models/{model_id}/versions/{version}``.

    - ``clinic_id_for_scope`` set: that clinic's MLAir project only.
    - Both unset and ``all_catalog_clinics`` false: **global** project only.
    - ``all_catalog_clinics`` true: global + every clinic from the Vet-AI clinic catalog.
    """
    if clinic_id_for_scope and all_catalog_clinics:
        raise ValueError("Specify clinic_id_for_scope or all_catalog_clinics=True, not both")

    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    model_name = os.getenv("MLAIR_MODEL_NAME", "vet-ai")
    scopes: list[tuple[str, str, str]] = []

    if all_catalog_clinics:
        g = _resolve_scope(cfg, None)
        scopes.append((g["tenant_id"], g["project_id"], "global"))
        try:
            clinics, _ = get_clinics_for_mlops()
        except Exception:
            clinics = []
        for clinic in clinics or []:
            raw = str((clinic or {}).get("id") or "").strip()
            ck = normalize_clinic_key(raw)
            if not ck:
                continue
            sc = _resolve_scope(cfg, ck)
            scopes.append((sc["tenant_id"], sc["project_id"], ck))
    elif clinic_id_for_scope:
        ck = normalize_clinic_key(str(clinic_id_for_scope).strip())
        if not ck:
            raise ValueError("clinic_id_for_scope is empty after normalization")
        sc = _resolve_scope(cfg, ck)
        scopes.append((sc["tenant_id"], sc["project_id"], ck))
    else:
        g = _resolve_scope(cfg, None)
        scopes.append((g["tenant_id"], g["project_id"], "global"))

    per_scope: list[Dict[str, Any]] = []
    total_deleted = 0
    total_would = 0

    for tid, pid, label in scopes:
        ck_name: Optional[str] = None if label == "global" else label
        scope_model_name = _scoped_model_name(
            model_name,
            ck_name if cfg["model_scope_per_clinic"] else None,
        )
        model_id = _find_model_id_for_name(cfg, tid, pid, scope_model_name)
        if not model_id:
            per_scope.append(
                {
                    "scope_label": label,
                    "tenant_id": tid,
                    "project_id": pid,
                    "skipped": "model_not_found",
                    "expected_name": scope_model_name,
                }
            )
            continue

        items = _list_model_versions_all_items(cfg, tid, pid, model_id)
        by_canon: Dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            uri = str(it.get("artifact_uri") or "").strip()
            if not uri:
                continue
            canon = _mlair_file_uri_canonical(uri)
            try:
                vn = int(it.get("version"))
            except (TypeError, ValueError):
                continue
            by_canon.setdefault(canon, []).append((vn, it))

        deleted: list[Dict[str, Any]] = []
        would_delete: list[Dict[str, Any]] = []
        groups = 0
        for canon, rows in by_canon.items():
            if len(rows) <= 1:
                continue
            groups += 1
            keep = max(rows, key=lambda r: (_mlair_stage_rank(r[1].get("stage")), int(r[0])))
            keep_v = int(keep[0])
            for vn, row in sorted(rows, key=lambda r: int(r[0])):
                if int(vn) == keep_v:
                    continue
                entry = {
                    "version": int(vn),
                    "stage": row.get("stage"),
                    "canonical_artifact": canon,
                }
                if dry_run:
                    would_delete.append(entry)
                else:
                    _delete_model_version_at(cfg, tid, pid, model_id, int(vn))
                    deleted.append(entry)

        if dry_run:
            total_would += len(would_delete)
        else:
            total_deleted += len(deleted)

        scope_report: Dict[str, Any] = {
            "scope_label": label,
            "tenant_id": tid,
            "project_id": pid,
            "model_id": model_id,
            "scoped_model_name": scope_model_name,
            "duplicate_groups": groups,
        }
        if dry_run:
            scope_report["would_delete"] = would_delete
        else:
            scope_report["deleted"] = deleted
        per_scope.append(scope_report)

    out: Dict[str, Any] = {"dry_run": dry_run, "scopes": per_scope}
    if dry_run:
        out["total_would_delete"] = total_would
    else:
        out["total_deleted"] = total_deleted
    return out


def _create_model(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    name: str,
    description: str | None = None,
) -> Dict[str, Any]:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/models"
    body: Dict[str, Any] = {"name": name}
    if description:
        body["description"] = description
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def _list_model_versions(cfg: Dict[str, Any], tenant_id: str, project_id: str, model_id: str) -> Dict[str, Any]:
    """Single GET (no pagination). Prefer :func:`_list_model_versions_all_items` for sync/dedupe."""
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    mid = quote(str(model_id).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/models/{mid}/versions"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _list_model_versions_all_items(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    model_id: str,
) -> list[dict[str, Any]]:
    """
    Load every model version row for ``model_id``.

    Current **ml-air** ``GET .../models/{id}/versions`` returns **all** rows in one response (no limit/offset
    in the handler). Vet-AI therefore uses a **single** GET by default. Set ``MLAIR_MODEL_VERSIONS_USE_OFFSET_PAGINATION=true``
    only if your MLAir fork adds pagination query params; otherwise offset loops can duplicate rows.
    """
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    mid = quote(str(model_id).strip(), safe="")
    base = str(cfg.get("base_url") or "").rstrip("/")
    timeout = float(cfg.get("timeout") or 10.0)

    if not _env_truthy("MLAIR_MODEL_VERSIONS_USE_OFFSET_PAGINATION", "false"):
        url = f"{base}/v1/tenants/{tid}/projects/{pid}/models/{mid}/versions"
        body = _request_json("GET", url, body=None, headers=_headers(cfg), timeout=timeout)
        items = body.get("items") if isinstance(body.get("items"), list) else []
        return [it for it in items if isinstance(it, dict)]

    page = max(10, min(int(os.getenv("MLAIR_MODEL_VERSIONS_PAGE_SIZE", "200") or "200"), 1000))
    all_rows: list[dict[str, Any]] = []
    offset = 0
    max_pages = max(1, min(int(os.getenv("MLAIR_MODEL_VERSIONS_MAX_PAGES", "50") or "50"), 500))
    pages = 0
    while True:
        url = f"{base}/v1/tenants/{tid}/projects/{pid}/models/{mid}/versions?limit={page}&offset={offset}"
        try:
            body = _request_json("GET", url, body=None, headers=_headers(cfg), timeout=timeout)
        except RuntimeError as exc:
            if offset == 0:
                raise
            logger.warning("MLAir model versions pagination stopped at offset=%s: %s", offset, exc)
            break
        items = body.get("items") if isinstance(body.get("items"), list) else []
        for it in items:
            if isinstance(it, dict):
                all_rows.append(it)
        pages += 1
        if len(items) < page or pages >= max_pages:
            if pages >= max_pages and len(items) == page:
                logger.warning(
                    "MLAir model versions pagination hit MLAIR_MODEL_VERSIONS_MAX_PAGES=%s for model_id=%s",
                    max_pages,
                    model_id,
                )
            break
        offset += page
    return all_rows


def inspect_serving_alignment_with_mlair(clinic_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Read-only comparison: Vet-AI active (or clinic pin fallback chain) vs MLAir model row
    named like ``_scoped_model_name(MLAIR_MODEL_NAME, clinic)`` and its **production** version row.

    ``clinic_id`` may be the canonical UUID or an exact catalog **name** (see ``resolve_clinic_identifier``).
    MLAir project id is always ``clinic_<slug>`` of that canonical id (hyphens kept for UUIDs).
    """
    from ai_service.app.domain.services.clinic_catalog_service import resolve_clinic_identifier
    from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
    from ai_service.app.infrastructure.storage import model_store as ms

    cfg = _cfg()
    resolved = resolve_clinic_identifier(clinic_id) if clinic_id else None
    ck = normalize_clinic_key(resolved if resolved is not None else clinic_id)
    out: Dict[str, Any] = {
        "enabled": bool(cfg.get("enabled")),
        "clinic_id_query": clinic_id,
        "clinic_id_resolved": resolved if clinic_id else None,
        "clinic_key_normalized": ck,
    }
    if not cfg["enabled"]:
        out["error"] = "MLAir disabled"
        return out

    scope = _resolve_scope(cfg, ck)
    tid = str(scope["tenant_id"]).strip()
    pid = str(scope["project_id"]).strip()
    out["mlair_scope"] = {"tenant_id": tid, "project_id": pid}

    if ck:
        eff = ms.get_active_model_for_clinic(ck)
        pinned = ms.get_clinic_pinned_model(ck)
    else:
        eff = ms.get_active_model()
        pinned = None

    out["vetai"] = {
        "scope": "clinic" if ck else "global",
        "effective_model_version": eff.model_version if eff else None,
        "effective_model_dir": eff.model_dir if eff else None,
        "clinic_pinned_version": pinned.model_version if pinned else None,
    }

    base_name = os.getenv("MLAIR_MODEL_NAME", "vet-ai").strip()
    expected_name = _scoped_model_name(base_name, ck)
    out["mlair_expected_model_name"] = expected_name

    listed = _list_models(cfg, tid, pid)
    items = listed.get("items") if isinstance(listed.get("items"), list) else []
    model_id: str | None = None
    for it in items:
        if not isinstance(it, dict):
            continue
        if str(it.get("name") or "").strip() == expected_name:
            model_id = str(it.get("model_id") or it.get("id") or "").strip() or None
            break
    out["mlair_model_id"] = model_id

    notes: list[str] = []
    if not model_id:
        out["aligned"] = False
        notes.append(
            "No MLAir model with expected name in this tenant/project; run POST /mlair/models/sync or check MLAIR_MODEL_NAME."
        )
        out["notes"] = notes
        return out

    vitems = _list_model_versions_all_items(cfg, tid, pid, model_id)
    production_rows = [
        it
        for it in vitems
        if isinstance(it, dict) and str(it.get("stage") or "").lower() == "production"
    ]
    best: Dict[str, Any] | None = None
    best_n = -1
    for it in production_rows:
        try:
            vn = int(it.get("version") or 0)
        except (TypeError, ValueError):
            vn = 0
        if vn >= best_n:
            best_n = vn
            best = it
    out["mlair_production_version_row"] = best
    if best:
        out["mlair_production_approval_status"] = str(best.get("approval_status") or "").strip() or None

    vet_ver = str((eff.model_version if eff else "") or "").strip()
    vet_dir = str((eff.model_dir if eff else "") or "").strip()
    uri = str((best or {}).get("artifact_uri") or "").strip() if best else ""

    aligned = False
    reasons: list[str] = []
    if vet_ver and uri:
        if vet_ver in uri:
            aligned = True
            reasons.append("artifact_uri contains vet-ai model_version string")
        if "vetai://model-version/" in uri and vet_ver in uri:
            aligned = True
            reasons.append("vetai://model-version URI matches active version")
    if vet_dir and uri.startswith("file://"):
        path_part = uri.replace("file://", "").split("?", 1)[0]
        if vet_dir.rstrip("/") in path_part or path_part.rstrip("/") in vet_dir:
            aligned = True
            reasons.append("file:// artifact path overlaps effective model_dir")

    if not best:
        out["aligned"] = False
        notes.append("MLAir model exists but no production-staged version row returned.")
    else:
        ap_st = str(best.get("approval_status") or "").strip().lower()
        if ap_st and ap_st != "approved":
            notes.append(
                "MLAir production row has approval_status other than 'approved'; "
                "ml-air may not treat it as production weights for runs until approved "
                "(POST /mlair/models/sync with VETAI_MLAIR_AUTO_APPROVE_ON_SYNC=true, "
                "or ML_AIR_SKIP_APPROVAL_FOR_PROMOTE=1 on the ml-air server)."
            )
        out["aligned"] = bool(aligned)
        if not aligned:
            notes.append(
                "Vet-AI effective version/dir does not match MLAir production artifact_uri heuristically; "
                "webhook or sync may be missing or IDs diverged."
            )

    out["alignment_reasons"] = reasons
    out["notes"] = notes
    return out


def _create_model_version(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    model_id: str,
    artifact_uri: str,
    stage: str,
) -> Dict[str, Any]:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    mid = quote(str(model_id).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/models/{mid}/versions"
    body = {"artifact_uri": artifact_uri, "stage": stage}
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def _promote_model_version(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    model_id: str,
    version: int,
    stage: str = "production",
) -> Dict[str, Any]:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    mid = quote(str(model_id).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/models/{mid}/promote"
    body = {"version": int(version), "stage": stage}
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def _training_pipeline_dag_config() -> Dict[str, Any]:
    raw = (os.getenv("MLAIR_TRAINING_PIPELINE_CONFIG_JSON") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("tasks"):
                return parsed
            logger.warning("MLAIR_TRAINING_PIPELINE_CONFIG_JSON ignored: expected object with 'tasks'")
        except json.JSONDecodeError as exc:
            logger.warning("MLAIR_TRAINING_PIPELINE_CONFIG_JSON invalid: %s", exc)
    return dict(_DEFAULT_TRAINING_PIPELINE_CONFIG)


def _list_pipeline_versions(
    cfg: Dict[str, Any], tenant_id: str, project_id: str, pipeline_id: str
) -> Dict[str, Any]:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    pl = quote(str(pipeline_id).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/pipelines/{pl}/versions?limit=50"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _create_pipeline_version(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    pipeline_id: str,
    dag: Dict[str, Any],
) -> Dict[str, Any]:
    tid = quote(str(tenant_id).strip(), safe="")
    pid = quote(str(project_id).strip(), safe="")
    pl = quote(str(pipeline_id).strip(), safe="")
    url = f"{cfg['base_url']}/v1/tenants/{tid}/projects/{pid}/pipelines/{pl}/versions"
    body = {"config": dag}
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def _ensure_clinic_training_pipeline(
    cfg: Dict[str, Any], tenant_id: str, project_id: str, pipeline_id: str
) -> bool:
    """Create initial pipeline version for this scope if none exist. Returns True if created."""
    listed = _list_pipeline_versions(cfg, tenant_id, project_id, pipeline_id)
    items = listed.get("items") if isinstance(listed, dict) else []
    items = items if isinstance(items, list) else []
    if items:
        return False
    dag = _training_pipeline_dag_config()
    _create_pipeline_version(cfg, tenant_id, project_id, pipeline_id, dag)
    return True


def _list_runs(cfg: Dict[str, Any], tenant_id: str, project_id: str, limit: int = 200) -> Dict[str, Any]:
    url = f"{cfg['base_url']}/v1/tenants/{tenant_id}/projects/{project_id}/runs?limit={int(limit)}"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _update_run_status(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    run_id: str,
    status: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{cfg['base_url']}/v1/tenants/{tenant_id}/projects/{project_id}/runs/{run_id}/status"
    body: Dict[str, Any] = {"status": status}
    if reason:
        body["reason"] = reason
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def _log_run_param(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    run_id: str,
    key: str,
    value: str,
) -> Dict[str, Any]:
    url = f"{cfg['base_url']}/v1/tenants/{tenant_id}/projects/{project_id}/runs/{run_id}/params"
    body = {"key": key, "value": value}
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def _find_run_id_by_idempotency_key(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    idempotency_key: str,
) -> str:
    runs_payload = _list_runs(cfg, tenant_id, project_id, limit=200)
    items = runs_payload.get("items") if isinstance(runs_payload, dict) else []
    items = items if isinstance(items, list) else []
    for row in items:
        if not isinstance(row, dict):
            continue
        if str(row.get("idempotency_key") or "").strip() == idempotency_key:
            run_id = str(row.get("run_id") or "").strip()
            if run_id:
                return run_id
    return ""


def sync_training_outcome_to_mlair(
    *,
    idempotency_key: str,
    status: str,
    clinic_id: Optional[str] = None,
    reason: Optional[str] = None,
    run_params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")
    scope = _resolve_scope(cfg, clinic_id)
    tenant_id = scope["tenant_id"]
    project_id = scope["project_id"]
    target_run_id = _find_run_id_by_idempotency_key(cfg, tenant_id, project_id, idempotency_key)
    if not target_run_id:
        raise RuntimeError(f"Cannot find MLAir run by idempotency_key={idempotency_key}")
    updated = _update_run_status(cfg, tenant_id, project_id, target_run_id, status, reason)
    logged_params = []
    if run_params:
        for key, value in run_params.items():
            k = str(key).strip()
            if not k:
                continue
            v = str(value)
            _log_run_param(cfg, tenant_id, project_id, target_run_id, k, v)
            logged_params.append(k)
    return {
        "run_id": target_run_id,
        "tenant_id": tenant_id,
        "project_id": project_id,
        "status": status,
        "updated": updated,
        "logged_params": logged_params,
    }


def _sync_disk_version_to_mlair(
    cfg: Dict[str, Any],
    *,
    clinic_id_for_scope: Optional[str],
    version_name: str,
    model_dir: str,
    model_name: str,
    model_desc: str,
    default_stage: str,
    active_global_version: Optional[str],
    seen_models: Dict[str, str],
    dedupe_artifact_scope: set[tuple[str, str, str]],
    pipeline_map_targets: Optional[set[tuple[Optional[str], str]]] = None,
) -> tuple[int, int, int, Optional[Dict[str, Any]]]:
    """
    Register one on-disk Vet-AI version under the MLAir tenant/project for ``clinic_id_for_scope``
    (``None`` = default/global project). Skips duplicate ``artifact_uri`` in the same project.
    """
    vn = str(version_name or "").strip()
    md = str(model_dir or "").strip()
    if not vn or not md or not os.path.isdir(md):
        return 0, 0, 0, None

    try:
        md_real = os.path.realpath(md)
    except OSError:
        md_real = os.path.normpath(md)
    artifact_uri = _mlair_file_uri_canonical(f"file://{md_real}")
    scope = _resolve_scope(cfg, clinic_id_for_scope)
    tenant_id = scope["tenant_id"]
    project_id = scope["project_id"]
    dedupe_key = (tenant_id, project_id, artifact_uri)
    if dedupe_key in dedupe_artifact_scope:
        return 0, 0, 0, None
    dedupe_artifact_scope.add(dedupe_key)

    if clinic_id_for_scope:
        ck_pin = normalize_clinic_key(str(clinic_id_for_scope).strip())
        pin = model_store.get_clinic_pinned_model(ck_pin) if ck_pin else None
        target_active = (pin.model_version or "").strip() if pin else ""
    else:
        target_active = (active_global_version or "").strip()

    desired_stage = "production" if target_active and target_active == vn else default_stage
    scope_model_name = _scoped_model_name(
        model_name, clinic_id_for_scope if cfg["model_scope_per_clinic"] else None
    )

    model_key = f"{tenant_id}:{project_id}:{scope_model_name}"
    model_id = seen_models.get(model_key)
    if not model_id:
        listed = _list_models(cfg, tenant_id, project_id)
        items = listed.get("items") if isinstance(listed, dict) else []
        items = items if isinstance(items, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "").strip() == scope_model_name:
                model_id = str(item.get("model_id") or item.get("id") or "").strip()
                if model_id:
                    break
        if not model_id:
            created = _create_model(cfg, tenant_id, project_id, scope_model_name, model_desc)
            model_id = str(created.get("model_id") or created.get("id") or "").strip()
            if not model_id:
                raise RuntimeError("Cannot resolve model_id after create_model")
        seen_models[model_key] = model_id

    if pipeline_map_targets is not None:
        _note_pipeline_map_target(pipeline_map_targets, clinic_id_for_scope, model_id)

    existing_items = _list_model_versions_all_items(cfg, tenant_id, project_id, model_id)
    existing_version_by_canon_uri: Dict[str, int] = {}
    for item in existing_items:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("artifact_uri") or "").strip()
        if not uri:
            continue
        canon = _mlair_file_uri_canonical(uri)
        try:
            ver_n = int(item.get("version"))
        except (TypeError, ValueError):
            continue
        prev = existing_version_by_canon_uri.get(canon)
        if prev is None or ver_n > prev:
            existing_version_by_canon_uri[canon] = ver_n

    if artifact_uri in existing_version_by_canon_uri:
        promoted_here = 0
        if desired_stage == "production":
            existing_version = existing_version_by_canon_uri.get(artifact_uri)
            if existing_version is not None:
                _maybe_auto_approve_version_for_promote(
                    cfg, tenant_id, project_id, model_id, int(existing_version)
                )
                _promote_model_version(
                    cfg=cfg,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    model_id=model_id,
                    version=existing_version,
                    stage="production",
                )
                promoted_here = 1
        if promoted_here:
            return 0, 1, 1, {
                "clinic_key": clinic_id_for_scope or "global",
                "tenant_id": tenant_id,
                "project_id": project_id,
                "model_name": scope_model_name,
                "stage": desired_stage,
                "action": "promoted_existing",
            }
        return 0, 1, 0, None

    created_row = _create_model_version(cfg, tenant_id, project_id, model_id, artifact_uri, desired_stage)
    if _env_truthy("VETAI_MLAIR_AUTO_APPROVE_ON_SYNC", "true") and str(desired_stage or "").strip().lower() == "production":
        try:
            vn = int((created_row or {}).get("version") or 0)
            if vn:
                _put_model_version_approval(
                    cfg,
                    tenant_id,
                    project_id,
                    model_id,
                    vn,
                    approval_status="approved",
                    reason="vet-ai sync (production artifact)",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLAir auto-approve after version create model_id=%s: %s", model_id, exc)
    return 1, 0, 0, {
        "clinic_key": clinic_id_for_scope or "global",
        "tenant_id": tenant_id,
        "project_id": project_id,
        "model_name": scope_model_name,
        "stage": desired_stage,
        "action": "created_version",
    }


def _fill_pipeline_map_targets_from_seen_models(
    cfg: Dict[str, Any],
    *,
    seen_models: Dict[str, str],
    clinics: list[Any],
    discovered: list[Any],
    work: list[tuple[Optional[str], str, str]],
    pipeline_map_targets: set[tuple[Optional[str], str]],
) -> None:
    """
    Ensure pipeline mapping runs for every MLAir model row we track: global (default project),
    every catalog/discovered/mirror clinic, and any clinic project inferred from ``clinic_<slug>``.
    """
    ordered_clinic_ids: list[str] = []
    seen_ck: set[str] = set()
    for clinic in clinics or []:
        raw = str((clinic or {}).get("id") or "").strip()
        if not raw or raw in seen_ck:
            continue
        seen_ck.add(raw)
        ordered_clinic_ids.append(raw)
    for row in discovered or []:
        raw = str(row.get("clinic_key") or "").strip() or None
        if not raw or raw in seen_ck:
            continue
        seen_ck.add(raw)
        ordered_clinic_ids.append(raw)
    for ck, _, _ in work:
        if not ck or ck in seen_ck:
            continue
        seen_ck.add(ck)
        ordered_clinic_ids.append(ck)

    slug_to_clinic: dict[str, str] = {}
    for raw_ck in ordered_clinic_ids:
        nk = normalize_clinic_key(raw_ck) or raw_ck
        slug_to_clinic[clinic_dir_slug(nk)] = raw_ck
        slug_to_clinic[clinic_dir_slug(raw_ck)] = raw_ck

    for model_key, mid in seen_models.items():
        mid_s = str(mid).strip()
        if not mid_s:
            continue
        parts = str(model_key).split(":", 2)
        if len(parts) != 3:
            continue
        tid_k, pid_k, _ = parts[0], parts[1], parts[2]
        gsc = _resolve_scope(cfg, None)
        if tid_k == gsc["tenant_id"] and pid_k == gsc["project_id"]:
            _note_pipeline_map_target(pipeline_map_targets, None, mid_s)
            continue
        resolved_raw: Optional[str] = None
        for raw_ck in ordered_clinic_ids:
            sc = _resolve_scope(cfg, raw_ck)
            if sc["tenant_id"] == tid_k and sc["project_id"] == pid_k:
                resolved_raw = raw_ck
                break
        if resolved_raw:
            _note_pipeline_map_target(pipeline_map_targets, resolved_raw, mid_s)
            continue
        if str(pid_k).startswith("clinic_"):
            slug = str(pid_k)[len("clinic_") :]
            guess = slug_to_clinic.get(slug)
            if guess:
                _note_pipeline_map_target(pipeline_map_targets, guess, mid_s)


def sync_all_models_to_mlair() -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    model_name = os.getenv("MLAIR_MODEL_NAME", "vet-ai")
    model_desc = os.getenv("MLAIR_MODEL_DESCRIPTION", "Synced from Vet-AI model store")
    default_stage = os.getenv("MLAIR_MODEL_STAGE", "staging")
    discovered = model_store.list_model_versions_with_scope()
    created_count = 0
    promoted_count = 0
    created_models = 0
    skipped_count = 0
    synced_scopes: list[Dict[str, Any]] = []
    seen_models: Dict[str, str] = {}
    dedupe_artifact_scope: set[tuple[str, str, str]] = set()
    pipeline_map_targets: set[tuple[Optional[str], str]] = set()

    active_global = model_store.get_active_model()
    active_global_version = active_global.model_version if active_global else None

    work: list[tuple[Optional[str], str, str]] = []
    seen_work: set[tuple[Optional[str], str, str]] = set()

    for row in discovered:
        clinic_key = str(row.get("clinic_key") or "").strip() or None
        version_name = str(row.get("version") or "").strip()
        model_dir = str(row.get("model_dir") or "").strip()
        if not model_dir or not version_name:
            continue
        wk = (clinic_key, version_name, os.path.normpath(model_dir))
        if wk in seen_work:
            continue
        seen_work.add(wk)
        work.append((clinic_key, version_name, model_dir))

    mirror_globals = _env_truthy("VETAI_MLAIR_MIRROR_GLOBAL_VERSIONS_IN_CLINIC_PROJECTS", "true")
    if mirror_globals:
        try:
            clinics_mirror, _src_m = get_clinics_for_mlops()
        except Exception:
            clinics_mirror = []
        for clinic in clinics_mirror or []:
            raw_id = str((clinic or {}).get("id") or "").strip()
            ck = normalize_clinic_key(raw_id)
            if not ck:
                continue
            vis = model_store.list_user_visible_model_versions(ck)
            try:
                vmax = int(str(os.getenv("VETAI_MLAIR_MIRROR_CLINIC_MAX_VERSIONS", "0") or "0").strip() or "0")
            except ValueError:
                vmax = 0
            if vmax > 0:
                vis = vis[:vmax]
            for vname in vis:
                mdir = model_store.resolve_model_dir(vname, ck)
                if not mdir or not vname:
                    continue
                wk = (ck, vname, os.path.normpath(mdir))
                if wk in seen_work:
                    continue
                seen_work.add(wk)
                work.append((ck, vname, mdir))

    for clinic_key, version_name, model_dir in work:
        c, s, p, info = _sync_disk_version_to_mlair(
            cfg,
            clinic_id_for_scope=clinic_key,
            version_name=version_name,
            model_dir=model_dir,
            model_name=model_name,
            model_desc=model_desc,
            default_stage=default_stage,
            active_global_version=active_global_version,
            seen_models=seen_models,
            dedupe_artifact_scope=dedupe_artifact_scope,
            pipeline_map_targets=pipeline_map_targets,
        )
        created_count += c
        skipped_count += s
        promoted_count += p
        if info:
            synced_scopes.append(info)

    # Ensure every clinic from business DB/catalog has an MLAir model scope,
    # even before the first clinic-specific model artifact is uploaded.
    try:
        clinics, _source = get_clinics_for_mlops()
    except Exception:
        clinics = []
    ensure_empty_scopes = os.getenv("MLAIR_ENSURE_CLINIC_SCOPES", "true").lower() == "true"
    if ensure_empty_scopes:
        for clinic in clinics:
            clinic_key = str((clinic or {}).get("id") or "").strip()
            if not clinic_key:
                continue
            scope = _resolve_scope(cfg, clinic_key)
            tenant_id = scope["tenant_id"]
            project_id = scope["project_id"]
            scope_model_name = _scoped_model_name(model_name, clinic_key if cfg["model_scope_per_clinic"] else None)
            model_key = f"{tenant_id}:{project_id}:{scope_model_name}"
            if model_key in seen_models:
                continue
            listed = _list_models(cfg, tenant_id, project_id)
            items = listed.get("items") if isinstance(listed, dict) else []
            items = items if isinstance(items, list) else []
            model_id = ""
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("name") or "").strip() == scope_model_name:
                    model_id = str(item.get("model_id") or item.get("id") or "").strip()
                    if model_id:
                        break
            if not model_id:
                created = _create_model(
                    cfg,
                    tenant_id,
                    project_id,
                    scope_model_name,
                    f"{model_desc} (clinic scope placeholder)",
                )
                model_id = str(created.get("model_id") or created.get("id") or "").strip()
                if model_id:
                    created_models += 1
            if model_id:
                seen_models[model_key] = model_id
                _note_pipeline_map_target(pipeline_map_targets, clinic_key, model_id)

    # Default (global) project: same placeholder behavior as clinics so pipeline mapping has a model_id.
    if ensure_empty_scopes:
        gscope = _resolve_scope(cfg, None)
        g_tenant_id = gscope["tenant_id"]
        g_project_id = gscope["project_id"]
        g_scope_model_name = _scoped_model_name(model_name, None)
        g_model_key = f"{g_tenant_id}:{g_project_id}:{g_scope_model_name}"
        if g_model_key not in seen_models:
            g_model_id = ""
            g_listed = _list_models(cfg, g_tenant_id, g_project_id)
            g_items = g_listed.get("items") if isinstance(g_listed, dict) else []
            g_items = g_items if isinstance(g_items, list) else []
            for item in g_items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("name") or "").strip() == g_scope_model_name:
                    g_model_id = str(item.get("model_id") or item.get("id") or "").strip()
                    if g_model_id:
                        break
            if not g_model_id:
                g_created = _create_model(
                    cfg,
                    g_tenant_id,
                    g_project_id,
                    g_scope_model_name,
                    f"{model_desc} (global scope placeholder)",
                )
                g_model_id = str(g_created.get("model_id") or g_created.get("id") or "").strip()
                if g_model_id:
                    created_models += 1
            if g_model_id:
                seen_models[g_model_key] = g_model_id
                _note_pipeline_map_target(pipeline_map_targets, None, g_model_id)

    _fill_pipeline_map_targets_from_seen_models(
        cfg,
        seen_models=seen_models,
        clinics=clinics,
        discovered=discovered,
        work=work,
        pipeline_map_targets=pipeline_map_targets,
    )

    # One pipeline version per project (global + clinic) so MLAir can resolve latest DAG for training runs.
    clinic_scopes_for_pipeline: set[tuple[str, str]] = set()
    g0 = _resolve_scope(cfg, None)
    clinic_scopes_for_pipeline.add((g0["tenant_id"], g0["project_id"]))
    for clinic in clinics:
        ck = str((clinic or {}).get("id") or "").strip()
        if not ck:
            continue
        sc = _resolve_scope(cfg, ck)
        clinic_scopes_for_pipeline.add((sc["tenant_id"], sc["project_id"]))
    for row in discovered:
        ck = str(row.get("clinic_key") or "").strip() or None
        if not ck:
            continue
        sc = _resolve_scope(cfg, ck)
        clinic_scopes_for_pipeline.add((sc["tenant_id"], sc["project_id"]))

    for mk in seen_models:
        bits = str(mk).split(":", 2)
        if len(bits) == 3:
            clinic_scopes_for_pipeline.add((bits[0], bits[1]))

    ensure_clinic_pipelines = os.getenv("MLAIR_ENSURE_CLINIC_TRAINING_PIPELINES", "true").lower() == "true"
    explicit_mlair_pipeline = str(os.getenv("MLAIR_PIPELINE_ID", "") or "").strip()
    pipeline_id = str(cfg.get("pipeline_id") or "").strip()
    created_pipeline_versions = 0
    clinic_pipeline_scope_count = sum(1 for _, p in clinic_scopes_for_pipeline if str(p).startswith("clinic_"))
    if ensure_clinic_pipelines and pipeline_id:
        for tid, pid in sorted(clinic_scopes_for_pipeline):
            try:
                if _ensure_clinic_training_pipeline(cfg, tid, pid, pipeline_id):
                    created_pipeline_versions += 1
            except Exception as exc:
                logger.warning(
                    "Failed to ensure training pipeline tenant=%s project=%s pipeline_id=%s: %s",
                    tid,
                    pid,
                    pipeline_id,
                    exc,
                )

    pipeline_mappings_upserted = 0
    pipeline_mapping_errors: list[Dict[str, Any]] = []
    if pipeline_id and _env_truthy("VETAI_MLAIR_MAP_PIPELINE_ON_SYNC", "true") and pipeline_map_targets:
        for ck_map, mid_map in sorted(pipeline_map_targets, key=lambda x: (x[0] or "", x[1])):
            try:
                registry_put_pipeline_mapping(
                    clinic_id=ck_map,
                    model_id=mid_map,
                    pipeline_id=pipeline_id,
                )
                pipeline_mappings_upserted += 1
            except Exception as exc:  # noqa: BLE001
                pipeline_mapping_errors.append(
                    {"clinic_id": ck_map, "model_id": mid_map, "error": str(exc)}
                )

    return {
        "base_model_name": model_name,
        "discovered_versions": len(discovered),
        "sync_work_items": len(work),
        "mirror_global_into_clinic_projects": mirror_globals,
        "ensured_clinic_scopes": len(clinics) if ensure_empty_scopes else 0,
        "created_models": created_models,
        "created_versions": created_count,
        "promoted_versions": promoted_count,
        "skipped_versions": skipped_count,
        "synced_scopes": synced_scopes,
        "training_pipeline_id_used": pipeline_id,
        "used_default_training_pipeline_id": bool(pipeline_id and not explicit_mlair_pipeline),
        "created_pipeline_versions": created_pipeline_versions,
        "clinic_pipeline_scopes": clinic_pipeline_scope_count,
        "pipeline_scopes_total": len(clinic_scopes_for_pipeline),
        "pipeline_mappings_upserted": pipeline_mappings_upserted,
        "pipeline_mapping_errors": pipeline_mapping_errors,
    }


def _require_mlair_enabled() -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg.get("enabled"):
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")
    return cfg


def _registry_scope_ids(clinic_id: Optional[str]) -> tuple[str, str]:
    cfg = _require_mlair_enabled()
    sc = _resolve_scope(cfg, str(clinic_id).strip() if clinic_id else None)
    return str(sc["tenant_id"]).strip(), str(sc["project_id"]).strip()


def _registry_model_base_url(cfg: Dict[str, Any], tenant_id: str, project_id: str, model_id: str) -> str:
    tq = quote(str(tenant_id).strip(), safe="")
    pq = quote(str(project_id).strip(), safe="")
    mq = quote(str(model_id).strip(), safe="")
    return f"{str(cfg.get('base_url') or '').rstrip('/')}/v1/tenants/{tq}/projects/{pq}/models/{mq}"


def registry_list_models(
    *,
    clinic_id: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    return _list_models(cfg, tid, pid, limit=limit, offset=offset)


def registry_get_model(*, clinic_id: Optional[str], model_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id)
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_get_model_status(*, clinic_id: Optional[str], model_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id) + "/status"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_get_resolved_pipeline(*, clinic_id: Optional[str], model_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id) + "/resolved-pipeline"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_put_pipeline_mapping(*, clinic_id: Optional[str], model_id: str, pipeline_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id) + "/pipeline-mapping"
    body = {"pipeline_id": str(pipeline_id).strip()}
    return _request_json("PUT", url, body=body, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_get_trigger_policy(*, clinic_id: Optional[str], model_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id) + "/trigger-policy"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_put_trigger_policy(
    *,
    clinic_id: Optional[str],
    model_id: str,
    trigger_mode: str = "manual",
    debounce_minutes: int = 10,
    schedule_cron: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id) + "/trigger-policy"
    body: Dict[str, Any] = {
        "trigger_mode": str(trigger_mode or "manual").strip(),
        "debounce_minutes": int(debounce_minutes),
        "schedule_cron": schedule_cron,
    }
    return _request_json("PUT", url, body=body, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_list_model_versions(*, clinic_id: Optional[str], model_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    items = _list_model_versions_all_items(cfg, tid, pid, model_id)
    return {"items": items}


def registry_preview_next_artifact_uri(*, clinic_id: Optional[str], model_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id) + "/next-artifact-uri"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_promote_model_version(
    *,
    clinic_id: Optional[str],
    model_id: str,
    version: int,
    stage: str = "production",
) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id) + "/promote"
    body = {"version": int(version), "stage": str(stage or "production").strip()}
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_get_version_approval(*, clinic_id: Optional[str], model_id: str, version: int) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = f"{_registry_model_base_url(cfg, tid, pid, model_id)}/versions/{int(version)}/approval"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_put_version_approval(
    *,
    clinic_id: Optional[str],
    model_id: str,
    version: int,
    approval_status: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = f"{_registry_model_base_url(cfg, tid, pid, model_id)}/versions/{int(version)}/approval"
    body: Dict[str, Any] = {"approval_status": str(approval_status).strip()}
    if reason is not None and str(reason).strip():
        body["reason"] = str(reason).strip()
    return _request_json("PUT", url, body=body, headers=_headers(cfg), timeout=float(cfg["timeout"]))


def registry_delete_model_version(*, clinic_id: Optional[str], model_id: str, version: int) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = f"{_registry_model_base_url(cfg, tid, pid, model_id)}/versions/{int(version)}"
    return _request_delete_json(url, cfg)


def registry_delete_model(*, clinic_id: Optional[str], model_id: str) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    url = _registry_model_base_url(cfg, tid, pid, model_id)
    return _request_delete_json(url, cfg)


def registry_create_model(
    *,
    clinic_id: Optional[str],
    name: str,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    return _create_model(cfg, tid, pid, str(name).strip(), description)


def registry_create_model_version(
    *,
    clinic_id: Optional[str],
    model_id: str,
    artifact_uri: str,
    run_id: Optional[str] = None,
    stage: str = "staging",
) -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    tid, pid = _registry_scope_ids(clinic_id)
    return register_model_version_for_scope(
        tenant_id=tid,
        project_id=pid,
        model_id=str(model_id).strip(),
        artifact_uri=str(artifact_uri).strip(),
        run_id=run_id,
        stage=stage,
    )


def registry_list_plugins() -> Dict[str, Any]:
    cfg = _require_mlair_enabled()
    url = f"{str(cfg.get('base_url') or '').rstrip('/')}/v1/plugins"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=float(cfg["timeout"]))
