from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug
from ai_service.app.infrastructure.storage import model_store


def _cfg() -> Dict[str, Any]:
    clinic_project_map = os.getenv("MLAIR_CLINIC_PROJECT_MAP_JSON", "").strip()
    clinic_tenant_map = os.getenv("MLAIR_CLINIC_TENANT_MAP_JSON", "").strip()
    clinic_model_alias_map = os.getenv("MLAIR_CLINIC_MODEL_ALIAS_MAP_JSON", "").strip()
    return {
        "enabled": os.getenv("MLAIR_ENABLED", "false").lower() == "true",
        "base_url": os.getenv("MLAIR_API_BASE_URL", "http://localhost:8080").rstrip("/"),
        "tenant_id": os.getenv("MLAIR_TENANT_ID", "default"),
        "project_id": os.getenv("MLAIR_PROJECT_ID", "default_project"),
        "pipeline_id": os.getenv("MLAIR_PIPELINE_ID", "vet_ai_training_pipeline"),
        "token": os.getenv("MLAIR_AUTH_TOKEN", ""),
        "timeout": float(os.getenv("MLAIR_TIMEOUT_SECONDS", "10")),
        "clinic_project_map": json.loads(clinic_project_map) if clinic_project_map else {},
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


def config_summary() -> Dict[str, Any]:
    cfg = _cfg()
    return {
        "enabled": cfg["enabled"],
        "base_url": cfg["base_url"],
        "tenant_id": cfg["tenant_id"],
        "project_id": cfg["project_id"],
        "pipeline_id": cfg["pipeline_id"],
        "has_token": bool(cfg["token"]),
        "model_scope_per_clinic": cfg["model_scope_per_clinic"],
    }


def _resolve_scope(cfg: Dict[str, Any], clinic_id: Optional[str]) -> Dict[str, str]:
    clinic_key = str(clinic_id).strip() if clinic_id else ""
    tenant_id = cfg["tenant_id"]
    project_id = cfg["project_id"]
    if clinic_key:
        tenant_id = str(cfg["clinic_tenant_map"].get(clinic_key) or tenant_id)
        mapped_project = str(cfg["clinic_project_map"].get(clinic_key) or "").strip()
        # DB-driven default: if clinic is present and no explicit map, derive
        # one stable project id from clinic id so data is not forced into global.
        project_id = mapped_project or f"clinic_{clinic_dir_slug(clinic_key)}"
    return {"tenant_id": tenant_id, "project_id": project_id}


def _clinic_model_suffix(cfg: Dict[str, Any], clinic_key: str) -> str:
    alias = str(cfg.get("clinic_model_alias_map", {}).get(clinic_key) or "").strip()
    if alias:
        return clinic_dir_slug(alias)
    return clinic_dir_slug(clinic_key)


def trigger_training_run(
    *,
    idempotency_key: str,
    pipeline_id: str | None = None,
    clinic_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    pid = pipeline_id or cfg["pipeline_id"]
    scope = _resolve_scope(cfg, clinic_id)
    url = f"{cfg['base_url']}/v1/tenants/{scope['tenant_id']}/projects/{scope['project_id']}/runs"
    body: Dict[str, Any] = {"pipeline_id": pid, "idempotency_key": idempotency_key}
    if clinic_id:
        body["context"] = {"clinic_id": clinic_id}
    if context:
        merged = dict(body.get("context") or {})
        merged.update(context)
        body["context"] = merged
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def get_run(run_id: str) -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    url = f"{cfg['base_url']}/v1/tenants/{cfg['tenant_id']}/projects/{cfg['project_id']}/runs/{run_id}"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _list_models(cfg: Dict[str, Any], tenant_id: str, project_id: str) -> Dict[str, Any]:
    url = f"{cfg['base_url']}/v1/tenants/{tenant_id}/projects/{project_id}/models?limit=200"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _create_model(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    name: str,
    description: str | None = None,
) -> Dict[str, Any]:
    url = f"{cfg['base_url']}/v1/tenants/{tenant_id}/projects/{project_id}/models"
    body: Dict[str, Any] = {"name": name}
    if description:
        body["description"] = description
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


def _list_model_versions(cfg: Dict[str, Any], tenant_id: str, project_id: str, model_id: str) -> Dict[str, Any]:
    url = f"{cfg['base_url']}/v1/tenants/{tenant_id}/projects/{project_id}/models/{model_id}/versions"
    return _request_json("GET", url, body=None, headers=_headers(cfg), timeout=cfg["timeout"])


def _create_model_version(
    cfg: Dict[str, Any],
    tenant_id: str,
    project_id: str,
    model_id: str,
    artifact_uri: str,
    stage: str,
) -> Dict[str, Any]:
    url = f"{cfg['base_url']}/v1/tenants/{tenant_id}/projects/{project_id}/models/{model_id}/versions"
    body = {"artifact_uri": artifact_uri, "stage": stage}
    return _request_json("POST", url, body=body, headers=_headers(cfg), timeout=cfg["timeout"])


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


def sync_all_models_to_mlair() -> Dict[str, Any]:
    cfg = _cfg()
    if not cfg["enabled"]:
        raise RuntimeError("MLAir integration disabled (set MLAIR_ENABLED=true)")

    model_name = os.getenv("MLAIR_MODEL_NAME", "vet-ai")
    model_desc = os.getenv("MLAIR_MODEL_DESCRIPTION", "Synced from Vet-AI model store")
    default_stage = os.getenv("MLAIR_MODEL_STAGE", "staging")
    discovered = model_store.list_model_versions_with_scope()
    created_count = 0
    skipped_count = 0
    synced_scopes = []
    seen_models: Dict[str, str] = {}

    for row in discovered:
        clinic_key = str(row.get("clinic_key") or "").strip() or None
        model_dir = str(row.get("model_dir") or "").strip()
        if not model_dir:
            continue
        artifact_uri = f"file://{model_dir}"
        scope = _resolve_scope(cfg, clinic_key)
        tenant_id = scope["tenant_id"]
        project_id = scope["project_id"]
        scope_model_name = model_name
        if cfg["model_scope_per_clinic"] and clinic_key:
            scope_model_name = f"{model_name}-{_clinic_model_suffix(cfg, clinic_key)}"

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

        versions_payload = _list_model_versions(cfg, tenant_id, project_id, model_id)
        existing_uris = set()
        existing_items = versions_payload.get("items") if isinstance(versions_payload, dict) else []
        if isinstance(existing_items, list):
            for item in existing_items:
                if not isinstance(item, dict):
                    continue
                uri = str(item.get("artifact_uri") or "").strip()
                if uri:
                    existing_uris.add(uri)

        if artifact_uri in existing_uris:
            skipped_count += 1
            continue

        _create_model_version(cfg, tenant_id, project_id, model_id, artifact_uri, default_stage)
        created_count += 1
        synced_scopes.append(
            {
                "clinic_key": clinic_key or "global",
                "tenant_id": tenant_id,
                "project_id": project_id,
                "model_name": scope_model_name,
            }
        )

    return {
        "base_model_name": model_name,
        "discovered_versions": len(discovered),
        "created_versions": created_count,
        "skipped_versions": skipped_count,
        "synced_scopes": synced_scopes,
    }
