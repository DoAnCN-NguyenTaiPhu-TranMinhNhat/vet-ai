"""
HTTP client for MLAir (control plane) — registry import after training.

Mirrors the role of MLflow logging: push a trained artifact bundle into MLAir so runs/models
stay visible in the MLAir UI. Full pipeline/task runs live in MLAir separately; this path is
the minimal vet-ai ↔ MLAir bridge for Docker Compose and external stacks.

Env:
  MLAIR_API_BASE_URL   e.g. http://ml-air-api:8080 — if unset, all calls are no-ops.
  MLAIR_API_TOKEN    Bearer token (maintainer+ on target tenant/project). Defaults to admin-token.
  MLAIR_TENANT_ID    default: default (matches MLAir quickstart static tokens)
  MLAIR_PROJECT_GLOBAL        default: default_project
  MLAIR_PROJECT_CLINIC_PREFIX default: clinic_  → project clinic_<slug> per ml-air docs
  MLAIR_IMPORT_STAGE          default: staging
  MLAIR_IMPORT_INCLUDE_RUN_ID default: unset — do **not** send multipart ``run_id`` unless ``1``; synthetic ids
        (``vet-ai-disk-*`` / ``vet-ai-training-*``) are not ``runs`` rows in MLAir and trigger ``model_versions_run_id_fkey`` (HTTP 500).
  MLAIR_TIMEOUT_SECONDS       default: 120
  MLAIR_REGISTRY_SYNC_AT_STARTUP  default: 1 — POST /v1/tenants/.../projects/registry for catalog + global project
  MLAIR_REGISTRY_GLOBAL_PROJECT_NAME  optional display name for the global MLAir project row
  MLAIR_REGISTRY_BUST_CLINIC_CACHE  default: 1 — refetch clinic catalog before each full sync
  MLAIR_REGISTRY_BOOTSTRAP_MAX_WAIT_SECONDS  default: 120 — retry bootstrap until success or timeout
  MLAIR_REGISTRY_BOOTSTRAP_INTERVAL_SECONDS  default: 5 — sleep between bootstrap attempts
  MLAIR_REGISTRY_RESYNC_SECONDS  default: 120 — periodic full sync (0 disables)
  MLAIR_STUB_PIPELINE_AT_REGISTRY  default: 0 — optional empty pipeline version (UI-only placeholder)
  MLAIR_STUB_PIPELINE_ID           default: demo_pipeline (only if stub enabled)
  MLAIR_SEED_DEMO_EXECUTOR         default: 0 — set 1 to create MLAir official seed_demo DAG + plugins
  MLAIR_SEED_PIPELINE_ID           default: fail_once_demo_pipeline (same as ml-air scripts/seed_demo.py)
  MLAIR_SEED_PIPELINE_CONFIG_JSON  optional full {"tasks":[...]} override
  MLAIR_SEED_DEMO_EXECUTOR_RUN     default: 0 — set 1 to POST one maintainer run (idempotent key)
  MLAIR_SEED_DEMO_EXECUTOR_PROJECTS  comma project_ids to seed (default: global MLAIR_PROJECT_GLOBAL only)
  MLAIR_TRAINING_TRACKING         default: 0 — set 1 to POST an MLAir run + experiments/params/metrics after each training
  MLAIR_TRACKING_PIPELINE_ID      default: vetai_local_training (empty-task pipeline version; scheduler uses plugin_name)
  MLAIR_TRACKING_PLUGIN_NAME      default: echo_tracking (fallback plugin for synthetic task:1)
  MLAIR_TRACKING_EXPERIMENT_NAME  default: Vet-AI continuous training (reuse by name when MLAIR_TRACKING_EXPERIMENT_ID unset)
  MLAIR_TRACKING_EXPERIMENT_ID    optional fixed experiment UUID from MLAir
  MLAIR_TRACKING_POLL_SECONDS     default: 90 — wait for run terminal state after POST /runs before logging metrics
  MLAIR_TRACKING_FILE_ARTIFACT_URI default: unset — do **not** POST ``file://`` URIs for local vet-ai paths (they are
        not visible inside mlair-api; UI shows "Selected files: 0"). Set to ``1`` only if mlair-api shares the same
        absolute path (e.g. identical bind mount).
  MLAIR_DISK_MODEL_IMPORT_AT_REGISTRY  default: 1 — after project registry, sync on-disk model.pkl bundles into MLAir
        (imports **missing** folders such as ``v2`` even when ``v1`` already exists; duplicate imports are skipped)
  MLAIR_DISK_IMPORT_CLINIC_FALLBACK_GLOBAL  default: 1 — if a clinic has no local model.pkl dirs, import global on-disk
        bundles into that clinic's MLAir logical model so versions are not empty
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key

logger = logging.getLogger(__name__)


def mlair_api_base_url() -> str:
    return os.getenv("MLAIR_API_BASE_URL", "").strip().rstrip("/")


def mlair_enabled() -> bool:
    return bool(mlair_api_base_url())


def mlair_api_token() -> str:
    return (
        os.getenv("MLAIR_API_TOKEN", "").strip()
        or os.getenv("ML_AIR_TRACKING_TOKEN", "").strip()
        or "admin-token"
    )


def mlair_tenant_id() -> str:
    return os.getenv("MLAIR_TENANT_ID", "default").strip() or "default"


def mlair_project_for_clinic(clinic_key: Optional[str]) -> str:
    ck = normalize_clinic_key(clinic_key)
    if ck is None:
        return os.getenv("MLAIR_PROJECT_GLOBAL", "default_project").strip() or "default_project"
    prefix = os.getenv("MLAIR_PROJECT_CLINIC_PREFIX", "clinic_").strip() or "clinic_"
    return f"{prefix}{clinic_dir_slug(ck)}"


def mlair_logical_model_name(clinic_key: Optional[str]) -> str:
    ck = normalize_clinic_key(clinic_key)
    if ck is None:
        return os.getenv("MLAIR_MODEL_NAME_GLOBAL", "vetai-diagnosis-global").strip() or "vetai-diagnosis-global"
    base = os.getenv("MLAIR_MODEL_NAME_CLINIC_PREFIX", "vetai-diagnosis-clinic").strip() or "vetai-diagnosis-clinic"
    return f"{base}-{clinic_dir_slug(ck)}"


def _timeout() -> float:
    try:
        return float(os.getenv("MLAIR_TIMEOUT_SECONDS", "120"))
    except ValueError:
        return 120.0


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {mlair_api_token()}"}


def _registry_sync_at_startup_enabled() -> bool:
    raw = os.getenv("MLAIR_REGISTRY_SYNC_AT_STARTUP", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _bust_clinic_cache_before_sync() -> None:
    if os.getenv("MLAIR_REGISTRY_BUST_CLINIC_CACHE", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    from ai_service.app.domain.services.clinic_catalog_service import bust_clinic_catalog_cache

    bust_clinic_catalog_cache()


def _stub_pipeline_at_registry_enabled() -> bool:
    return os.getenv("MLAIR_STUB_PIPELINE_AT_REGISTRY", "0").strip().lower() not in ("0", "false", "no", "off")


def stub_pipeline_id() -> str:
    return (os.getenv("MLAIR_STUB_PIPELINE_ID", "demo_pipeline") or "demo_pipeline").strip() or "demo_pipeline"


def ensure_mlair_pipeline_stub_if_empty(tenant_id: str, project_id: str) -> Dict[str, Any]:
    """
    MLAir lists pipelines only from runs + pipeline_versions. After registering a project,
    create one empty pipeline version so the Pipelines page is not blank (idempotent).
    """
    base = mlair_api_base_url()
    if not base:
        return {"status": "skipped", "reason": "MLAIR_API_BASE_URL unset"}
    if not _stub_pipeline_at_registry_enabled():
        return {"status": "skipped", "reason": "MLAIR_STUB_PIPELINE_AT_REGISTRY disabled"}
    tid = str(tenant_id or "").strip() or mlair_tenant_id()
    pid = str(project_id or "").strip()
    if not pid:
        return {"status": "skipped", "reason": "empty_project_id"}
    pipe = stub_pipeline_id()
    safe_pipe = quote(pipe, safe="")
    list_url = f"{base}/v1/tenants/{tid}/projects/{quote(pid, safe='')}/pipelines?limit=200"
    create_url = f"{base}/v1/tenants/{tid}/projects/{quote(pid, safe='')}/pipelines/{safe_pipe}/versions"
    try:
        tmo = min(_timeout(), 30.0)
        with httpx.Client(timeout=tmo) as client:
            lr = client.get(list_url, headers=_headers())
            if lr.status_code in (404, 405):
                return {"status": "skipped", "reason": "mlair_pipelines_not_supported", "project_id": pid}
            lr.raise_for_status()
            body = lr.json() if lr.content else {}
            items = body.get("items") if isinstance(body, dict) else None
            if isinstance(items, list) and len(items) > 0:
                return {"status": "skipped", "reason": "pipelines_already_present", "project_id": pid, "count": len(items)}
            pr = client.post(
                create_url,
                headers={**_headers(), "Content-Type": "application/json"},
                json={"config": {}},
            )
            if pr.status_code in (404, 405):
                return {"status": "skipped", "reason": "mlair_pipeline_versions_not_supported", "project_id": pid}
            pr.raise_for_status()
            return {"status": "ok", "project_id": pid, "pipeline_id": pipe, "mlair": pr.json() if pr.content else {}}
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1500] if exc.response is not None else ""
        logger.warning("MLAir pipeline stub HTTP error project=%s: %s body=%s", pid, exc, detail)
        return {"status": "error", "project_id": pid, "error": str(exc), "http_detail": detail}
    except Exception as exc:
        logger.warning("MLAir pipeline stub failed project=%s: %s", pid, exc)
        return {"status": "error", "project_id": pid, "error": str(exc)}


def _seed_demo_executor_enabled() -> bool:
    return os.getenv("MLAIR_SEED_DEMO_EXECUTOR", "0").strip().lower() in ("1", "true", "yes", "on")


def _seed_demo_executor_run_enabled() -> bool:
    return os.getenv("MLAIR_SEED_DEMO_EXECUTOR_RUN", "0").strip().lower() in ("1", "true", "yes", "on")


def _seed_demo_executor_project_allowlist() -> set[str]:
    raw = os.getenv("MLAIR_SEED_DEMO_EXECUTOR_PROJECTS", "").strip()
    if raw:
        return {x.strip() for x in raw.split(",") if x.strip()}
    return {mlair_project_for_clinic(None)}


def _default_seed_pipeline_config() -> dict[str, Any]:
    raw = os.getenv("MLAIR_SEED_PIPELINE_CONFIG_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(data.get("tasks"), list):
                return data
        except json.JSONDecodeError:
            logger.warning("MLAIR_SEED_PIPELINE_CONFIG_JSON is not valid JSON; using built-in seed_demo tasks")
    return {
        "tasks": [
            {"id": "extract", "plugin": "app_etl_adapter"},
            {"id": "transform", "plugin": "app_etl_adapter", "depends_on": ["extract"]},
            {"id": "train", "plugin": "echo_tracking", "depends_on": ["transform"]},
        ]
    }


def seed_pipeline_id() -> str:
    return (os.getenv("MLAIR_SEED_PIPELINE_ID", "fail_once_demo_pipeline") or "fail_once_demo_pipeline").strip()


def seed_mlair_demo_executor_track(tenant_id: str, project_id: str) -> Dict[str, Any]:
    """
    Create the same pipeline version + optional run as ml-air ``scripts/seed_demo.py`` (real plugins:
    app_etl_adapter, echo_tracking). Idempotent: skips version create if versions exist; run uses stable idempotency_key.
    """
    base = mlair_api_base_url()
    if not base:
        return {"status": "skipped", "reason": "MLAIR_API_BASE_URL unset"}
    if not _seed_demo_executor_enabled():
        return {"status": "skipped", "reason": "MLAIR_SEED_DEMO_EXECUTOR off"}
    tid = str(tenant_id or "").strip() or mlair_tenant_id()
    pid = str(project_id or "").strip()
    if not pid:
        return {"status": "skipped", "reason": "empty_project_id"}
    if pid not in _seed_demo_executor_project_allowlist():
        return {"status": "skipped", "reason": "project_not_in_seed_allowlist", "project_id": pid}

    pipe = seed_pipeline_id()
    safe_proj = quote(pid, safe="")
    safe_pipe = quote(pipe, safe="")
    versions_url = f"{base}/v1/tenants/{tid}/projects/{safe_proj}/pipelines/{safe_pipe}/versions"
    runs_url = f"{base}/v1/tenants/{tid}/projects/{safe_proj}/runs"
    cfg = _default_seed_pipeline_config()
    out: Dict[str, Any] = {"status": "pending", "tenant_id": tid, "project_id": pid, "pipeline_id": pipe}

    try:
        tmo = min(_timeout(), 60.0)
        with httpx.Client(timeout=tmo) as client:
            lv = client.get(f"{versions_url}?limit=5&offset=0", headers=_headers())
            if lv.status_code in (404, 405):
                out["status"] = "skipped"
                out["reason"] = "mlair_pipeline_versions_not_supported"
                return out
            lv.raise_for_status()
            vbody = lv.json() if lv.content else {}
            vitems = vbody.get("items") if isinstance(vbody, dict) else None
            version_created = False
            if not (isinstance(vitems, list) and len(vitems) > 0):
                pr = client.post(
                    versions_url,
                    headers={**_headers(), "Content-Type": "application/json"},
                    json={"config": cfg},
                )
                if pr.status_code in (404, 405):
                    out["status"] = "skipped"
                    out["reason"] = "mlair_pipeline_version_post_not_supported"
                    return out
                pr.raise_for_status()
                version_created = True
                out["pipeline_version"] = pr.json() if pr.content else {}
            else:
                out["pipeline_version"] = "existing"

            out["version_created"] = version_created

            if not _seed_demo_executor_run_enabled():
                out["status"] = "ok"
                out["run"] = "skipped"
                out["run_reason"] = "MLAIR_SEED_DEMO_EXECUTOR_RUN off"
                return out

            run_tag = str(int(time.time() * 1000))
            idem = f"vet-ai-exec-seed-{tid}-{pid}-{pipe}"
            plugin_context: Dict[str, Any] = {
                "params": {"demo_source": "vet_ai_seed", "run_tag": run_tag},
                "metrics": {"demo_score": {"step": 1, "value": 0.93}},
                "artifacts": [{"path": f"demo/{run_tag}/model.pkl", "uri": f"s3://mlair/demo/{run_tag}/model.pkl"}],
                "lineage": {
                    "inputs": [
                        {"name": "raw_data", "version": f"v{run_tag}", "uri": f"s3://mlair/demo/{run_tag}/raw.parquet"}
                    ],
                    "outputs": [
                        {
                            "name": "features",
                            "version": f"v{run_tag}",
                            "uri": f"s3://mlair/demo/{run_tag}/features.parquet",
                        }
                    ],
                },
            }
            trigger = {
                "pipeline_id": pipe,
                "idempotency_key": idem,
                "plugin_name": "echo_tracking",
                "context": plugin_context,
                "use_latest_pipeline_version": True,
            }
            rr = client.post(
                runs_url,
                headers={**_headers(), "Content-Type": "application/json"},
                json=trigger,
            )
            rr.raise_for_status()
            out["run"] = rr.json() if rr.content else {}
            out["status"] = "ok"
            return out
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:2000] if exc.response is not None else ""
        logger.warning("MLAir demo executor seed HTTP error: %s body=%s", exc, detail)
        out["status"] = "error"
        out["error"] = str(exc)
        out["http_detail"] = detail
        return out
    except Exception as exc:
        logger.warning("MLAir demo executor seed failed: %s", exc)
        out["status"] = "error"
        out["error"] = str(exc)
        return out


def _mlair_register_result_is_transient(r: Dict[str, Any]) -> bool:
    if r.get("status") != "error":
        return False
    msg = f"{r.get('error', '')}|{r.get('http_detail', '')}".lower()
    needles = (
        "connect",
        "connection refused",
        "timed out",
        "timeout",
        "unreachable",
        "name or service not known",
        "nodename nor servname",
        "503",
        "502",
        "504",
        "temporary failure",
    )
    return any(n in msg for n in needles)


def register_mlair_project(
    *,
    tenant_id: str,
    project_id: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upsert a logical project in MLAir (tenant_projects) via POST .../projects/registry.
    Older MLAir builds without this route return skipped (404/405) without failing callers.
    """
    base = mlair_api_base_url()
    if not base:
        return {"status": "skipped", "reason": "MLAIR_API_BASE_URL unset"}
    pid = str(project_id or "").strip()
    if not pid:
        return {"status": "skipped", "reason": "empty_project_id"}
    tid = str(tenant_id or "").strip() or mlair_tenant_id()
    payload: Dict[str, Any] = {"project_id": pid}
    if name is not None and str(name).strip():
        payload["name"] = str(name).strip()
    url = f"{base}/v1/tenants/{tid}/projects/registry"
    try:
        tmo = min(_timeout(), 30.0)
        with httpx.Client(timeout=tmo) as client:
            r = client.post(
                url,
                headers={**_headers(), "Content-Type": "application/json"},
                json=payload,
            )
            if r.status_code in (404, 405):
                return {"status": "skipped", "reason": "mlair_registry_not_supported", "project_id": pid}
            r.raise_for_status()
            body: Dict[str, Any] = {}
            if r.content:
                try:
                    parsed = r.json()
                    if isinstance(parsed, dict):
                        body = parsed
                except Exception:
                    body = {}
            return {"status": "ok", "project_id": pid, "mlair": body}
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1500] if exc.response is not None else ""
        logger.warning("MLAir registry HTTP error for %s: %s body=%s", pid, exc, detail)
        return {"status": "error", "project_id": pid, "error": str(exc), "http_detail": detail}
    except Exception as exc:
        logger.warning("MLAir registry failed for %s: %s", pid, exc)
        return {"status": "error", "project_id": pid, "error": str(exc)}


def _sync_mlair_project_registry_core() -> Dict[str, Any]:
    """Register global + per-clinic projects in MLAir (no MLAIR_REGISTRY_SYNC_AT_STARTUP gate)."""
    if not mlair_enabled():
        return {"status": "skipped", "reason": "MLAIR_API_BASE_URL unset"}

    _bust_clinic_cache_before_sync()

    from ai_service.app.domain.services.clinic_catalog_service import get_clinics_for_mlops

    tenant_id = mlair_tenant_id()
    clinics, source = get_clinics_for_mlops()
    global_pid = mlair_project_for_clinic(None)
    global_name = (os.getenv("MLAIR_REGISTRY_GLOBAL_PROJECT_NAME") or "").strip() or "Vet-AI (global)"

    entries: list[Dict[str, Any]] = [{"project_id": global_pid, "name": global_name, "clinic_key": None}]
    for row in clinics:
        if not isinstance(row, dict):
            continue
        ck = normalize_clinic_key(row.get("id"))
        if not ck:
            continue
        pid = mlair_project_for_clinic(ck)
        disp = row.get("name")
        disp_s = str(disp).strip() if disp is not None else ""
        entries.append({"project_id": pid, "name": disp_s or pid, "clinic_key": ck})

    seen: set[str] = set()
    deduped: list[Dict[str, Any]] = []
    for e in entries:
        p = e["project_id"]
        if p in seen:
            continue
        seen.add(p)
        deduped.append(e)

    results: list[Dict[str, Any]] = []
    for e in deduped:
        results.append(
            register_mlair_project(tenant_id=tenant_id, project_id=e["project_id"], name=e["name"])
        )

    any_err = any(r.get("status") == "error" for r in results)

    disk_model_imports: list[Dict[str, Any]] = []
    if _disk_model_import_at_registry_enabled() and not any_err:
        try:
            tmo = min(_timeout(), 240.0)
            with httpx.Client(timeout=tmo) as client:
                for e, reg in zip(deduped, results):
                    if reg.get("status") == "error":
                        continue
                    disk_model_imports.append(
                        bootstrap_mlair_models_from_disk_for_project(
                            client,
                            tenant_id=tenant_id,
                            project_id=str(e["project_id"]),
                            clinic_key=e.get("clinic_key"),
                        )
                    )
        except Exception as exc:
            logger.warning("MLAir disk model import sweep failed: %s", exc)
            disk_model_imports.append({"status": "error", "error": str(exc)})

    pipeline_stubs: list[Dict[str, Any]] = []
    if _stub_pipeline_at_registry_enabled() and not any_err:
        for e, reg in zip(deduped, results):
            if reg.get("status") == "error":
                continue
            pipeline_stubs.append(
                ensure_mlair_pipeline_stub_if_empty(tenant_id, e["project_id"])
            )

    executor_seeds: list[Dict[str, Any]] = []
    if _seed_demo_executor_enabled() and not any_err:
        for e, reg in zip(deduped, results):
            if reg.get("status") == "error":
                continue
            executor_seeds.append(seed_mlair_demo_executor_track(tenant_id, e["project_id"]))

    n_clinics = sum(
        1
        for row in clinics
        if isinstance(row, dict) and normalize_clinic_key(row.get("id"))
    )
    return {
        "status": "error" if any_err else "ok",
        "tenant_id": tenant_id,
        "catalog_source": source,
        "catalog_clinic_count": n_clinics,
        "registered": results,
        "disk_model_imports": disk_model_imports,
        "pipeline_stubs": pipeline_stubs,
        "executor_seeds": executor_seeds,
    }


def sync_mlair_project_registry_from_catalog() -> Dict[str, Any]:
    """
    Register global + per-clinic MLAir projects from the same clinic catalog used by MLOps UI.
    Intended to run at vet-ai startup (and is idempotent on the MLAir side).
    """
    if not _registry_sync_at_startup_enabled():
        return {"status": "skipped", "reason": "MLAIR_REGISTRY_SYNC_AT_STARTUP disabled"}
    return _sync_mlair_project_registry_core()


def sync_mlair_project_registry_periodic() -> Dict[str, Any]:
    """Same as core registry sync; ignores MLAIR_REGISTRY_SYNC_AT_STARTUP (for background resync)."""
    return _sync_mlair_project_registry_core()


def run_mlair_registry_bootstrap_with_retries() -> Dict[str, Any]:
    """
    Retry full registry sync while MLAir is still starting or the clinic catalog is temporarily empty
    (e.g. customers-service not ready yet).
    """
    import time

    if not mlair_enabled() or not _registry_sync_at_startup_enabled():
        return sync_mlair_project_registry_from_catalog()

    try:
        max_wait = float(os.getenv("MLAIR_REGISTRY_BOOTSTRAP_MAX_WAIT_SECONDS", "120"))
    except ValueError:
        max_wait = 120.0
    try:
        interval = float(os.getenv("MLAIR_REGISTRY_BOOTSTRAP_INTERVAL_SECONDS", "5"))
    except ValueError:
        interval = 5.0
    max_wait = max(5.0, min(max_wait, 600.0))
    interval = max(1.0, min(interval, 60.0))

    deadline = time.monotonic() + max_wait
    last: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _sync_mlair_project_registry_core()

        if last.get("status") == "skipped":
            return last

        reg = last.get("registered") or []
        if reg and all(
            r.get("status") == "skipped" and r.get("reason") == "mlair_registry_not_supported" for r in reg
        ):
            return last

        transient = False
        if last.get("status") == "error":
            transient = any(_mlair_register_result_is_transient(r) for r in reg)

        catalog_warming = False
        if last.get("status") == "ok":
            src = str(last.get("catalog_source") or "")
            ncl = int(last.get("catalog_clinic_count") or 0)
            if src in ("error", "stale") and ncl == 0:
                catalog_warming = True

        if not transient and not catalog_warming:
            return last

        time.sleep(interval)

    logger.warning("MLAir registry bootstrap stopped after %.0fs (last status=%s)", max_wait, last.get("status"))
    return last


def mlair_whoami() -> Dict[str, Any]:
    """GET /v1/auth/whoami — for health checks and token scope debugging."""
    base = mlair_api_base_url()
    if not base:
        return {"status": "disabled", "reason": "MLAIR_API_BASE_URL is not set"}
    try:
        with httpx.Client(timeout=_timeout()) as client:
            r = client.get(f"{base}/v1/auth/whoami", headers=_headers())
            r.raise_for_status()
            data = r.json()
            return {"status": "ok", "whoami": data}
    except Exception as exc:
        logger.warning("MLAir whoami failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def list_mlair_models_for_mlops_ui(clinic_key: Optional[str] = None) -> Dict[str, Any]:
    """
    List logical models from MLAir for the MLOps UI: global project only when no clinic; when a clinic
    is selected, merge that clinic's project with the global project (global rows get displayLabel ``… - global``).
    """
    base = mlair_api_base_url()
    if not base:
        return {"status": "skipped", "reason": "MLAIR_API_BASE_URL unset"}
    tenant_id = mlair_tenant_id()
    global_pid = mlair_project_for_clinic(None)
    ck = normalize_clinic_key(clinic_key)

    def _fetch(client: httpx.Client, project_id: str) -> tuple[list, Optional[str]]:
        url = f"{base}/v1/tenants/{tenant_id}/projects/{quote(project_id, safe='')}/models"
        r = client.get(url, headers=_headers(), params={"limit": 200, "offset": 0})
        if r.status_code == 404:
            return [], None
        if r.status_code == 405:
            return [], "unsupported"
        r.raise_for_status()
        body = r.json()
        items = body.get("items") if isinstance(body, dict) else None
        return (items if isinstance(items, list) else []), None

    try:
        tmo = min(_timeout(), 60.0)
        with httpx.Client(timeout=tmo) as client:
            g_items, g_skip = _fetch(client, global_pid)
            if g_skip == "unsupported":
                return {"status": "skipped", "reason": "mlair_models_endpoint_unsupported", "tenant_id": tenant_id}

            rows: list[Dict[str, Any]] = []
            if ck is None:
                for row in g_items:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("name") or row.get("model_id") or "").strip()
                    merged = dict(row)
                    merged["sourceProject"] = global_pid
                    merged["displayLabel"] = name or str(row.get("model_id") or "")
                    rows.append(merged)
                return {"status": "ok", "tenant_id": tenant_id, "global_project_id": global_pid, "items": rows}

            cpid = mlair_project_for_clinic(ck)
            c_items, c_skip = _fetch(client, cpid)
            if c_skip == "unsupported":
                return {"status": "skipped", "reason": "mlair_models_endpoint_unsupported", "tenant_id": tenant_id}

            for row in c_items:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or row.get("model_id") or "").strip()
                merged = dict(row)
                merged["sourceProject"] = cpid
                merged["displayLabel"] = name or str(row.get("model_id") or "")
                rows.append(merged)

            clinic_ids = {str(r.get("model_id")) for r in c_items if isinstance(r, dict) and r.get("model_id")}
            for row in g_items:
                if not isinstance(row, dict):
                    continue
                mid = str(row.get("model_id") or "").strip()
                if mid and mid in clinic_ids:
                    continue
                name = str(row.get("name") or mid).strip()
                name_for_label = name or mid
                label = name_for_label if name_for_label.endswith(" - global") else f"{name_for_label} - global"
                merged = dict(row)
                merged["sourceProject"] = global_pid
                merged["displayLabel"] = label
                rows.append(merged)

            return {
                "status": "ok",
                "tenant_id": tenant_id,
                "clinic_id": ck,
                "clinic_project_id": cpid,
                "global_project_id": global_pid,
                "items": rows,
            }
    except Exception as exc:
        logger.warning("MLAir list models for MLOps failed: %s", exc)
        return {"status": "error", "error": str(exc), "tenant_id": tenant_id}


def _list_model_items(client: httpx.Client, tenant_id: str, project_id: str) -> list:
    url = f"{mlair_api_base_url()}/v1/tenants/{tenant_id}/projects/{project_id}/models"
    r = client.get(url, headers=_headers(), params={"limit": 200, "offset": 0})
    r.raise_for_status()
    body = r.json()
    items = body.get("items") if isinstance(body, dict) else None
    return items if isinstance(items, list) else []


def _ensure_model_id(client: httpx.Client, tenant_id: str, project_id: str, name: str, description: str) -> str:
    for row in _list_model_items(client, tenant_id, project_id):
        if isinstance(row, dict) and row.get("name") == name and row.get("model_id"):
            return str(row["model_id"])

    url = f"{mlair_api_base_url()}/v1/tenants/{tenant_id}/projects/{project_id}/models"
    r = client.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        json={"name": name, "description": description},
    )
    r.raise_for_status()
    created = r.json()
    mid = created.get("model_id") if isinstance(created, dict) else None
    if not mid:
        raise RuntimeError(f"MLAir create model returned unexpected payload: {created!r}")
    return str(mid)


def _disk_model_import_at_registry_enabled() -> bool:
    return os.getenv("MLAIR_DISK_MODEL_IMPORT_AT_REGISTRY", "1").strip().lower() not in ("0", "false", "no", "off")


def _mlair_model_version_item_count(client: httpx.Client, tenant_id: str, project_id: str, model_id: str) -> int:
    items = _mlair_fetch_model_version_items(client, tenant_id, project_id, model_id)
    if items is None:
        return -1
    return len(items)


def _mlair_fetch_model_version_items(
    client: httpx.Client, tenant_id: str, project_id: str, model_id: str
) -> Optional[list]:
    """GET ``.../models/{model_id}/versions`` items, or ``None`` if the endpoint is unsupported."""
    base = mlair_api_base_url().rstrip("/")
    safe_pid = quote(project_id, safe="")
    safe_mid = quote(model_id, safe="")
    url = f"{base}/v1/tenants/{tenant_id}/projects/{safe_pid}/models/{safe_mid}/versions"
    r = client.get(url, headers=_headers(), params={"limit": 200, "offset": 0})
    if r.status_code == 404:
        return []
    if r.status_code == 405:
        return None
    r.raise_for_status()
    body = r.json() if r.content else {}
    items = body.get("items") if isinstance(body, dict) else None
    return items if isinstance(items, list) else []


def _mlair_disk_folder_hints_from_version_rows(items: list) -> set[str]:
    """
    Best-effort parse of MLAir version rows so we can skip re-importing folders already registered.

    Matches path segments like ``.../v1/model.pkl`` emitted under ``ML_AIR_DEFAULT_MODEL_ARTIFACT_ROOT``.
    """
    out: set[str] = set()
    rx = re.compile(r"/(v\d[A-Za-z0-9_-]*)/")
    for it in items:
        if not isinstance(it, dict):
            continue
        blob = json.dumps(it, default=str)
        out.update(m.group(1) for m in rx.finditer(blob))
    return out


def _mlair_import_http_looks_duplicate(exc: httpx.HTTPStatusError) -> bool:
    code = exc.response.status_code if exc.response is not None else 0
    if code in (409, 423):
        return True
    body = (exc.response.text if exc.response is not None else "")[:2000].lower()
    return any(
        s in body
        for s in (
            "already exists",
            "duplicate",
            "conflict",
            "unique constraint",
            "idempotency",
        )
    )


def _clinic_fallback_global_disk_import_enabled() -> bool:
    return os.getenv("MLAIR_DISK_IMPORT_CLINIC_FALLBACK_GLOBAL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _disk_model_import_candidates(clinic_key: Optional[str]) -> list[tuple[str, str]]:
    """``(model_version, version_dir)`` for global root (``clinic_key`` None) or clinic storage only."""
    from ai_service.app.infrastructure.storage.model_store import (
        find_primary_model_pkl,
        list_model_versions_clinic_storage_only,
        resolve_model_dir,
    )

    ck = normalize_clinic_key(clinic_key)
    out: list[tuple[str, str]] = []
    for v in list_model_versions_clinic_storage_only(ck):
        d = resolve_model_dir(v, ck)
        if find_primary_model_pkl(d) is not None:
            out.append((v, d))

    if ck and not out and _clinic_fallback_global_disk_import_enabled():
        for v in list_model_versions_clinic_storage_only(None):
            d = resolve_model_dir(v, None)
            if find_primary_model_pkl(d) is not None:
                out.append((v, d))
    return out


def import_mlair_model_bundle_with_client(
    client: httpx.Client,
    *,
    tenant_id: str,
    project_id: str,
    model_id: str,
    model_version: str,
    version_dir: str,
    training_id: Optional[int],
    training_mode: str,
) -> Dict[str, Any]:
    """POST ``.../models/{model_id}/versions/import`` (multipart). Shared by post-train sync and disk bootstrap."""
    from ai_service.app.infrastructure.storage.model_store import find_primary_model_pkl

    base = mlair_api_base_url().rstrip("/")
    model_pkl = find_primary_model_pkl(version_dir)
    if model_pkl is None or not model_pkl.is_file():
        return {"status": "error", "error": f"missing model artifact under {version_dir} (model.pkl / .model.pkl)"}
    meta_path = Path(version_dir) / "metadata.json"
    stage = (os.getenv("MLAIR_IMPORT_STAGE", "staging").strip() or "staging").lower()
    # MLAir persists ``model_versions.run_id`` as FK to ``runs``. Synthetic strings like
    # ``vet-ai-disk-v1`` are not runs rows → Postgres raises model_versions_run_id_fkey (HTTP 500).
    # OpenAPI allows ``run_id`` null; omit for disk/bootstrap imports. Training imports may pass a
    # real MLAir ``run_id`` when the caller has created a run first; otherwise omit too.
    run_id_form: Optional[str] = None
    if training_id is not None:
        run_id_form = f"vet-ai-training-{training_id}"

    safe_pid = quote(project_id, safe="")
    safe_mid = quote(model_id, safe="")
    url = f"{base}/v1/tenants/{tenant_id}/projects/{safe_pid}/models/{safe_mid}/versions/import"
    files: Dict[str, Any] = {
        "model_file": ("model.pkl", Path(model_pkl).read_bytes(), "application/octet-stream"),
    }
    data: Dict[str, Any] = {"stage": stage}
    if run_id_form and os.getenv("MLAIR_IMPORT_INCLUDE_RUN_ID", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        data["run_id"] = run_id_form
    if meta_path.is_file():
        files["metadata_file"] = ("metadata.json", meta_path.read_bytes(), "application/json")

    idem_raw = f"{tenant_id}|{project_id}|{model_id}|{run_id_form or ''}|{model_version}"
    idem = hashlib.sha256(idem_raw.encode("utf-8", errors="replace")).hexdigest()[:48]
    r = client.post(url, headers={**_headers(), "Idempotency-Key": idem}, files=files, data=data)
    r.raise_for_status()
    return {"status": "ok", "mlair_version": r.json() if r.content else {}}


def bootstrap_mlair_models_from_disk_for_project(
    client: httpx.Client,
    *,
    tenant_id: str,
    project_id: str,
    clinic_key: Optional[str],
) -> Dict[str, Any]:
    """
    Import every on-disk ``model.pkl`` bundle for this scope into the logical MLAir model.

    Previously skipped entirely once *any* version existed, which left later folders (e.g. ``v2``)
    on disk never pushed while MLAir only showed ``v1``. We now attempt each disk folder and treat
    HTTP duplicate / idempotent responses as success. If MLAir's GET ``.../versions`` returns rows,
    we skip folders whose path segment (``v*``) already appears in that payload.
    """
    if not mlair_enabled() or not _disk_model_import_at_registry_enabled():
        return {"status": "skipped", "reason": "MLAIR_DISK_MODEL_IMPORT_AT_REGISTRY off or MLAIR disabled"}
    logical_name = mlair_logical_model_name(clinic_key)
    desc = (f"vet-ai on-disk registry; clinic={clinic_key or 'global'}")[:2000]
    try:
        model_id = _ensure_model_id(client, tenant_id, project_id, logical_name, desc)
        version_items = _mlair_fetch_model_version_items(client, tenant_id, project_id, model_id)
        if version_items is None:
            return {"status": "skipped", "reason": "mlair_model_versions_not_supported", "project_id": project_id}
        api_hints = {h.lower() for h in _mlair_disk_folder_hints_from_version_rows(version_items)}
        n = len(version_items)
        candidates = _disk_model_import_candidates(clinic_key)
        if not candidates:
            return {"status": "skipped", "reason": "no_disk_bundles", "project_id": project_id, "model_id": model_id}
        imported: list[Dict[str, Any]] = []
        for mv, vdir in candidates:
            if str(mv).strip().lower() in api_hints:
                imported.append({"version": mv, "status": "skipped_already_in_api", "reason": "version_list_hint"})
                continue
            try:
                row = import_mlair_model_bundle_with_client(
                    client,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    model_id=model_id,
                    model_version=mv,
                    version_dir=vdir,
                    training_id=None,
                    training_mode="disk_bootstrap",
                )
                imported.append({"version": mv, **row})
            except httpx.HTTPStatusError as one_exc:
                if _mlair_import_http_looks_duplicate(one_exc):
                    imported.append(
                        {
                            "version": mv,
                            "status": "skipped_duplicate",
                            "http_status": one_exc.response.status_code if one_exc.response else None,
                        }
                    )
                else:
                    logger.warning(
                        "MLAir disk bootstrap import failed tenant=%s project=%s version=%s: %s",
                        tenant_id,
                        project_id,
                        mv,
                        one_exc,
                    )
                    imported.append({"version": mv, "status": "error", "error": str(one_exc)})
            except Exception as one_exc:
                logger.warning(
                    "MLAir disk bootstrap import failed tenant=%s project=%s version=%s: %s",
                    tenant_id,
                    project_id,
                    mv,
                    one_exc,
                )
                imported.append({"version": mv, "status": "error", "error": str(one_exc)})
        ok_n = sum(1 for x in imported if x.get("status") == "ok")
        skip_dup = sum(1 for x in imported if x.get("status") in ("skipped_duplicate", "skipped_already_in_api"))
        err_n = sum(1 for x in imported if x.get("status") == "error")
        if ok_n == 0 and err_n > 0 and skip_dup == 0:
            return {
                "status": "error",
                "project_id": project_id,
                "model_id": model_id,
                "imported": imported,
                "error": "all_disk_imports_failed",
            }
        logger.info(
            "MLAir disk bootstrap tenant=%s project=%s model_id=%s ok=%s skipped=%s errors=%s prior_api_count=%s",
            tenant_id,
            project_id,
            model_id,
            ok_n,
            skip_dup,
            err_n,
            n,
        )
        return {
            "status": "ok",
            "project_id": project_id,
            "model_id": model_id,
            "imported": imported,
            "count": ok_n,
            "skipped": skip_dup,
            "errors": err_n,
        }
    except Exception as exc:
        logger.warning("MLAir disk bootstrap failed project=%s: %s", project_id, exc)
        return {"status": "error", "project_id": project_id, "error": str(exc)}


def _mlair_training_tracking_enabled() -> bool:
    return os.getenv("MLAIR_TRAINING_TRACKING", "0").strip().lower() in ("1", "true", "yes", "on")


def mlair_tracking_pipeline_id() -> str:
    return (os.getenv("MLAIR_TRACKING_PIPELINE_ID", "vetai_local_training") or "vetai_local_training").strip()


def mlair_tracking_plugin_name() -> str:
    return (os.getenv("MLAIR_TRACKING_PLUGIN_NAME", "echo_tracking") or "echo_tracking").strip()


def _mlair_tracking_poll_seconds() -> float:
    try:
        return float(os.getenv("MLAIR_TRACKING_POLL_SECONDS", "90"))
    except ValueError:
        return 90.0


def _mlair_tracking_experiment_name() -> str:
    return (os.getenv("MLAIR_TRACKING_EXPERIMENT_NAME", "Vet-AI continuous training") or "Vet-AI continuous training").strip()


def _mlair_norm_training_mode(mode: str) -> str:
    m = str(mode or "full").strip().lower()
    if m in {"quick", "standard", "full"}:
        return m
    return "full"


def _mlair_training_idempotency_key(
    *,
    training_id: Optional[int],
    model_version: str,
) -> str:
    tid = str(int(training_id)) if training_id is not None else "na"
    h = hashlib.sha256(str(model_version).encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"vetai-train-{tid}-{h}"


def ensure_mlair_tracking_pipeline_version(client: httpx.Client, tenant_id: str, project_id: str) -> Dict[str, Any]:
    """
    Ensure a pipeline version exists with an empty task list so MLAir accepts POST /runs with
    use_latest_pipeline_version; the scheduler still materializes a synthetic task:1 using plugin_name on the run.
    """
    base = mlair_api_base_url()
    pipe = mlair_tracking_pipeline_id()
    safe_proj = quote(project_id, safe="")
    safe_pipe = quote(pipe, safe="")
    versions_url = f"{base}/v1/tenants/{tenant_id}/projects/{safe_proj}/pipelines/{safe_pipe}/versions"
    try:
        lv = client.get(f"{versions_url}?limit=1&offset=0", headers=_headers())
        if lv.status_code in (404, 405):
            return {"status": "skipped", "reason": "mlair_pipeline_versions_not_supported"}
        lv.raise_for_status()
        vbody = lv.json() if lv.content else {}
        vitems = vbody.get("items") if isinstance(vbody, dict) else None
        if isinstance(vitems, list) and len(vitems) > 0:
            return {"status": "ok", "reason": "existing_version", "pipeline_id": pipe}
        pr = client.post(
            versions_url,
            headers={**_headers(), "Content-Type": "application/json"},
            json={"config": {"tasks": []}},
        )
        if pr.status_code in (404, 405):
            return {"status": "skipped", "reason": "mlair_pipeline_version_post_not_supported"}
        pr.raise_for_status()
        return {"status": "ok", "reason": "created", "pipeline_id": pipe, "version": pr.json() if pr.content else {}}
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1500] if exc.response is not None else ""
        logger.warning("MLAir tracking pipeline HTTP error: %s body=%s", exc, detail)
        return {"status": "error", "error": str(exc), "http_detail": detail}
    except Exception as exc:
        logger.warning("MLAir tracking pipeline ensure failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def _resolve_mlair_tracking_experiment_id(client: httpx.Client, tenant_id: str, project_id: str) -> Optional[str]:
    fixed = (os.getenv("MLAIR_TRACKING_EXPERIMENT_ID") or "").strip()
    if fixed:
        return fixed
    safe_proj = quote(project_id, safe="")
    list_url = f"{mlair_api_base_url()}/v1/tenants/{tenant_id}/projects/{safe_proj}/experiments"
    name = _mlair_tracking_experiment_name()
    try:
        lr = client.get(f"{list_url}?limit=200&offset=0", headers=_headers())
        if lr.status_code in (404, 405):
            return None
        lr.raise_for_status()
        body = lr.json() if lr.content else {}
        items = body.get("items") if isinstance(body, dict) else None
        if isinstance(items, list):
            for row in items:
                if isinstance(row, dict) and row.get("name") == name and row.get("experiment_id"):
                    return str(row["experiment_id"])
        cr = client.post(
            list_url,
            headers={**_headers(), "Content-Type": "application/json"},
            json={"name": name, "description": "vet-ai continuous training exports"},
        )
        cr.raise_for_status()
        created = cr.json() if cr.content else {}
        eid = created.get("experiment_id") if isinstance(created, dict) else None
        return str(eid) if eid else None
    except Exception as exc:
        logger.warning("MLAir experiment resolve/create failed: %s", exc)
        return None


def _mlair_poll_run_until_terminal(
    client: httpx.Client,
    tenant_id: str,
    project_id: str,
    run_id: str,
    deadline: float,
) -> Dict[str, Any]:
    safe_proj = quote(project_id, safe="")
    url = f"{mlair_api_base_url()}/v1/tenants/{tenant_id}/projects/{safe_proj}/runs/{run_id}"
    terminal = {"SUCCESS", "FAILED", "CANCELED"}
    last_status = ""
    while time.monotonic() < deadline:
        try:
            gr = client.get(url, headers=_headers())
            gr.raise_for_status()
            row = gr.json() if gr.content else {}
            last_status = str(row.get("status") or "")
            if last_status.upper() in terminal:
                return {"status": "ok", "run": row}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "last_status": last_status}
        time.sleep(0.4)
    return {"status": "timeout", "last_status": last_status}


def _mlair_post_param(client: httpx.Client, tenant_id: str, project_id: str, run_id: str, key: str, value: str) -> None:
    safe_proj = quote(project_id, safe="")
    url = f"{mlair_api_base_url()}/v1/tenants/{tenant_id}/projects/{safe_proj}/runs/{run_id}/params"
    k = str(key).strip()[:200] or "param"
    v = str(value) if value is not None else ""
    if len(v) > 8000:
        v = v[:8000] + "…"
    r = client.post(url, headers={**_headers(), "Content-Type": "application/json"}, json={"key": k, "value": v})
    r.raise_for_status()


def _mlair_post_metric(client: httpx.Client, tenant_id: str, project_id: str, run_id: str, key: str, value: float, step: int = 0) -> None:
    safe_proj = quote(project_id, safe="")
    url = f"{mlair_api_base_url()}/v1/tenants/{tenant_id}/projects/{safe_proj}/runs/{run_id}/metrics"
    k = str(key).strip()[:200] or "metric"
    r = client.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        json={"key": k, "value": float(value), "step": int(step)},
    )
    r.raise_for_status()


def _mlair_post_artifact(client: httpx.Client, tenant_id: str, project_id: str, run_id: str, path: str, uri: Optional[str]) -> None:
    safe_proj = quote(project_id, safe="")
    url = f"{mlair_api_base_url()}/v1/tenants/{tenant_id}/projects/{safe_proj}/runs/{run_id}/artifacts"
    body: Dict[str, Any] = {"path": path}
    if uri:
        body["uri"] = uri
    r = client.post(url, headers={**_headers(), "Content-Type": "application/json"}, json=body)
    r.raise_for_status()


def _mlair_training_tracking_file_uri_for_vetai_path(local_file: Path) -> Optional[str]:
    """
    MLAir stores artifact URIs for the **API server** filesystem. Posting ``file:///app/...`` paths
    from the vet-ai container makes the control plane resolve under ``ML_AIR_DEFAULT_MODEL_ARTIFACT_ROOT``
    on mlair-api, where those paths do not exist — the UI then shows empty version folders.
    """
    flag = (os.getenv("MLAIR_TRACKING_FILE_ARTIFACT_URI") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return f"file://{local_file.resolve()}"
    return None


def push_mlair_training_tracking(
    *,
    model_version: str,
    version_dir: str,
    clinic_key: Optional[str],
    training_id: Optional[int],
    training_mode: str,
    training_metrics: Dict[str, Any],
    pipeline_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """
    After local training, create (or reuse) an MLAir run and mirror key params/metrics into MLAir tracking APIs
    so the control plane UI shows real experiment data tied to vet-ai training.
    """
    base = mlair_api_base_url()
    if not base:
        return {"status": "skipped", "reason": "MLAIR_API_BASE_URL unset"}
    if not _mlair_training_tracking_enabled():
        return {"status": "skipped", "reason": "MLAIR_TRAINING_TRACKING off"}

    tenant_id = mlair_tenant_id()
    project_id = mlair_project_for_clinic(clinic_key)
    pipe = mlair_tracking_pipeline_id()
    plugin = mlair_tracking_plugin_name()
    idem = _mlair_training_idempotency_key(training_id=training_id, model_version=model_version)

    out: Dict[str, Any] = {
        "status": "pending",
        "tenant_id": tenant_id,
        "project_id": project_id,
        "pipeline_id": pipe,
        "idempotency_key": idem,
    }

    try:
        tmo = min(_timeout(), 120.0)
        with httpx.Client(timeout=tmo) as client:
            pv = ensure_mlair_tracking_pipeline_version(client, tenant_id, project_id)
            out["pipeline_ensure"] = pv
            if pv.get("status") == "error":
                out["status"] = "error"
                out["error"] = pv.get("error")
                return out
            if pv.get("status") == "skipped":
                out["status"] = "skipped"
                out["reason"] = pv.get("reason")
                return out

            experiment_id = _resolve_mlair_tracking_experiment_id(client, tenant_id, project_id)
            out["experiment_id"] = experiment_id

            safe_proj = quote(project_id, safe="")
            runs_url = f"{base}/v1/tenants/{tenant_id}/projects/{safe_proj}/runs"
            ck = normalize_clinic_key(clinic_key)
            ctx: Dict[str, Any] = {
                "vetai_source": "continuous_training",
                "vetai_model_version": model_version,
                "vetai_clinic_id": ck if ck is not None else "global",
                "vetai_training_id": training_id,
                "vetai_pipeline_kind": pipeline_kind or training_metrics.get("pipeline_kind"),
            }
            trigger: Dict[str, Any] = {
                "pipeline_id": pipe,
                "plugin_name": plugin,
                "context": ctx,
                "idempotency_key": idem,
                "use_latest_pipeline_version": True,
                "training_mode": _mlair_norm_training_mode(training_mode),
                "override_config": {},
            }
            if experiment_id:
                trigger["experiment_id"] = experiment_id

            rr = client.post(runs_url, headers={**_headers(), "Content-Type": "application/json"}, json=trigger)
            rr.raise_for_status()
            run_row = rr.json() if rr.content else {}
            run_id = str(run_row.get("run_id") or "") if isinstance(run_row, dict) else ""
            if not run_id:
                out["status"] = "error"
                out["error"] = "MLAir POST /runs returned no run_id"
                return out
            out["run_id"] = run_id
            out["run"] = run_row

            poll_budget = max(5.0, min(_mlair_tracking_poll_seconds(), 180.0))
            poll = _mlair_poll_run_until_terminal(
                client,
                tenant_id,
                project_id,
                run_id,
                time.monotonic() + poll_budget,
            )
            out["poll"] = poll

            # Params (string values per MLAir API)
            try:
                _mlair_post_param(client, tenant_id, project_id, run_id, "vetai_model_version", model_version)
                _mlair_post_param(client, tenant_id, project_id, run_id, "vetai_training_mode", str(training_mode))
                if training_id is not None:
                    _mlair_post_param(client, tenant_id, project_id, run_id, "vetai_training_id", str(int(training_id)))
                if ck is not None:
                    _mlair_post_param(client, tenant_id, project_id, run_id, "vetai_clinic_id", ck)
                pk = pipeline_kind or training_metrics.get("pipeline_kind")
                if pk is not None:
                    _mlair_post_param(client, tenant_id, project_id, run_id, "vetai_pipeline_kind", str(pk))
                mp = training_metrics.get("model_params")
                if isinstance(mp, dict) and mp:
                    _mlair_post_param(
                        client,
                        tenant_id,
                        project_id,
                        run_id,
                        "vetai_model_params_json",
                        json.dumps(mp, default=str, ensure_ascii=False)[:8000],
                    )
                for k, v in training_metrics.items():
                    if k in ("model_params",) or k.startswith("mlair_"):
                        continue
                    if isinstance(v, (dict, list)):
                        try:
                            _mlair_post_param(
                                client,
                                tenant_id,
                                project_id,
                                run_id,
                                f"vetai_{k}",
                                json.dumps(v, default=str, ensure_ascii=False)[:8000],
                            )
                        except Exception:
                            continue
                    elif v is not None and not isinstance(v, (int, float, bool)):
                        try:
                            _mlair_post_param(client, tenant_id, project_id, run_id, f"vetai_{k}", str(v)[:8000])
                        except Exception:
                            continue
            except Exception as exc:
                out["params_error"] = str(exc)

            _skip_metric_keys = {"dataset_window_days", "model_params"}
            try:
                for k, v in training_metrics.items():
                    if k in _skip_metric_keys or k.startswith("mlair_"):
                        continue
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        _mlair_post_metric(client, tenant_id, project_id, run_id, str(k), float(v), 0)
            except Exception as exc:
                out["metrics_error"] = str(exc)

            try:
                from ai_service.app.infrastructure.storage.model_store import find_primary_model_pkl

                mpkl = find_primary_model_pkl(version_dir)
                if mpkl is not None and mpkl.is_file():
                    _mlair_post_artifact(
                        client,
                        tenant_id,
                        project_id,
                        run_id,
                        "vet-ai/model.pkl",
                        _mlair_training_tracking_file_uri_for_vetai_path(mpkl),
                    )
                mj = Path(version_dir) / "metrics.json"
                if mj.is_file():
                    _mlair_post_artifact(
                        client,
                        tenant_id,
                        project_id,
                        run_id,
                        "vet-ai/metrics.json",
                        _mlair_training_tracking_file_uri_for_vetai_path(mj),
                    )
            except Exception as exc:
                out["artifacts_error"] = str(exc)

            out["status"] = "ok"
            return out
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:2000] if exc.response is not None else ""
        logger.warning("MLAir training tracking HTTP error: %s body=%s", exc, detail)
        out["status"] = "error"
        out["error"] = str(exc)
        out["http_detail"] = detail
        return out
    except Exception as exc:
        logger.warning("MLAir training tracking failed: %s", exc)
        out["status"] = "error"
        out["error"] = str(exc)
        return out


def sync_training_directory_to_mlair(
    *,
    model_version: str,
    version_dir: str,
    clinic_key: Optional[str],
    training_id: Optional[int],
    training_mode: str,
) -> Dict[str, Any]:
    """
    Register (if needed) a logical model in MLAir and import model.pkl (+ optional metadata.json)
    from the vet-ai version directory as a new registry version.
    """
    base = mlair_api_base_url()
    if not base:
        return {"status": "skipped", "reason": "MLAIR_API_BASE_URL unset"}

    tenant_id = mlair_tenant_id()
    project_id = mlair_project_for_clinic(clinic_key)
    logical_name = mlair_logical_model_name(clinic_key)
    desc = (
        f"vet-ai training export; version={model_version!r}; "
        f"training_id={training_id}; mode={training_mode}; clinic={clinic_key or 'global'}"
    )[:2000]

    from ai_service.app.infrastructure.storage.model_store import find_primary_model_pkl

    if find_primary_model_pkl(version_dir) is None:
        return {"status": "error", "error": f"missing model artifact under {version_dir} (model.pkl / .model.pkl)"}

    out: Dict[str, Any] = {
        "status": "pending",
        "tenant_id": tenant_id,
        "project_id": project_id,
        "logical_model_name": logical_name,
    }

    try:
        out["mlair_registry"] = register_mlair_project(
            tenant_id=tenant_id, project_id=project_id, name=logical_name
        )
        with httpx.Client(timeout=_timeout()) as client:
            model_id = _ensure_model_id(client, tenant_id, project_id, logical_name, desc)
            out["model_id"] = model_id

            imp = import_mlair_model_bundle_with_client(
                client,
                tenant_id=tenant_id,
                project_id=project_id,
                model_id=model_id,
                model_version=model_version,
                version_dir=version_dir,
                training_id=training_id,
                training_mode=training_mode,
            )
            out["mlair_version"] = imp.get("mlair_version")
            out["status"] = "ok"
            logger.info(
                "MLAir registry import ok tenant=%s project=%s model_id=%s response=%s",
                tenant_id,
                project_id,
                model_id,
                imp.get("mlair_version"),
            )
            return out
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:2000] if exc.response is not None else ""
        logger.warning("MLAir HTTP error: %s body=%s", exc, detail)
        out["status"] = "error"
        out["error"] = str(exc)
        out["http_detail"] = detail
        return out
    except Exception as exc:
        logger.warning("MLAir sync failed: %s", exc)
        out["status"] = "error"
        out["error"] = str(exc)
        return out
