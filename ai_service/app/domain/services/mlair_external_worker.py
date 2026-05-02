"""
Background worker: lease tasks from MLAir (external execution mode), run Vet-AI work, complete/fail with real metrics.

Requires MLAir `ML_AIR_TASK_EXECUTION_MODE=external` and migration 0011; Vet-AI needs `MLAIR_ENABLED=true`
and `MLAIR_AUTH_TOKEN` matching MLAir (maintainer or `ML_AIR_WORKER_TOKEN`).

No changes to ml-air core — only Vet-AI repo.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _internal_api_base() -> str:
    explicit = os.getenv("VETAI_INTERNAL_API_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    port = int(os.getenv("PORT", "8000"))
    return f"http://127.0.0.1:{port}"


def _admin_token() -> str:
    """No default secret in code — set ADMIN_TOKEN (or VETAI_INTERNAL_API_TOKEN) in the environment."""
    return (os.getenv("ADMIN_TOKEN", "") or os.getenv("VETAI_INTERNAL_API_TOKEN", "")).strip()


def _ct_request_json(method: str, path: str, *, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    base = _internal_api_base()
    url = f"{base}{path}"
    token = _admin_token()
    headers = {"Authorization": f"Bearer {token}"}
    if method.upper() == "GET":
        req = urllib.request.Request(url, headers=headers, method=method)
    else:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Vet-AI internal HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Vet-AI internal API: {exc.reason}") from exc


def _mlair_api_base() -> str:
    return os.getenv("MLAIR_API_BASE_URL", "").strip().rstrip("/")


def _mlair_auth_token() -> str:
    return os.getenv("MLAIR_AUTH_TOKEN", "").strip()


def _download_dataset_version_csv(*, tenant_id: str, project_id: str, dataset_version_id: str) -> tuple[bytes, str]:
    base = _mlair_api_base()
    token = _mlair_auth_token()
    if not base or not token:
        raise RuntimeError("Missing MLAIR_API_BASE_URL or MLAIR_AUTH_TOKEN for dataset-driven training")
    from urllib.parse import quote

    url = (
        f"{base}/v1/tenants/{quote(tenant_id, safe='')}/projects/{quote(project_id, safe='')}"
        f"/dataset-versions/{quote(dataset_version_id, safe='')}/download"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=float(os.getenv("VETAI_MLAIR_DATASET_DOWNLOAD_TIMEOUT_SECONDS", "120"))) as resp:  # noqa: S310
            blob = resp.read()
            cd = str(resp.headers.get("Content-Disposition") or "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"MLAir dataset download HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot download MLAir dataset version: {exc.reason}") from exc
    filename = "dataset.csv"
    if "filename=" in cd:
        filename = cd.split("filename=", 1)[1].strip().strip('"').strip() or filename
    return blob, filename


def _trigger_bootstrap_csv_training(*, csv_bytes: bytes, filename: str, clinic_id: str | None) -> int:
    base = _internal_api_base()
    token = _admin_token()
    boundary = f"----mlair-boundary-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="' + filename.replace('"', "") + '"\r\n'
            "Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(csv_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"training_mode\"\r\n\r\nlocal\r\n".encode("utf-8"))
    if clinic_id:
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"clinic_id\"\r\n\r\n{clinic_id}\r\n"
            ).encode("utf-8")
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    payload = b"".join(parts)
    req = urllib.request.Request(
        f"{base}/continuous-training/training/bootstrap-csv",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(os.getenv("VETAI_MLAIR_CT_TRIGGER_TIMEOUT_SECONDS", "120"))) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            out = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Vet-AI bootstrap CSV HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Vet-AI bootstrap CSV API: {exc.reason}") from exc
    tid = out.get("training_id")
    if tid is None:
        raise RuntimeError(f"unexpected_bootstrap_trigger_response: {out!r}")
    return int(tid)


def _trigger_continuous_training(
    *,
    clinic_id: str | None,
    force: bool,
    finetune_base_model_version: str | None = None,
) -> list[int]:
    """POST /continuous-training/training/trigger; returns one or more training_ids."""
    body: dict[str, Any] = {
        "trigger_type": "manual",
        "trigger_reason": "mlair_external_worker",
        "force": force,
        "training_mode": "local",
    }
    if clinic_id:
        body["clinic_id"] = clinic_id
    if finetune_base_model_version and str(finetune_base_model_version).strip():
        body["finetune_base_model_version"] = str(finetune_base_model_version).strip()
    out = _ct_request_json(
        "POST",
        "/continuous-training/training/trigger",
        body=body,
        timeout=float(os.getenv("VETAI_MLAIR_CT_TRIGGER_TIMEOUT_SECONDS", "120")),
    )
    if out.get("status") == "triggered_multi_clinic" and isinstance(out.get("items"), list):
        ids: list[int] = []
        for it in out["items"]:
            if isinstance(it, dict) and it.get("training_id") is not None:
                ids.append(int(it["training_id"]))
        return ids
    tid = out.get("training_id")
    if tid is not None:
        return [int(tid)]
    raise RuntimeError(f"unexpected_trigger_response: {out!r}")


def _get_training_status(training_id: int) -> dict[str, Any]:
    return _ct_request_json(
        "GET",
        f"/continuous-training/training/status?training_id={training_id}",
        body=None,
        timeout=float(os.getenv("VETAI_MLAIR_CT_STATUS_TIMEOUT_SECONDS", "30")),
    )


def _wait_training_complete(
    training_ids: list[int],
    *,
    heartbeat_fn: Any | None,
    task_id: str,
    worker_id: str,
) -> dict[str, Any]:
    """Poll until all terminal; merge last status dict for metrics."""
    poll = float(os.getenv("VETAI_MLAIR_CT_POLL_SECONDS", "4"))
    max_wait = float(os.getenv("VETAI_MLAIR_CT_MAX_WAIT_SECONDS", "3600"))
    import time

    deadline = time.monotonic() + max_wait
    last_by_id: dict[int, dict[str, Any]] = {}
    pending = set(training_ids)
    last_hb = 0.0
    while pending and time.monotonic() < deadline:
        for tid in list(pending):
            st = _get_training_status(tid)
            last_by_id[tid] = st
            status = str(st.get("status", "")).lower()
            if status in ("completed", "failed"):
                pending.discard(tid)
                if status == "failed":
                    err = st.get("error_message") or "training_failed"
                    raise RuntimeError(f"training_id={tid} failed: {err}")
        if pending:
            now = time.monotonic()
            if heartbeat_fn and (now - last_hb) >= float(os.getenv("VETAI_MLAIR_WORKER_HEARTBEAT_SECONDS", "15")):
                try:
                    heartbeat_fn(task_id=task_id, worker_id=worker_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("mlair_heartbeat_failed task_id=%s err=%s", task_id, exc)
                last_hb = now
            time.sleep(poll)
    if pending:
        raise RuntimeError(f"training_poll_timeout pending={pending}")
    if len(training_ids) == 1:
        return last_by_id[training_ids[0]]
    # Multi-clinic: use the last job row for primary metrics; all jobs completed successfully.
    last_tid = training_ids[-1]
    return last_by_id.get(last_tid) or {}


def _status_to_mlair_metrics(st: dict[str, Any]) -> dict[str, Any]:
    """Shape metrics for MLAir tracking (value + step)."""
    metrics: dict[str, Any] = {}
    if st.get("f1_score") is not None:
        metrics["f1_score"] = {"value": float(st["f1_score"]), "step": 0}
    if st.get("validation_accuracy") is not None:
        metrics["validation_accuracy"] = {"value": float(st["validation_accuracy"]), "step": 0}
    if st.get("training_accuracy") is not None:
        metrics["training_accuracy"] = {"value": float(st["training_accuracy"]), "step": 0}
    if st.get("training_id") is not None:
        metrics["vetai_training_id"] = {"value": float(int(st["training_id"])), "step": 0}
    if st.get("dataset_row_count") is not None:
        metrics["dataset_row_count"] = {"value": float(int(st["dataset_row_count"])), "step": 0}
    if not metrics:
        metrics["vetai_training_note"] = {"value": 1.0, "step": 0}
    return metrics


def _artifact_uri_from_status(st: dict[str, Any]) -> str | None:
    nv = st.get("new_model_version")
    if isinstance(nv, str) and nv.strip():
        return f"vetai://model-version/{nv.strip()}"
    return None


def _plugin_etl_stub(task: dict[str, Any]) -> dict[str, Any]:
    """ETL steps: no separate engine in Vet-AI; acknowledge with lightweight metrics."""
    key = task.get("task_key") or ""
    return {
        f"etl_{key}_ok": {"value": 1.0, "step": 0},
        "vetai_plugin": {"value": 1.0, "step": 0},
    }


def _resolve_train_clinic_id(task: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    """
    Clinic scope for Vet-AI continuous training must match MLAir project convention ``clinic_<slug>``.

    Prefer explicit ``clinic_id`` / ``source_clinic_id`` in plugin_context; if missing, derive from
    ``task.project_id`` so MLAir UI runs (which often omit context.clinic_id) still train per-clinic.
    """
    for key in ("clinic_id", "source_clinic_id"):
        raw = ctx.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.lower() == "global":
            continue
        return s
    pid = str(task.get("project_id") or "").strip()
    if pid.startswith("clinic_"):
        slug = pid[len("clinic_") :].strip()
        if slug:
            logger.info(
                "mlair_train_clinic_derived_from_project project_id=%s clinic_id=%s",
                pid,
                slug,
            )
            return slug
    return None


def _materialize_plugin_context_weights(
    *,
    artifact_uri: str,
    clinic_id: str | None,
    model_id_hint: str,
) -> str:
    """Symlink ``file://`` weights from MLAir ``plugin_context`` into MODEL_ROOT; returns version folder name."""
    from ai_service.app.domain.services.clinic_scope_service import normalize_clinic_key
    from ai_service.app.infrastructure.storage import model_store as ms

    uri = str(artifact_uri or "").strip()
    if not uri.startswith("file://"):
        raise RuntimeError("plugin_context_artifact_uri_must_be_file_scheme")
    ck = normalize_clinic_key(clinic_id) if clinic_id else None
    mode = (os.getenv("VETAI_MLAIR_PROMOTION_MODE") or "materialize").strip().lower()
    if mode == "reuse_only":
        src_path = os.path.realpath(urlparse(uri).path)
        existing = ms.find_version_label_for_artifact_realpath(src_path, ck)
        if existing:
            logger.info(
                "mlair_worker_reuse_existing_weights path=%s version=%s (VETAI_MLAIR_PROMOTION_MODE=reuse_only)",
                src_path,
                existing,
            )
            return existing
        raise RuntimeError(
            "reuse_only: plugin_context artifact path does not match any existing v* folder under MODEL_ROOT; "
            f"path={src_path}"
        )
    mid = re.sub(r"[^a-zA-Z0-9_-]+", "_", (model_id_hint or "model").strip())[:40] or "model"
    h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12]
    version_label = f"v_mlair_{mid}_u{h}"
    return ms.materialize_mlair_artifact_as_version(
        artifact_uri=uri,
        version_label=version_label,
        clinic_key=ck,
    )


def _plugin_train(task: dict[str, Any], heartbeat_fn: Any | None) -> tuple[dict[str, Any], str | None]:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    ctx = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    override_cfg = payload.get("override_config") if isinstance(payload.get("override_config"), dict) else {}
    dataset_version_id = str(override_cfg.get("dataset_version_id") or "").strip()
    clinic_id = _resolve_train_clinic_id(task, ctx)
    force = str(os.getenv("VETAI_MLAIR_TRAIN_FORCE", "true")).lower() in ("1", "true", "yes", "y")
    task_id = str(task.get("task_id", ""))
    worker_id = str(task.get("_worker_id", ""))
    finetune_base: str | None = None
    art = str(ctx.get("artifact_uri") or "").strip()
    if art.startswith("file://"):
        mid_hint = str(ctx.get("mlair_model_id") or ctx.get("model_id") or "model").strip()
        finetune_base = _materialize_plugin_context_weights(
            artifact_uri=art,
            clinic_id=clinic_id,
            model_id_hint=mid_hint,
        )
    if dataset_version_id:
        tenant_id = str(task.get("tenant_id") or "").strip()
        project_id = str(task.get("project_id") or "").strip()
        if not tenant_id or not project_id:
            raise RuntimeError("dataset_train_missing_tenant_or_project")
        csv_bytes, filename = _download_dataset_version_csv(
            tenant_id=tenant_id,
            project_id=project_id,
            dataset_version_id=dataset_version_id,
        )
        training_ids = [_trigger_bootstrap_csv_training(csv_bytes=csv_bytes, filename=filename, clinic_id=clinic_id)]
    else:
        training_ids = _trigger_continuous_training(
            clinic_id=clinic_id,
            force=force,
            finetune_base_model_version=finetune_base,
        )
    st = _wait_training_complete(
        training_ids,
        heartbeat_fn=heartbeat_fn,
        task_id=task_id,
        worker_id=worker_id,
    )
    metrics = _status_to_mlair_metrics(st if isinstance(st, dict) else {})
    artifact = _artifact_uri_from_status(st if isinstance(st, dict) else {})
    mlair_model_id = str(ctx.get("mlair_model_id") or "").strip() or None
    if (
        mlair_model_id
        and artifact
        and str(os.getenv("VETAI_MLAIR_REGISTER_OUTPUT_MODEL_VERSION", "true")).lower() in ("1", "true", "yes")
    ):
        tenant_id = str(task.get("tenant_id") or "").strip()
        project_id = str(task.get("project_id") or "").strip()
        run_id = str(task.get("run_id") or "").strip() or None
        if not tenant_id or not project_id:
            raise RuntimeError("mlair_model_register_missing_tenant_or_project")
        from ai_service.app.infrastructure.external import mlair_client as mc

        reg_stage = str(ctx.get("mlair_new_version_stage") or "").strip() or None
        mc.register_model_version_for_scope(
            tenant_id=tenant_id,
            project_id=project_id,
            model_id=mlair_model_id,
            artifact_uri=artifact,
            run_id=run_id,
            stage=reg_stage,
        )
        logger.info(
            "mlair_model_version_registered model_id=%s run_id=%s artifact=%s",
            mlair_model_id,
            run_id,
            artifact,
        )
    return metrics, artifact


def process_one_task(task: dict[str, Any], worker_id: str) -> None:
    from ai_service.app.infrastructure.external import mlair_task_worker_client as mtw

    task_id = str(task["task_id"])
    plugin = str(task.get("plugin") or "").strip()
    task["_worker_id"] = worker_id

    def hb(**kw: Any) -> bool:
        return mtw.heartbeat_task(task_id=kw["task_id"], worker_id=kw["worker_id"])

    try:
        if plugin in {"app_etl_adapter", "echo_tracking"}:
            metrics = _plugin_etl_stub(task)
            artifact = None
        elif plugin == "app_train_adapter":
            metrics, artifact = _plugin_train(task, hb)
        else:
            raise RuntimeError(f"unsupported_plugin:{plugin}")
        mtw.complete_task(task_id=task_id, worker_id=worker_id, metrics=metrics, artifact_uri=artifact)
        logger.info("mlair_worker_task_completed task_id=%s plugin=%s", task_id, plugin)
    except Exception as exc:  # noqa: BLE001
        logger.exception("mlair_worker_task_failed task_id=%s plugin=%s", task_id, plugin)
        try:
            mtw.fail_task(task_id=task_id, worker_id=worker_id, error=str(exc))
        except Exception as exc2:  # noqa: BLE001
            logger.error("mlair_worker_fail_callback_failed task_id=%s err=%s", task_id, exc2)


def _capabilities_list() -> list[str]:
    raw = os.getenv("VETAI_MLAIR_WORKER_CAPABILITIES", "app_etl_adapter,app_train_adapter").strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


async def run_mlair_worker_loop(stop_event: asyncio.Event) -> None:
    from ai_service.app.infrastructure.external import mlair_task_worker_client as mtw

    if not mtw.worker_config_ok():
        logger.warning("mlair_worker_disabled missing MLAIR_API_BASE_URL or MLAIR_AUTH_TOKEN")
        return

    worker_id = os.getenv("VETAI_MLAIR_WORKER_ID", "vet-ai-mlair-worker-1").strip()
    caps = _capabilities_list()
    poll = float(os.getenv("VETAI_MLAIR_WORKER_POLL_SECONDS", "3"))
    logger.info(
        "mlair_worker_loop_started worker_id=%s capabilities=%s poll_s=%s",
        worker_id,
        caps,
        poll,
    )
    while not stop_event.is_set():
        try:
            tasks = await asyncio.to_thread(mtw.lease_tasks, worker_id=worker_id, capabilities=caps, max_tasks=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mlair_lease_error err=%s", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll)
            except asyncio.TimeoutError:
                pass
            continue
        if not tasks:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll)
            except asyncio.TimeoutError:
                pass
            continue
        for t in tasks:
            if stop_event.is_set():
                break
            await asyncio.to_thread(process_one_task, t, worker_id)
