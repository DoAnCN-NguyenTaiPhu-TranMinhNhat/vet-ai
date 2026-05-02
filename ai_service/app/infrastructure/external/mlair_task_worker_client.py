"""HTTP client for MLAir external worker APIs (/v1/tasks/lease, complete, fail, heartbeat)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _cfg() -> dict[str, Any]:
    return {
        "base_url": os.getenv("MLAIR_API_BASE_URL", "http://localhost:8080").rstrip("/"),
        "token": os.getenv("MLAIR_AUTH_TOKEN", "").strip(),
        "timeout": float(os.getenv("MLAIR_WORKER_HTTP_TIMEOUT_SECONDS", "60")),
    }


def _headers(token: str) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _request_json(method: str, url: str, *, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    cfg = _cfg()
    if not cfg["token"]:
        raise RuntimeError("MLAIR_AUTH_TOKEN is required for MLAir worker lease/complete")
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method=method, headers=_headers(cfg["token"]))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"MLAir worker HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach MLAir worker API: {exc.reason}") from exc


def lease_tasks(*, worker_id: str, capabilities: list[str], max_tasks: int = 1) -> list[dict[str, Any]]:
    cfg = _cfg()
    url = f"{cfg['base_url']}/v1/tasks/lease"
    body = {"worker_id": worker_id, "capabilities": capabilities, "max_tasks": max_tasks}
    data = _request_json("POST", url, body=body, timeout=cfg["timeout"])
    if str(data.get("execution_mode", "")).lower() != "external":
        return []
    tasks = data.get("tasks")
    return tasks if isinstance(tasks, list) else []


def heartbeat_task(*, task_id: str, worker_id: str) -> bool:
    cfg = _cfg()
    from urllib.parse import quote

    safe = quote(task_id, safe=":")
    url = f"{cfg['base_url']}/v1/tasks/{safe}/heartbeat"
    data = _request_json("POST", url, body={"worker_id": worker_id}, timeout=min(cfg["timeout"], 30.0))
    return bool(data.get("ok"))


def complete_task(
    *,
    task_id: str,
    worker_id: str,
    metrics: dict[str, Any],
    artifact_uri: str | None = None,
) -> dict[str, Any]:
    cfg = _cfg()
    from urllib.parse import quote

    safe = quote(task_id, safe=":")
    url = f"{cfg['base_url']}/v1/tasks/{safe}/complete"
    body: dict[str, Any] = {"worker_id": worker_id, "metrics": metrics}
    if artifact_uri:
        body["artifact_uri"] = artifact_uri
    return _request_json("POST", url, body=body, timeout=cfg["timeout"])


def fail_task(*, task_id: str, worker_id: str, error: str) -> dict[str, Any]:
    cfg = _cfg()
    from urllib.parse import quote

    safe = quote(task_id, safe=":")
    url = f"{cfg['base_url']}/v1/tasks/{safe}/fail"
    return _request_json(
        "POST",
        url,
        body={"worker_id": worker_id, "error": error[:8000]},
        timeout=min(cfg["timeout"], 60.0),
    )


def worker_config_ok() -> bool:
    c = _cfg()
    return bool(c["base_url"] and c["token"])
