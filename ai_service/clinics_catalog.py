"""
Resolve clinic list for MLOps UI: prefer customers-service HTTP + TTL cache, then env JSON, then defaults.

Clinic ids from customers-service are UUID strings; legacy configs may use integer ids (coerced to string).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache_clinics: Optional[List[Dict[str, Any]]] = None
_cache_until: float = 0.0
_last_good: Optional[List[Dict[str, Any]]] = None


def _coerce_catalog_clinic_id(raw: Any) -> Optional[str]:
    """Normalize clinic id to a non-empty string (UUID or legacy positive int)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        if raw < 1:
            return None
        return str(raw)
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


def _parse_env_json_clinics() -> Optional[List[Dict[str, Any]]]:
    raw = (os.getenv("MLOPS_CLINICS_JSON") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return None
        out: List[Dict[str, Any]] = []
        for x in data:
            if not isinstance(x, dict) or "id" not in x:
                continue
            cid = _coerce_catalog_clinic_id(x["id"])
            if cid is None:
                continue
            name = x.get("name")
            out.append({"id": cid, "name": str(name) if name else f"Clinic {cid}"})
        return out or None
    except Exception as e:
        logger.warning("MLOPS_CLINICS_JSON invalid: %s", e)
        return None


def _default_placeholder_clinics() -> List[Dict[str, Any]]:
    # Align with demo seed UUIDs in customers-service hsqldb/data.sql when no upstream is configured.
    return [
        {"id": "78343a5e-047b-5edb-9975-678bf3f815c6", "name": "Demo Veterinary Clinic"},
        {"id": "f4b59806-b23b-598c-bf07-e3f87bb5cd99", "name": "Demo1 Veterinary Clinic"},
    ]


def _normalize_remote_item(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    cid = _coerce_catalog_clinic_id(item.get("id"))
    if cid is None:
        return None
    name = item.get("name")
    return {"id": cid, "name": str(name) if name else f"Clinic {cid}"}


def _fetch_customers_service_clinics() -> List[Dict[str, Any]]:
    base = (os.getenv("CUSTOMERS_SERVICE_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return []
    path = (os.getenv("CUSTOMERS_CLINICS_PATH") or "/clinics").strip()
    if not path.startswith("/"):
        path = "/" + path
    url = f"{base}{path}"
    timeout = float(os.getenv("CUSTOMERS_SERVICE_TIMEOUT_SECONDS", "5"))
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, headers={"Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, list):
        raise ValueError("customers clinics endpoint must return a JSON array")
    out: List[Dict[str, Any]] = []
    for row in data:
        n = _normalize_remote_item(row)
        if n:
            out.append(n)
    if not out and data:
        logger.warning(
            "customers-service returned %d clinic rows but none had a usable id field; "
            "expected objects with id (UUID or int) and optional name",
            len(data),
        )
    out.sort(key=lambda x: str(x["id"]))
    return out


def get_clinics_for_mlops() -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns (clinics, source) where source is one of:
    customers-service | stale | env | default | error
    """
    global _cache_clinics, _cache_until, _last_good

    base = (os.getenv("CUSTOMERS_SERVICE_BASE_URL") or "").strip()
    ttl = max(0, int(os.getenv("CUSTOMERS_CLINICS_CACHE_TTL_SECONDS", "60")))

    if base:
        now = time.monotonic()
        with _lock:
            if ttl > 0 and _cache_clinics is not None and now < _cache_until:
                return list(_cache_clinics), "customers-service"

        try:
            fresh = _fetch_customers_service_clinics()
            with _lock:
                _last_good = fresh
                _cache_clinics = fresh
                _cache_until = now + ttl if ttl > 0 else 0.0
            return list(fresh), "customers-service"
        except Exception as e:
            logger.warning("Failed to fetch clinics from customers-service: %s", e)
            with _lock:
                if _last_good is not None:
                    return list(_last_good), "stale"
            env_list = _parse_env_json_clinics()
            if env_list:
                return env_list, "env"
            return [], "error"

    env_list = _parse_env_json_clinics()
    if env_list:
        return env_list, "env"
    return _default_placeholder_clinics(), "default"
