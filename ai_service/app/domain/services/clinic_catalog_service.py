"""
Resolve clinic list for MLOps UI: prefer customers-service HTTP + TTL cache,
then env JSON, then defaults.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from ai_service.app.infrastructure.external.customers_client import fetch_customers_service_clinics

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache_clinics: list[dict[str, Any]] | None = None
_cache_until: float = 0.0
_last_good: list[dict[str, Any]] | None = None


def _coerce_catalog_clinic_id(raw: Any) -> str | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return str(raw) if raw >= 1 else None

    text = str(raw).strip()
    if not text or text.lower() in ("null", "none"):
        return None
    return text


def _parse_env_json_clinics() -> list[dict[str, Any]] | None:
    raw = (os.getenv("MLOPS_CLINICS_JSON") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return None
        output: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict) or "id" not in item:
                continue
            clinic_id = _coerce_catalog_clinic_id(item["id"])
            if clinic_id is None:
                continue
            name = item.get("name")
            output.append({"id": clinic_id, "name": str(name) if name else f"Clinic {clinic_id}"})
        return output or None
    except Exception as exc:
        logger.warning("MLOPS_CLINICS_JSON invalid: %s", exc)
        return None


def _default_placeholder_clinics() -> list[dict[str, Any]]:
    return [
        {"id": "78343a5e-047b-5edb-9975-678bf3f815c6", "name": "Demo Veterinary Clinic"},
        {"id": "f4b59806-b23b-598c-bf07-e3f87bb5cd99", "name": "Demo1 Veterinary Clinic"},
    ]


def _normalize_remote_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    clinic_id = _coerce_catalog_clinic_id(item.get("id"))
    if clinic_id is None:
        return None
    name = item.get("name")
    return {"id": clinic_id, "name": str(name) if name else f"Clinic {clinic_id}"}


def _fetch_customers_service_clinics() -> list[dict[str, Any]]:
    data = fetch_customers_service_clinics()
    output: list[dict[str, Any]] = []
    for row in data:
        normalized = _normalize_remote_item(row)
        if normalized:
            output.append(normalized)

    if not output and data:
        logger.warning(
            "customers-service returned %d clinic rows but none had usable id",
            len(data),
        )
    output.sort(key=lambda x: str(x["id"]))
    return output


def get_clinics_for_mlops() -> tuple[list[dict[str, Any]], str]:
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
        except Exception as exc:
            logger.warning("Failed to fetch clinics from customers-service: %s", exc)
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
