"""
Resolve clinic list for MLOps UI: prefer customers-service HTTP + TTL cache,
then env JSON, then defaults.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Optional

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

from ai_service.app.infrastructure.external.customers_client import fetch_customers_service_clinics

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache_clinics: list[dict[str, Any]] | None = None
_cache_until: float = 0.0
_last_good: list[dict[str, Any]] | None = None


def bust_clinic_catalog_cache() -> None:
    """Invalidate the in-process clinic list cache (e.g. after MLAir becomes reachable or before a registry sync)."""
    global _cache_clinics, _cache_until
    with _lock:
        _cache_clinics = None
        _cache_until = 0.0


def _merge_clinic_lists(*lists: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for clinics in lists:
        if not clinics:
            continue
        for clinic in clinics:
            if not isinstance(clinic, dict):
                continue
            clinic_id = _coerce_catalog_clinic_id(clinic.get("id"))
            if clinic_id is None:
                continue
            name = clinic.get("name")
            normalized = {"id": clinic_id, "name": str(name) if name else f"Clinic {clinic_id}"}
            # Keep first occurrence (higher-priority source passed earlier).
            merged.setdefault(clinic_id, normalized)
    return sorted(merged.values(), key=lambda x: str(x["id"]))


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


def resolve_clinic_identifier(user_clinic: Optional[str]) -> Optional[str]:
    """
    Map a UI label to the canonical clinic id used on disk (e.g. ``models/clinics/<slug>``).

    - If ``user_clinic`` is already a UUID string, it is returned unchanged.
    - Otherwise we look up ``get_clinics_for_mlops()`` for an exact **id** match (case-insensitive)
      or an exact **name** match (case-insensitive), e.g. name ``demo0`` → that row's ``id`` UUID.

    If nothing matches, the original string is returned (legacy callers may use non-UUID ids).
    """
    if user_clinic is None:
        return None
    s = str(user_clinic).strip()
    if not s:
        return None
    if _UUID_RE.match(s):
        return s
    low = s.lower()
    clinics, _src = get_clinics_for_mlops()
    for c in clinics:
        cid = str(c.get("id") or "").strip()
        if cid.lower() == low:
            return cid
        name = str(c.get("name") or "").strip()
        if name.lower() == low:
            return cid
    return s


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
            env_list = _parse_env_json_clinics()
            merged = _merge_clinic_lists(fresh, env_list)
            with _lock:
                _last_good = merged
                _cache_clinics = merged
                _cache_until = now + ttl if ttl > 0 else 0.0
            if env_list:
                return list(merged), "customers-service+env"
            return list(merged), "customers-service"
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
