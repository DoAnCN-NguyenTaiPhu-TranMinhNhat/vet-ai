from __future__ import annotations

import os
from typing import Any

import httpx


def fetch_customers_service_clinics() -> list[dict[str, Any]]:
    base = (os.getenv("CUSTOMERS_SERVICE_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return []

    path = (os.getenv("CUSTOMERS_CLINICS_PATH") or "/clinics").strip()
    if not path.startswith("/"):
        path = "/" + path

    url = f"{base}{path}"
    timeout = float(os.getenv("CUSTOMERS_SERVICE_TIMEOUT_SECONDS", "5"))

    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, list):
        raise ValueError("customers clinics endpoint must return a JSON array")

    return payload
