"""
Canonical clinic identifiers for multi-tenant MLOps.

Customers-service uses UUID strings; legacy demos may use small integers.
All keys are normalized to a non-empty string for DB filters, disk paths, and MLflow names.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_SLUG_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def normalize_clinic_key(raw: Any) -> Optional[str]:
    """Return canonical clinic key, or None for global (all clinics / shared pool)."""
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def clinic_dir_slug(key: str) -> str:
    """Filesystem-safe segment under MODEL_ROOT/clinics/ and state/clinics/."""
    if not key:
        return "global"
    slug = _SLUG_SAFE.sub("_", key.strip())
    return (slug[:200] if len(slug) > 200 else slug) or "clinic"
