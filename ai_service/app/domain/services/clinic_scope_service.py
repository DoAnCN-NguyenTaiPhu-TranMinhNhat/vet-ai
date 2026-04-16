from __future__ import annotations

import re
from typing import Any, Optional

_SLUG_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def normalize_clinic_key(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text else None


def clinic_dir_slug(key: str) -> str:
    if not key:
        return "global"
    slug = _SLUG_SAFE.sub("_", key.strip())
    return (slug[:200] if len(slug) > 200 else slug) or "clinic"
