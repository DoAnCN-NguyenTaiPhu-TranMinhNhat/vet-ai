from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def s3_bucket() -> Optional[str]:
    b = (os.getenv("VETAI_MODEL_ARTIFACT_S3_BUCKET") or "").strip()
    return b or None


def s3_prefix_base() -> str:
    return (os.getenv("VETAI_MODEL_ARTIFACT_S3_PREFIX") or "vet-ai/model-artifacts").strip().strip("/")


def _s3_client():
    import boto3

    return boto3.client("s3")


def s3_key_prefix(model_version: str, clinic_key: Optional[str] = None) -> str:
    from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key

    ck = normalize_clinic_key(clinic_key)
    base = s3_prefix_base()
    if ck:
        return f"{base}/clinics/{clinic_dir_slug(ck)}/{model_version}"
    return f"{base}/global/{model_version}"


def upload_model_directory(local_dir: str, model_version: str, clinic_key: Optional[str] = None) -> None:
    bucket = s3_bucket()
    if not bucket:
        return
    if not os.path.isdir(local_dir):
        logger.warning("S3 upload skipped: not a directory: %s", local_dir)
        return
    prefix = s3_key_prefix(model_version, clinic_key)
    client = _s3_client()
    n = 0
    for root, _, files in os.walk(local_dir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, local_dir).replace("\\", "/")
            key = f"{prefix}/{rel}"
            try:
                client.upload_file(full, bucket, key)
                n += 1
            except Exception as e:
                logger.error("S3 upload failed for %s: %s", key, e)
                raise
    logger.info("S3 upload complete: %s files to s3://%s/%s/", n, bucket, prefix)


def ensure_model_directory_from_s3(model_version: str, clinic_key: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    from ai_service.app.infrastructure.storage.model_store import find_primary_model_pkl, resolve_model_dir

    local = resolve_model_dir(model_version, clinic_key)
    if os.path.isdir(local):
        mp = find_primary_model_pkl(local)
        tab = os.path.join(local, "tab_preprocess.pkl")
        sym = os.path.join(local, "symptoms_mlb.pkl")
        if mp is not None and mp.is_file() and os.path.isfile(tab) and os.path.isfile(sym):
            return True, None

    bucket = s3_bucket()
    if not bucket:
        if os.path.isdir(local):
            return True, None
        return False, f"Model directory missing locally and S3 not configured: {local}"

    prefix = s3_key_prefix(model_version, clinic_key)
    try:
        client = _s3_client()
        os.makedirs(local, exist_ok=True)
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents") or []:
                k = obj.get("Key")
                if k:
                    keys.append(k)
        if not keys:
            return False, f"No objects at s3://{bucket}/{prefix}/"

        for key in keys:
            suffix = key[len(prefix) + 1 :] if key.startswith(prefix + "/") else key
            if not suffix:
                continue
            dest = os.path.join(local, suffix.replace("/", os.sep))
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            client.download_file(bucket, key, dest)
    except Exception as e:
        # If S3 creds are missing (common in local/dev) we must not crash the caller
        # with an uncaught botocore exception (e.g. during active-model restore).
        return False, f"S3 restore failed ({type(e).__name__}): {e}"

    if find_primary_model_pkl(local) is None:
        return False, f"Downloaded from S3 but no model pickle under {local}"

    logger.info("Restored model %s from s3://%s/%s/ to %s", model_version, bucket, prefix, local)
    return True, None
