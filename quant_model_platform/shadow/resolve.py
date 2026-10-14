"""Resolve shadow batch params from source_job_id and optional Mongo profiles."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import COLLECTION_JOBS, COLLECTION_RESULTS, COLLECTION_SHADOW_PROFILES
from quant_model_platform.shadow.dataset import apply_canonical_parquet_resolution


def _short_job_id(job_id: str) -> str:
    return job_id.replace("-", "")[:8]


def merge_job_into_shadow_params(params: Dict[str, Any], collections: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    job_id = p.get("source_job_id")
    if not job_id:
        return p
    job = collections[COLLECTION_JOBS].find_one({"job_id": str(job_id)})
    if not job:
        raise ValueError(f"unknown source_job_id={job_id}")
    jp = dict(job.get("params") or {})
    for key in (
        "index_code",
        "model_kind",
        "parquet_path",
        "snapshot_id",
        "label_col",
        "feature_names",
        "model_params",
        "data_source",
        "start_date",
        "end_date",
    ):
        if key not in p and key in jp:
            p[key] = jp[key]
    result = collections[COLLECTION_RESULTS].find_one({"job_id": str(job_id)}, sort=[("created_at", -1)])
    if result:
        for key in ("parquet_path", "snapshot_id", "artifact_uri", "model_kind", "artifact_hash"):
            if key not in p and result.get(key):
                p[key] = result[key]
    index_code = str(p.get("index_code", "unknown"))
    p.setdefault("model_id", f"promo-{index_code}-{_short_job_id(str(job_id))}")
    p["source_job_id"] = str(job_id)
    cfg = PlatformConfig.from_env()
    return apply_canonical_parquet_resolution(p, collections, mongo_db=cfg.mongo_db)


def load_shadow_profiles_from_env() -> List[Dict[str, Any]]:
    raw = (os.getenv("MODEL_SHADOW_PROFILES") or "").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError("MODEL_SHADOW_PROFILES must be JSON array or object")
    return [dict(x) for x in data]


def load_enabled_shadow_profiles(collections: Dict[str, Any]) -> List[Dict[str, Any]]:
    env_profiles = load_shadow_profiles_from_env()
    if env_profiles:
        return [p for p in env_profiles if p.get("enabled", True)]
    coll = collections.get(COLLECTION_SHADOW_PROFILES)
    if coll is None:
        return []
    docs = list(coll.find({"enabled": {"$ne": False}}, {"_id": 0}))
    return docs


def resolve_profile_params(profile: Dict[str, Any], collections: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(profile.get("params") or profile)
    if profile.get("source_job_id") and "source_job_id" not in base:
        base["source_job_id"] = profile["source_job_id"]
    if profile.get("model_id"):
        base["model_id"] = profile["model_id"]
    return merge_job_into_shadow_params(base, collections)
