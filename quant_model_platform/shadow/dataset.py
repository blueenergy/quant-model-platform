"""Resolve canonical production_current snapshots for shadow batches."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from quant_model_platform.dataset.channels import PRODUCTION_CURRENT_CHANNEL
from quant_model_platform.dataset.production import latest_score_date_for_index
from quant_model_platform.db import COLLECTION_DATASETS, get_ml_client
from quant_model_platform.storage.s3 import canonical_snapshot_id

log = logging.getLogger(__name__)


def _parquet_resolution_mode(params: Dict[str, Any]) -> str:
    explicit = params.get("parquet_resolution")
    if explicit is not None:
        return str(explicit).lower()
    env = (os.getenv("MODEL_SHADOW_PARQUET_RESOLUTION") or "canonical").strip().lower()
    return env


def load_current_manifest(collections: Dict[str, Any], index_code: str) -> Optional[Dict[str, Any]]:
    sid = canonical_snapshot_id(index_code)
    coll = collections[COLLECTION_DATASETS]
    doc = coll.find_one({"snapshot_id": sid}, {"_id": 0})
    if doc:
        return doc
    return coll.find_one(
        {"index_code": index_code, "channel": PRODUCTION_CURRENT_CHANNEL},
        {"_id": 0},
    )


def _max_stale_calendar_days() -> int:
    raw = (os.getenv("MODEL_SHADOW_MAX_STALE_CALENDAR_DAYS") or "3").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def apply_canonical_parquet_resolution(
    params: Dict[str, Any],
    collections: Dict[str, Any],
    *,
    mongo_db: str,
) -> Dict[str, Any]:
    mode = _parquet_resolution_mode(params)
    if mode in ("job", "fixed", "false", "0", "off"):
        return params
    index_code = str(params.get("index_code") or "")
    if not index_code:
        return params
    manifest = load_current_manifest(collections, index_code)
    if not manifest:
        raise RuntimeError(
            f"no canonical dataset_manifests for index={index_code} "
            f"(snapshot_id={canonical_snapshot_id(index_code)}); run materialize --canonical --register-mongo"
        )
    path = manifest.get("parquet_path") or manifest.get("parquet_uri")
    if not path:
        raise RuntimeError(f"canonical manifest for {index_code} missing parquet_path")
    end_date = str(manifest.get("end_date") or "")
    client = get_ml_client()
    try:
        latest_scores = latest_score_date_for_index(client, mongo_db=mongo_db, index_code=index_code)
    finally:
        client.close()
    if latest_scores and end_date:
        try:
            end_dt = datetime.strptime(end_date[:8], "%Y%m%d")
            latest_dt = datetime.strptime(latest_scores[:8], "%Y%m%d")
            stale_days = (latest_dt - end_dt).days
            max_stale = _max_stale_calendar_days()
            if stale_days > max_stale:
                raise RuntimeError(
                    f"canonical snapshot end_date={end_date} is {stale_days} calendar days behind "
                    f"latest stock_scores={latest_scores} (max_stale={max_stale}); materialize current first"
                )
            if stale_days > 0:
                log.warning(
                    "canonical end_date=%s behind latest stock_scores=%s by %s day(s)",
                    end_date,
                    latest_scores,
                    stale_days,
                )
        except ValueError:
            log.warning("could not parse end_date=%s or latest=%s for stale check", end_date, latest_scores)
    p = dict(params)
    p["parquet_path"] = str(path)
    p["snapshot_id"] = manifest.get("snapshot_id") or canonical_snapshot_id(index_code)
    p["end_date"] = end_date
    p["parquet_resolution"] = "canonical"
    log.info(
        "shadow canonical parquet index=%s snapshot_id=%s end_date=%s",
        index_code,
        p["snapshot_id"],
        end_date,
    )
    return p
