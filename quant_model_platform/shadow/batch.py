"""Shadow prediction batch: real parquet + promoted-job recipe."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import COLLECTION_PREDICTIONS, get_ml_collections
from quant_model_platform.shadow.resolve import merge_job_into_shadow_params
from quant_model_platform.training.artifacts import load_trained_model
from quant_model_platform.training.fit import fit_full_sample_model
from quant_model_platform.training.models import predict_model
from quant_model_platform.training.panel_loader import load_training_panel

log = logging.getLogger(__name__)


def _latest_trade_date(panel: pd.DataFrame) -> str:
    return str(sorted(panel["trade_date"].unique())[-1])


def run_shadow_batch(params: Dict[str, Any], config: PlatformConfig, *, collections: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cols = collections or get_ml_collections(config=config)
    merged = merge_job_into_shadow_params(dict(params), cols)
    mode = str(merged.get("mode", "production")).lower()
    if mode == "synthetic":
        from quant_model_platform.shadow.synthetic import run_shadow_batch_synthetic

        return run_shadow_batch_synthetic(merged, config)

    parquet_path = merged.get("parquet_path")
    source_job_id = merged.get("source_job_id")
    if not parquet_path and not source_job_id:
        raise ValueError("shadow batch requires parquet_path or source_job_id (or mode=synthetic for dev)")

    index_code = str(merged.get("index_code", "csi1000"))
    model_id = str(merged.get("model_id") or f"shadow-{index_code}")
    label_col = str(merged.get("label_col", "label_fwd_20d"))
    parquet_root = Path(config.parquet_root)

    panel, snapshot_id, canonical_path = load_training_panel(merged, parquet_root=parquet_root, config=config)
    merged.setdefault("snapshot_id", snapshot_id)
    merged.setdefault("parquet_path", canonical_path)

    use_artifact = bool(merged.get("use_saved_artifact")) and merged.get("artifact_uri")
    cache_root = parquet_root / "cache"
    if use_artifact:
        model = load_trained_model(str(merged["artifact_uri"]), cache_root=cache_root)
        log.info("shadow loaded artifact %s", merged["artifact_uri"])
    else:
        model = fit_full_sample_model(panel, merged, label_col=label_col)
        log.info("shadow refit full-sample model kind=%s rows=%s", model.kind, len(panel))

    as_of = _latest_trade_date(panel)
    day = panel.loc[panel["trade_date"] == as_of].copy()
    if day.empty:
        raise ValueError(f"no rows for trade_date={as_of}")
    day["score"] = predict_model(day, model)

    preds = cols[COLLECTION_PREDICTIONS]
    preds.delete_many({"model_id": model_id, "trade_date": as_of})

    now = datetime.utcnow()
    records: List[Dict[str, Any]] = []
    for _, row in day.iterrows():
        records.append(
            {
                "prediction_id": str(uuid.uuid4()),
                "model_id": model_id,
                "snapshot_id": merged.get("snapshot_id", snapshot_id),
                "source_job_id": merged.get("source_job_id"),
                "index_code": index_code,
                "trade_date": as_of,
                "ts_code": row["ts_code"],
                "score": float(row["score"]),
                "created_at": now,
                "shadow": True,
            }
        )
    if records:
        preds.insert_many(records)

    return {
        "model_id": model_id,
        "index_code": index_code,
        "trade_date": as_of,
        "snapshot_id": merged.get("snapshot_id"),
        "source_job_id": merged.get("source_job_id"),
        "parquet_path": canonical_path,
        "n_predictions": len(records),
        "refit": not use_artifact,
    }
