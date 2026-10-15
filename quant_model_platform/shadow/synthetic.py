"""Dev-only synthetic shadow batch (not used in production Cron)."""

from __future__ import annotations

import uuid
from quant_model_platform.timeutil import utcnow
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.builder import build_synthetic_dataset
from quant_model_platform.dataset.features import registry_names
from quant_model_platform.db import COLLECTION_PREDICTIONS, get_ml_collections
from quant_model_platform.storage.s3 import publish_local_parquet
from quant_model_platform.training.models import predict_model, train_model


def _latest_trade_date(panel: pd.DataFrame) -> str:
    return str(sorted(panel["trade_date"].unique())[-1])


def run_shadow_batch_synthetic(params: Dict[str, Any], config: PlatformConfig) -> Dict[str, Any]:
    index_code = str(params.get("index_code", "csi1000"))
    model_id = str(params.get("model_id") or f"shadow-{index_code}")
    model_kind = str(params.get("model_kind", "ridge"))
    parquet_dir = Path(config.parquet_root) / "shadow"
    panel, manifest = build_synthetic_dataset(index_code=index_code, parquet_dir=parquet_dir)
    local_snap = parquet_dir / f"{index_code}_synthetic.parquet"
    publish_local_parquet(local_snap, index_code=index_code, snapshot_id=manifest.snapshot_id, config=config)
    label_col = "label_fwd_20d"
    features = registry_names()
    train = panel.dropna(subset=[label_col])
    model = train_model(model_kind, train, features, label_col)
    as_of = _latest_trade_date(panel)
    day = panel.loc[panel["trade_date"] == as_of].copy()
    day["score"] = predict_model(day, model)
    records: List[Dict[str, Any]] = []
    now = utcnow()
    for _, row in day.iterrows():
        records.append(
            {
                "prediction_id": str(uuid.uuid4()),
                "model_id": model_id,
                "snapshot_id": manifest.snapshot_id,
                "trade_date": as_of,
                "ts_code": row["ts_code"],
                "score": float(row["score"]),
                "created_at": now,
                "shadow": True,
            }
        )
    collections = get_ml_collections(config=config)
    preds = collections[COLLECTION_PREDICTIONS]
    preds.delete_many({"model_id": model_id, "trade_date": as_of})
    if records:
        preds.insert_many(records)
    return {"model_id": model_id, "trade_date": as_of, "n_predictions": len(records), "mode": "synthetic"}
