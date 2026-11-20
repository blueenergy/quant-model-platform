"""Shadow prediction batch: real parquet + promoted-job recipe."""

from __future__ import annotations

import logging
import uuid
from quant_model_platform.timeutil import utcnow
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
from quant_model_platform.training.walk_forward import (
    DEFAULT_TRAIN_END_OFFSET_DAYS,
    signal_rows,
    walk_forward_fit,
)

log = logging.getLogger(__name__)

#: Training modes that fit on the past of the signal date only. Every other
#: value keeps the historical full-sample behaviour, which is only correct for
#: a model whose label window cannot reach into the live period.
WALK_FORWARD_TRAIN_MODES = {"walk_forward", "walk_forward_expanding"}


def _latest_trade_date(panel: pd.DataFrame) -> str:
    return str(sorted(panel["trade_date"].unique())[-1])


def assert_model_features(day: pd.DataFrame, model: Any, model_id: str) -> None:
    """Refuse to score a model whose declared features the panel cannot supply.

    A panel built without ``stock_scores`` no longer carries the four rule-score
    columns. An old artifact that still declares them would otherwise be scored
    on all-NaN inputs and write silent garbage; fail loudly instead so the
    profile is skipped/retrained, not trusted.
    """
    missing = [
        column
        for column in model.feature_names
        if column not in day.columns or not day[column].notna().any()
    ]
    if missing:
        raise ValueError(
            f"shadow model_id={model_id} requires features not populated in the "
            f"panel: {missing}; retrain the model or disable this profile"
        )


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
    train_mode = str(
        merged.get("train_mode") or merged.get("deploy_mode") or "full_sample"
    ).lower()
    as_of = _latest_trade_date(panel)
    fit_meta: Dict[str, Any] = {}

    if train_mode in WALK_FORWARD_TRAIN_MODES:
        # Point-in-time deployment: the artifact is a function of the past of
        # the signal date, not of the whole snapshot.
        if use_artifact:
            log.warning(
                "shadow model_id=%s asked for a saved artifact but train_mode=%s "
                "refits point-in-time instead; the artifact is ignored",
                model_id,
                train_mode,
            )
        signal_date = str(merged.get("signal_date") or as_of)
        offset = int(merged.get("train_end_offset_days") or DEFAULT_TRAIN_END_OFFSET_DAYS)
        fit = walk_forward_fit(
            panel,
            merged,
            signal_date=signal_date,
            train_end_offset_days=offset,
            label_col=label_col,
            drop_incomplete_features=bool(
                merged.get("drop_incomplete_features", True)
            ),
        )
        model = fit.model
        as_of = fit.signal_date
        fit_meta = fit.as_dict()
        log.info(
            "shadow refit walk-forward model kind=%s train_end=%s rows=%s dates=%s",
            model.kind,
            fit.train_end,
            fit.n_train_rows,
            fit.n_train_dates,
        )
        day = signal_rows(
            panel,
            signal_date=as_of,
            feature_names=model.feature_names,
            drop_incomplete_features=bool(merged.get("drop_incomplete_features", True)),
        )
    else:
        if use_artifact:
            model = load_trained_model(str(merged["artifact_uri"]), cache_root=cache_root)
            log.info("shadow loaded artifact %s", merged["artifact_uri"])
        else:
            model = fit_full_sample_model(panel, merged, label_col=label_col)
            log.info("shadow refit full-sample model kind=%s rows=%s", model.kind, len(panel))
        day = panel.loc[panel["trade_date"] == as_of].copy()
        if day.empty:
            raise ValueError(f"no rows for trade_date={as_of}")

    assert_model_features(day, model, model_id)
    day["score"] = predict_model(day, model)

    preds = cols[COLLECTION_PREDICTIONS]
    preds.delete_many({"model_id": model_id, "trade_date": as_of})

    now = utcnow()
    records: List[Dict[str, Any]] = []
    for _, row in day.iterrows():
        record = {
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
        record.update(fit_meta)
        records.append(record)
    if records:
        preds.insert_many(records)

    summary = {
        "model_id": model_id,
        "index_code": index_code,
        "trade_date": as_of,
        "snapshot_id": merged.get("snapshot_id"),
        "source_job_id": merged.get("source_job_id"),
        "parquet_path": canonical_path,
        "n_predictions": len(records),
        "refit": not use_artifact,
        "train_mode": train_mode,
    }
    summary.update(fit_meta)
    return summary
