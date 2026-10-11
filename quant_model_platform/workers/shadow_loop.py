"""Daily shadow predictions into quant_ml.model_predictions."""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import COLLECTION_PREDICTIONS, COLLECTION_SHADOW_JOBS, ensure_ml_indexes, get_ml_collections
from quant_model_platform.dataset.builder import build_synthetic_dataset
from quant_model_platform.dataset.features import registry_names
from quant_model_platform.queue import JobQueue, WorkerLockHeld, install_shutdown_handlers, run_worker_forever, worker_lock
from quant_model_platform.training.models import predict_model, train_model

log = logging.getLogger("model_shadow_loop")


def _latest_trade_date(panel: pd.DataFrame) -> str:
    return str(sorted(panel["trade_date"].unique())[-1])


def run_shadow_batch(params: Dict[str, Any], config: PlatformConfig) -> Dict[str, Any]:
    from pathlib import Path

    index_code = str(params.get("index_code", "csi1000"))
    model_id = str(params.get("model_id") or f"shadow-{index_code}")
    model_kind = str(params.get("model_kind", "ridge"))
    panel, manifest = build_synthetic_dataset(index_code=index_code, parquet_dir=Path(config.parquet_root) / "shadow")
    label_col = "label_fwd_20d"
    features = registry_names()
    train = panel.dropna(subset=[label_col])
    model = train_model(model_kind, train, features, label_col)
    as_of = _latest_trade_date(panel)
    day = panel.loc[panel["trade_date"] == as_of].copy()
    day["score"] = predict_model(day, model)
    records: List[Dict[str, Any]] = []
    now = datetime.utcnow()
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
    if records:
        collections[COLLECTION_PREDICTIONS].insert_many(records)
    return {"model_id": model_id, "trade_date": as_of, "n_predictions": len(records)}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Shadow prediction loop")
    parser.add_argument("--worker-id", default=os.getenv("MODEL_SHADOW_WORKER_ID", "shadow-1"))
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("MODEL_SHADOW_LOOP_INTERVAL_SECONDS", "300")))
    parser.add_argument("--lock-file", default=os.getenv("MODEL_SHADOW_LOCK_FILE", "/tmp/model_shadow_worker.lock"))
    parser.add_argument("--once", action="store_true", help="Run one shadow batch and exit")
    args = parser.parse_args(argv)
    install_shutdown_handlers()
    cfg = PlatformConfig.from_env()
    collections = get_ml_collections(config=cfg)
    ensure_ml_indexes(collections)
    queue = JobQueue(jobs=collections[COLLECTION_SHADOW_JOBS], worker_type="model_shadow")

    def step() -> None:
        job = queue.claim(args.worker_id)
        if not job:
            # No queued shadow job: run default daily batch idempotently via synthetic params.
            run_shadow_batch({"index_code": "csi1000", "model_kind": "ridge"}, cfg)
            return
        try:
            queue.heartbeat(job["job_id"], "shadow", "predict")
            summary = run_shadow_batch(dict(job.get("params") or {}), cfg)
            queue.complete(job["job_id"], **summary)
        except Exception as exc:
            queue.fail(job["job_id"], exc)
            raise

    if args.once:
        summary = run_shadow_batch({"index_code": "csi1000", "model_kind": "ridge"}, cfg)
        log.info("shadow once complete: %s", summary)
        return 0

    try:
        with worker_lock(args.lock_file, label="model-shadow"):
            return run_worker_forever(step, interval_seconds=args.interval_seconds, max_consecutive_failures=5, logger=log)
    except WorkerLockHeld as exc:
        log.warning("%s", exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
