"""Process one model_training_jobs document."""

from __future__ import annotations

import logging
import uuid
from quant_model_platform.timeutil import utcnow
from pathlib import Path
from typing import Any, Dict, Optional

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import COLLECTION_JOBS, COLLECTION_RESULTS, get_ml_collections
from quant_model_platform.queue import HeartbeatTicker, JobQueue, clean_for_mongo
from quant_model_platform.training.pipeline import run_training_job

log = logging.getLogger(__name__)


def process_model_training_job(
    job: Dict[str, Any],
    queue: JobQueue,
    *,
    config: Optional[PlatformConfig] = None,
    heartbeat_seconds: float = 30.0,
) -> Dict[str, Any]:
    job_id = str(job["job_id"])
    ticker = HeartbeatTicker(queue, job_id, heartbeat_seconds).start()
    try:
        queue.heartbeat(job_id, "training", "running pipeline")
        cfg = config or PlatformConfig.from_env()
        result = run_training_job(
            dict(job.get("params") or {}),
            parquet_root=Path(cfg.parquet_root),
            config=cfg,
        )
        collections = get_ml_collections(config=cfg)
        result_doc = {
            **result,
            "job_id": job_id,
            "created_at": utcnow(),
        }
        collections[COLLECTION_RESULTS].insert_one(clean_for_mongo(result_doc))
        queue.complete(job_id, result_id=result["result_id"])
        return {"status": "completed", "job_id": job_id, "result_id": result["result_id"]}
    except Exception as exc:
        log.exception("model training job failed job_id=%s", job_id)
        queue.fail(job_id, exc)
        return {"status": "failed", "job_id": job_id, "error_message": str(exc)}
    finally:
        ticker.stop()


def enqueue_synthetic_job(collections: Dict[str, Any], *, user_id: str = "system", model_kind: str = "ridge") -> str:
    job_id = str(uuid.uuid4())
    doc = {
        "job_id": job_id,
        "user_id": user_id,
        "name": "synthetic-mvp",
        "status": "pending",
        "created_at": utcnow(),
        "params": {"index_code": "csi1000", "model_kind": model_kind, "universe_count": 2},
    }
    collections[COLLECTION_JOBS].insert_one(doc)
    return job_id
