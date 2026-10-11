"""Continuous model training worker loop."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import COLLECTION_JOBS, COLLECTION_WORKER_STATUS, ensure_ml_indexes, get_ml_collections
from quant_model_platform.queue import JobQueue, WorkerLockHeld, install_shutdown_handlers, run_worker_forever, worker_lock
from quant_model_platform.workers.process_jobs import process_model_training_job

log = logging.getLogger("model_training_loop")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process quant_ml model_training_jobs")
    p.add_argument("--worker-id", default=os.getenv("MODEL_TRAINING_WORKER_ID") or os.getenv("HOSTNAME") or "quant-model-trainer")
    p.add_argument("--interval-seconds", type=float, default=float(os.getenv("MODEL_TRAINING_LOOP_INTERVAL_SECONDS", "5")))
    p.add_argument("--limit-per-iteration", type=int, default=int(os.getenv("MODEL_TRAINING_LOOP_LIMIT", "1")))
    p.add_argument("--stale-running-minutes", type=int, default=int(os.getenv("MODEL_TRAINING_STALE_MINUTES", "120")))
    p.add_argument("--heartbeat-seconds", type=float, default=float(os.getenv("MODEL_TRAINING_HEARTBEAT_SECONDS", "30")))
    p.add_argument("--lock-file", default=os.getenv("MODEL_TRAINING_LOCK_FILE", "/tmp/model_training_worker.lock"))
    p.add_argument("--max-consecutive-failures", type=int, default=int(os.getenv("MODEL_TRAINING_MAX_CONSECUTIVE_FAILURES", "10")))
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    install_shutdown_handlers()
    cfg = PlatformConfig.from_env()
    collections = get_ml_collections(config=cfg)
    ensure_ml_indexes(collections)
    queue = JobQueue(
        jobs=collections[COLLECTION_JOBS],
        worker_type="model_trainer",
        worker_status=collections[COLLECTION_WORKER_STATUS],
    )
    requeued = queue.requeue_own_orphans(args.worker_id)
    if requeued:
        log.info("requeued %d orphan jobs for worker %s", requeued, args.worker_id)

    last_stale_check = 0.0

    def step() -> None:
        nonlocal last_stale_check
        now = time.time()
        if now - last_stale_check >= 60:
            n = queue.requeue_stale(args.stale_running_minutes)
            if n:
                log.info("requeued %d stale jobs", n)
            last_stale_check = now
        queue.record_worker_status(args.worker_id, "idle")
        processed = 0
        while processed < args.limit_per_iteration:
            job = queue.claim(args.worker_id, worker_pool=cfg.worker_pool or None)
            if not job:
                break
            queue.record_worker_status(args.worker_id, "busy", {"job_id": job["job_id"]})
            result = process_model_training_job(job, queue, config=cfg, heartbeat_seconds=args.heartbeat_seconds)
            log.info("processed job %s status=%s", result.get("job_id"), result.get("status"))
            processed += 1
        queue.record_worker_status(args.worker_id, "idle")

    try:
        with worker_lock(args.lock_file, label="model-trainer"):
            return run_worker_forever(
                step,
                interval_seconds=args.interval_seconds,
                max_consecutive_failures=args.max_consecutive_failures,
                logger=log,
            )
    except WorkerLockHeld as exc:
        log.warning("%s", exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
