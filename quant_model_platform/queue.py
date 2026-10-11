"""Mongo job queue (subset of stock-scoring-system.utils.job_queue)."""

from __future__ import annotations

import fcntl
import logging
import math
import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import PyMongoError

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.utcnow()


def clean_for_mongo(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_mongo(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_mongo(item) for item in value]
    if hasattr(value, "item"):
        try:
            return clean_for_mongo(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@dataclass(frozen=True)
class JobQueue:
    jobs: Any
    worker_type: str
    worker_status: Any = None

    def claim(self, worker_id: str, *, worker_pool: Optional[str] = None) -> Optional[Dict[str, Any]]:
        now = utcnow()
        query: Dict[str, Any] = {"status": "pending"}
        pool = str(worker_pool or "").strip()
        if pool:
            query["worker_pool"] = pool
        else:
            query["$or"] = [{"worker_pool": {"$exists": False}}, {"worker_pool": None}, {"worker_pool": ""}]
        return self.jobs.find_one_and_update(
            query,
            {
                "$set": {
                    "status": "running",
                    "worker_id": worker_id,
                    "started_at": now,
                    "updated_at": now,
                    "heartbeat_at": now,
                    "error_message": None,
                },
                "$inc": {"attempts": 1},
            },
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    def requeue_stale(self, stale_minutes: int) -> int:
        if stale_minutes <= 0:
            return 0
        now = utcnow()
        cutoff = now - timedelta(minutes=stale_minutes)
        result = self.jobs.update_many(
            {
                "status": "running",
                "$or": [
                    {"heartbeat_at": {"$lt": cutoff}},
                    {"heartbeat_at": {"$exists": False}, "updated_at": {"$lt": cutoff}},
                ],
            },
            {
                "$set": {
                    "status": "pending",
                    "worker_id": None,
                    "error_message": f"requeued after running stale for {stale_minutes} minutes",
                    "updated_at": now,
                }
            },
        )
        return int(getattr(result, "modified_count", 0) or 0)

    def requeue_own_orphans(self, worker_id: str) -> int:
        now = utcnow()
        result = self.jobs.update_many(
            {"status": "running", "worker_id": worker_id},
            {
                "$set": {
                    "status": "pending",
                    "worker_id": None,
                    "error_message": f"requeued after worker {worker_id} restarted",
                    "updated_at": now,
                }
            },
        )
        return int(getattr(result, "modified_count", 0) or 0)

    def heartbeat(self, job_id: str, stage: str, message: Optional[str] = None) -> None:
        now = utcnow()
        update: Dict[str, Any] = {"updated_at": now, "heartbeat_at": now, "progress_stage": stage}
        if message is not None:
            update["progress_message"] = message
        self.jobs.update_one({"job_id": job_id}, {"$set": update})

    def fail(self, job_id: str, error: BaseException) -> None:
        now = utcnow()
        self.jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "error_message": str(error), "completed_at": now, "updated_at": now}},
        )

    def complete(self, job_id: str, **fields: Any) -> None:
        now = utcnow()
        self.jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": "completed", "completed_at": now, "updated_at": now, **fields}},
        )

    def record_worker_status(self, worker_id: str, status: str, fields: Optional[Dict[str, Any]] = None) -> None:
        if self.worker_status is None:
            return
        now = utcnow()
        try:
            self.worker_status.update_one(
                {"worker_type": self.worker_type, "worker_id": worker_id},
                {
                    "$set": {
                        "worker_id": worker_id,
                        "worker_type": self.worker_type,
                        "status": status,
                        "last_seen_at": now,
                        "updated_at": now,
                        **(fields or {}),
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        except PyMongoError:
            log.warning("worker status write failed", exc_info=True)


class HeartbeatTicker:
    def __init__(self, queue: JobQueue, job_id: str, interval_seconds: float) -> None:
        self._queue = queue
        self._job_id = job_id
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "HeartbeatTicker":
        if self._interval > 0 and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ml-heartbeat", daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            now = utcnow()
            self._queue.jobs.update_one(
                {"job_id": self._job_id, "status": "running"},
                {"$set": {"heartbeat_at": now, "updated_at": now}},
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


class WorkerLockHeld(RuntimeError):
    pass


@contextmanager
def worker_lock(lock_path: str, label: str = "worker"):
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise WorkerLockHeld(f"another {label} worker holds lock {path}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def run_worker_forever(
    step: Callable[[], None],
    *,
    interval_seconds: float,
    max_consecutive_failures: int,
    logger: logging.Logger,
) -> int:
    consecutive_failures = 0
    while True:
        try:
            step()
        except Exception:
            consecutive_failures += 1
            logger.exception("iteration failed (%d consecutive)", consecutive_failures)
            if consecutive_failures >= max_consecutive_failures:
                return 1
            time.sleep(min(60.0, max(1.0, interval_seconds) * 2 ** (consecutive_failures - 1)))
            continue
        consecutive_failures = 0
        time.sleep(max(1.0, interval_seconds))


def install_shutdown_handlers() -> None:
    def _handler(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
