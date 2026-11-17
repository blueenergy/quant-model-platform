"""Daily shadow predictions into quant_ml.model_predictions."""

from __future__ import annotations

import argparse
import logging
import os

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import COLLECTION_SHADOW_JOBS, ensure_ml_indexes, get_ml_collections
from quant_model_platform.queue import JobQueue, WorkerLockHeld, install_shutdown_handlers, run_worker_forever, worker_lock
from quant_model_platform.shadow.batch import run_shadow_batch
from quant_model_platform.shadow.registry_sync import sync_shadow_profile_states
from quant_model_platform.shadow.resolve import load_enabled_shadow_profiles, resolve_profile_params

log = logging.getLogger("model_shadow_loop")


def _profile_labels(profile: dict) -> set:
    """Names a CLI ``--profile-id`` may match: profile/model id or source job."""
    labels = set()
    for key in ("profile_id", "model_id"):
        value = profile.get(key)
        if value:
            labels.add(str(value))
    params = profile.get("params") or {}
    for key in ("model_id", "profile_id", "source_job_id"):
        value = params.get(key)
        if value:
            labels.add(str(value))
    return labels


def run_all_configured_shadow_batches(config: PlatformConfig, profile_ids=None) -> list:
    collections = get_ml_collections(config=config)
    try:
        sync_shadow_profile_states(collections)
    except Exception:
        # Profile sync must not block today's predictions; the next run retries.
        log.exception("shadow registry sync failed")
    profiles = load_enabled_shadow_profiles(collections)
    if not profiles:
        raise RuntimeError(
            "no shadow profiles configured; set MODEL_SHADOW_PROFILES JSON or model_shadow_profiles in Mongo"
        )
    requested = {str(value) for value in (profile_ids or []) if str(value).strip()}
    if requested:
        profiles = [p for p in profiles if requested & _profile_labels(p)]
        if not profiles:
            raise RuntimeError(
                f"no enabled shadow profile matches {sorted(requested)}"
            )
    summaries = []
    failures = []
    for profile in profiles:
        params = resolve_profile_params(profile, collections)
        name = profile.get("profile_id") or params.get("model_id") or params.get("source_job_id")
        log.info("shadow profile start: %s", name)
        try:
            summary = run_shadow_batch(params, config, collections=collections)
        except Exception as exc:  # one broken profile must not block the others
            log.exception("shadow profile failed: %s", name)
            failures.append({"profile": name, "error": str(exc)})
            continue
        summaries.append(summary)
        log.info("shadow profile done: %s -> %s", name, summary)
    if failures:
        # Still raise so the run is not green, but only after every healthy
        # profile (notably the live csi1000 one) has written its rows.
        raise RuntimeError(f"{len(failures)} shadow profile(s) failed: {failures}")
    return summaries


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Shadow prediction loop")
    parser.add_argument("--worker-id", default=os.getenv("MODEL_SHADOW_WORKER_ID", "shadow-1"))
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("MODEL_SHADOW_LOOP_INTERVAL_SECONDS", "300")))
    parser.add_argument("--lock-file", default=os.getenv("MODEL_SHADOW_LOCK_FILE", "/tmp/model_shadow_worker.lock"))
    parser.add_argument("--once", action="store_true", help="Run configured shadow profiles once and exit")
    parser.add_argument(
        "--profile-id",
        action="append",
        default=[],
        help="Run only profiles matching this id/model_id/source_job_id (repeatable)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Dev only: run one synthetic csi1000 batch (ignores MODEL_SHADOW_PROFILES)",
    )
    args = parser.parse_args(argv)
    install_shutdown_handlers()
    cfg = PlatformConfig.from_env()
    collections = get_ml_collections(config=cfg)
    ensure_ml_indexes(collections)
    queue = JobQueue(jobs=collections[COLLECTION_SHADOW_JOBS], worker_type="model_shadow")

    def step() -> None:
        job = queue.claim(args.worker_id)
        if not job:
            try:
                run_all_configured_shadow_batches(cfg, profile_ids=args.profile_id)
            except Exception:
                # A stale/retrain-needed profile must not take the continuous
                # worker down after max_consecutive_failures. --once still
                # exits non-zero so the cron log carries the failure.
                log.exception("configured shadow batch had failures; worker continues")
            return
        try:
            queue.heartbeat(job["job_id"], "shadow", "predict")
            summary = run_shadow_batch(dict(job.get("params") or {}), cfg, collections=collections)
            queue.complete(job["job_id"], **summary)
        except Exception as exc:
            queue.fail(job["job_id"], exc)
            raise

    if args.synthetic:
        summary = run_shadow_batch({"index_code": "csi1000", "model_kind": "ridge", "mode": "synthetic"}, cfg)
        log.info("shadow synthetic once: %s", summary)
        return 0

    if args.once:
        summaries = run_all_configured_shadow_batches(cfg, profile_ids=args.profile_id)
        log.info("shadow once complete: %s", summaries)
        return 0

    try:
        with worker_lock(args.lock_file, label="model-shadow"):
            return run_worker_forever(step, interval_seconds=args.interval_seconds, max_consecutive_failures=5, logger=log)
    except WorkerLockHeld as exc:
        log.warning("%s", exc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
