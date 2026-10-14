"""quant_ml Mongo collections (never writes production score/trading DBs)."""

from __future__ import annotations

from typing import Any, Dict

from pymongo import ASCENDING, MongoClient

from quant_model_platform.config import PlatformConfig

COLLECTION_JOBS = "model_training_jobs"
COLLECTION_RESULTS = "model_training_results"
COLLECTION_DATASETS = "dataset_manifests"
COLLECTION_PREDICTIONS = "model_predictions"
COLLECTION_SHADOW_JOBS = "model_shadow_jobs"
COLLECTION_SHADOW_PROFILES = "model_shadow_profiles"
COLLECTION_WORKER_STATUS = "model_training_worker_status"


def get_ml_client(config: PlatformConfig | None = None) -> MongoClient:
    cfg = config or PlatformConfig.from_env()
    return MongoClient(cfg.mongo_uri)


def get_ml_collections(client: MongoClient | None = None, config: PlatformConfig | None = None) -> Dict[str, Any]:
    cfg = config or PlatformConfig.from_env()
    db = (client or get_ml_client(cfg))[cfg.quant_ml_db]
    return {
        "db": db,
        COLLECTION_JOBS: db[COLLECTION_JOBS],
        COLLECTION_RESULTS: db[COLLECTION_RESULTS],
        COLLECTION_DATASETS: db[COLLECTION_DATASETS],
        COLLECTION_PREDICTIONS: db[COLLECTION_PREDICTIONS],
        COLLECTION_SHADOW_JOBS: db[COLLECTION_SHADOW_JOBS],
        COLLECTION_SHADOW_PROFILES: db[COLLECTION_SHADOW_PROFILES],
        COLLECTION_WORKER_STATUS: db[COLLECTION_WORKER_STATUS],
    }


def ensure_ml_indexes(collections: Dict[str, Any]) -> None:
    jobs = collections[COLLECTION_JOBS]
    jobs.create_index([("job_id", ASCENDING)], unique=True)
    jobs.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
    results = collections[COLLECTION_RESULTS]
    results.create_index([("result_id", ASCENDING)], unique=True)
    results.create_index([("job_id", ASCENDING)])
    datasets = collections[COLLECTION_DATASETS]
    datasets.create_index([("snapshot_id", ASCENDING)], unique=True)
    datasets.create_index([("index_code", ASCENDING), ("channel", ASCENDING)], unique=True, sparse=True)
    datasets.create_index([("content_hash", ASCENDING)])
    preds = collections[COLLECTION_PREDICTIONS]
    preds.create_index([("model_id", ASCENDING), ("trade_date", ASCENDING), ("ts_code", ASCENDING)])
    preds.create_index([("trade_date", ASCENDING)])
    profiles = collections.get(COLLECTION_SHADOW_PROFILES)
    if profiles is not None:
        profiles.create_index([("profile_id", ASCENDING)], unique=True)
