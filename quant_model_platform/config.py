"""Environment-backed configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformConfig:
    mongo_uri: str
    quant_ml_db: str
    parquet_root: str
    s3_bucket: str
    s3_prefix: str
    runtime: str
    worker_pool: str

    @classmethod
    def from_env(cls) -> "PlatformConfig":
        return cls(
            mongo_uri=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
            quant_ml_db=os.environ.get("QUANT_ML_DB", "quant_ml"),
            parquet_root=os.environ.get("QUANT_ML_PARQUET_ROOT", "/data/quant_ml/parquet"),
            s3_bucket=os.environ.get("QUANT_ML_S3_BUCKET", ""),
            s3_prefix=os.environ.get("QUANT_ML_S3_PREFIX", "quant_ml"),
            runtime=os.environ.get("MODEL_TRAINING_RUNTIME", "local"),
            worker_pool=os.environ.get("MODEL_TRAINING_WORKER_POOL", ""),
        )
