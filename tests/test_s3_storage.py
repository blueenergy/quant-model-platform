import os

import pytest

from quant_model_platform.config import PlatformConfig
from quant_model_platform.storage.s3 import (
    is_s3_uri,
    s3_env_complete,
    s3_uri_for_snapshot,
    should_use_s3_parquet,
    snapshot_object_key,
)


def test_snapshot_object_key():
    key = snapshot_object_key("csi1000", "abc-123", prefix="quant_ml")
    assert key == "quant_ml/snapshots/csi1000/abc-123.parquet"


def test_s3_uri_for_snapshot():
    os.environ["QUANT_ML_S3_BUCKET"] = "aipoc"
    cfg = PlatformConfig.from_env()
    uri = s3_uri_for_snapshot("hs300", "snap1", config=cfg)
    assert uri == "s3://aipoc/quant_ml/snapshots/hs300/snap1.parquet"


def test_should_use_s3_external_k8s(monkeypatch):
    monkeypatch.setenv("MODEL_TRAINING_RUNTIME", "external_k8s")
    monkeypatch.setenv("QUANT_ML_S3_ENDPOINT", "https://s3.example")
    monkeypatch.setenv("QUANT_ML_S3_BUCKET", "b")
    monkeypatch.setenv("QUANT_ML_S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("QUANT_ML_S3_SECRET_KEY", "sk")
    assert should_use_s3_parquet()


def test_should_not_use_s3_local_docker(monkeypatch):
    monkeypatch.setenv("MODEL_TRAINING_RUNTIME", "local_docker")
    monkeypatch.delenv("QUANT_ML_PARQUET_BACKEND", raising=False)
    assert not should_use_s3_parquet()


def test_is_s3_uri():
    assert is_s3_uri("s3://bucket/key.parquet")
    assert not is_s3_uri("/tmp/x.parquet")
