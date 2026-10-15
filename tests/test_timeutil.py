"""The platform's UTC convention: write aware UTC, read aware UTC, show CST."""

from datetime import datetime, timedelta, timezone

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import get_ml_client
from quant_model_platform.queue import utcnow as queue_utcnow
from quant_model_platform.timeutil import CST, to_cst, utcnow


def test_utcnow_is_aware_utc():
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_queue_shares_the_same_helper():
    assert queue_utcnow is utcnow


def test_to_cst_reads_naive_values_as_utc():
    # What a plain (tz_aware=False) client hands back: UTC numbers, no tzinfo.
    naive = datetime(2026, 10, 15, 4, 12, 35)
    assert to_cst(naive) == datetime(2026, 10, 15, 12, 12, 35, tzinfo=CST)


def test_to_cst_accepts_aware_values():
    aware = datetime(2026, 10, 15, 4, 12, 35, tzinfo=timezone.utc)
    assert to_cst(aware) == datetime(2026, 10, 15, 12, 12, 35, tzinfo=CST)


def test_ml_client_decodes_timestamps_tz_aware():
    cfg = PlatformConfig(
        mongo_uri="mongodb://localhost:27017",
        quant_ml_db="quant_ml",
        mongo_db="finance",
        data_db_name="quant_data",
        parquet_root="/tmp/quant_ml/parquet",
        s3_bucket="",
        s3_prefix="quant_ml",
        runtime="local",
        worker_pool="",
    )
    client = get_ml_client(cfg)
    try:
        assert client.codec_options.tz_aware is True
    finally:
        client.close()
