from quant_model_platform.dataset.channels import PRODUCTION_CURRENT_CHANNEL
from quant_model_platform.db import COLLECTION_DATASETS, COLLECTION_JOBS, COLLECTION_RESULTS
from quant_model_platform.shadow.dataset import apply_canonical_parquet_resolution, load_current_manifest
from quant_model_platform.shadow.resolve import merge_job_into_shadow_params
from quant_model_platform.storage.s3 import canonical_snapshot_id


class _MemColl:
    def __init__(self, docs):
        self.docs = list(docs)

    def find_one(self, query, projection=None, sort=None):
        matches = [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]
        if not matches:
            return None
        if sort:
            key, direction = sort[0]
            matches = sorted(matches, key=lambda d: d.get(key))
            if direction < 0:
                matches = list(reversed(matches))
        return dict(matches[0])


def test_load_current_manifest_by_snapshot_id():
    cols = {
        COLLECTION_DATASETS: _MemColl(
            [
                {
                    "snapshot_id": "hs300-current",
                    "index_code": "hs300",
                    "channel": PRODUCTION_CURRENT_CHANNEL,
                    "parquet_path": "s3://b/quant_ml/snapshots/hs300/current.parquet",
                    "end_date": "20251231",
                }
            ]
        )
    }
    doc = load_current_manifest(cols, "hs300")
    assert doc["parquet_path"].endswith("current.parquet")


def test_apply_canonical_overrides_job_parquet(monkeypatch):
    monkeypatch.setenv("MODEL_SHADOW_MAX_STALE_CALENDAR_DAYS", "999")

    class _Client:
        def close(self):
            return None

    monkeypatch.setattr("quant_model_platform.shadow.dataset.get_ml_client", lambda: _Client())
    monkeypatch.setattr(
        "quant_model_platform.shadow.dataset.latest_score_date_for_index",
        lambda *args, **kwargs: "20251231",
    )

    def _fake_latest(*_a, **_k):
        return "20251231"

    monkeypatch.setattr(
        "quant_model_platform.shadow.dataset.latest_score_date_for_index",
        _fake_latest,
    )
    cols = {
        COLLECTION_DATASETS: _MemColl(
            [
                {
                    "snapshot_id": canonical_snapshot_id("hs300"),
                    "index_code": "hs300",
                    "channel": PRODUCTION_CURRENT_CHANNEL,
                    "parquet_path": "s3://b/current.parquet",
                    "end_date": "20251231",
                }
            ]
        )
    }
    out = apply_canonical_parquet_resolution(
        {"index_code": "hs300", "parquet_path": "s3://old/uuid.parquet"},
        cols,
        mongo_db="finance",
    )
    assert out["parquet_path"] == "s3://b/current.parquet"
    assert out["snapshot_id"] == "hs300-current"


def test_merge_job_uses_job_parquet_when_resolution_job(monkeypatch):
    monkeypatch.setenv("MODEL_SHADOW_PARQUET_RESOLUTION", "job")
    cols = {
        COLLECTION_JOBS: _MemColl(
            [
                {
                    "job_id": "j1",
                    "params": {
                        "index_code": "hs300",
                        "parquet_path": "s3://job/path.parquet",
                        "snapshot_id": "uuid-old",
                    },
                }
            ]
        ),
        COLLECTION_RESULTS: _MemColl([]),
        COLLECTION_DATASETS: _MemColl([]),
    }
    out = merge_job_into_shadow_params({"source_job_id": "j1"}, cols)
    assert out["parquet_path"] == "s3://job/path.parquet"
