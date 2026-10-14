from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from quant_model_platform import db as db_mod
from quant_model_platform.dataset.channels import PRODUCTION_CURRENT_CHANNEL
from quant_model_platform.tools import materialize_snapshot as ms


class _Manifest:
    snapshot_id = "hs300-current"
    row_count = 10
    end_date = "20261014"

    def to_mongo(self):
        return {"snapshot_id": self.snapshot_id, "index_code": "hs300", "created_at": datetime.utcnow()}


class _Datasets:
    def __init__(self, indexes=None):
        self.updates = []
        self.inserts = []
        self.indexes = dict(indexes or {})
        self.dropped = []
        self.created = []

    def update_one(self, flt, update, upsert=False):
        self.updates.append((flt, update, upsert))

    def insert_one(self, doc):
        self.inserts.append(doc)

    def index_information(self):
        return self.indexes

    def drop_index(self, name):
        self.dropped.append(name)
        self.indexes.pop(name, None)

    def create_index(self, keys, **kwargs):
        self.created.append((keys, kwargs))


def _patch(monkeypatch, tmp_path, datasets):
    cfg = SimpleNamespace(parquet_root=str(tmp_path), mongo_db="quant_trading")
    client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(ms.PlatformConfig, "from_env", classmethod(lambda cls: cfg))
    monkeypatch.setattr(ms, "get_ml_client", lambda c: client)
    monkeypatch.setattr(ms, "latest_score_date_for_index", lambda *a, **k: "20261014")
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return None, _Manifest()

    monkeypatch.setattr(ms, "build_production_panel", fake_build)
    monkeypatch.setattr(
        ms, "publish_canonical_local_parquet", lambda *a, **k: "s3://b/quant_ml/snapshots/hs300/current.parquet"
    )
    monkeypatch.setattr(ms, "get_ml_collections", lambda config: {db_mod.COLLECTION_DATASETS: datasets})
    return seen


def test_canonical_latest_register_mongo(monkeypatch, tmp_path):
    datasets = _Datasets()
    seen = _patch(monkeypatch, tmp_path, datasets)

    rc = ms.main(
        ["--index-code", "hs300", "--start-date", "20230101", "--end-date", "latest", "--canonical", "--register-mongo"]
    )

    assert rc == 0
    assert seen["end_date"] == "20261014"
    assert seen["use_canonical_snapshot"] is True
    assert not datasets.inserts
    (flt, update, upsert), = datasets.updates
    assert flt == {"snapshot_id": "hs300-current"}
    assert upsert is True
    assert not set(update["$set"]) & set(update["$setOnInsert"])
    assert update["$set"]["channel"] == PRODUCTION_CURRENT_CHANNEL


def _ml_collections(datasets):
    other = _Datasets()
    return {
        db_mod.COLLECTION_JOBS: other,
        db_mod.COLLECTION_RESULTS: other,
        db_mod.COLLECTION_DATASETS: datasets,
        db_mod.COLLECTION_PREDICTIONS: other,
    }


def _channel_index_kwargs(datasets):
    return next(kw for keys, kw in datasets.created if kw.get("name") == "index_code_1_channel_1")


def test_ensure_indexes_replaces_legacy_sparse_channel_index():
    datasets = _Datasets({"index_code_1_channel_1": {"key": [("index_code", 1), ("channel", 1)], "sparse": True}})

    db_mod.ensure_ml_indexes(_ml_collections(datasets))

    assert datasets.dropped == ["index_code_1_channel_1"]
    kw = _channel_index_kwargs(datasets)
    assert kw["unique"] is True
    assert kw["partialFilterExpression"] == {"channel": {"$exists": True}}
    assert "sparse" not in kw


def test_ensure_indexes_keeps_partial_channel_index():
    datasets = _Datasets(
        {"index_code_1_channel_1": {"key": [], "unique": True, "partialFilterExpression": {"channel": {"$exists": True}}}}
    )

    db_mod.ensure_ml_indexes(_ml_collections(datasets))

    assert datasets.dropped == []
