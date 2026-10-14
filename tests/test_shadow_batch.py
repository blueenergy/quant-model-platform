from pathlib import Path

from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import COLLECTION_JOBS, COLLECTION_PREDICTIONS, COLLECTION_RESULTS
from quant_model_platform.shadow.batch import run_shadow_batch
from quant_model_platform.training.pipeline import run_training_job


class _MemColl:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(dict(doc))

    def insert_many(self, docs):
        for doc in docs:
            self.docs.append(dict(doc))

    def find_one(self, query, sort=None, projection=None):
        matches = [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]
        if not matches:
            return None
        if sort:
            key, direction = sort[0]
            matches = sorted(matches, key=lambda d: d.get(key))
            if direction < 0:
                matches = list(reversed(matches))
        return dict(matches[0])

    def delete_many(self, query):
        before = len(self.docs)
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in query.items())]
        return type("R", (), {"deleted_count": before - len(self.docs)})()

    def count_documents(self, query):
        return sum(1 for d in self.docs if all(d.get(k) == v for k, v in query.items()))


def test_shadow_batch_from_training_parquet(tmp_path: Path):
    result = run_training_job(
        {
            "index_code": "csi1000",
            "model_kind": "ridge",
            "train_days": 8,
            "test_days": 4,
            "purge_days": 2,
            "n_perm": 10,
        },
        parquet_root=tmp_path,
    )
    cfg = PlatformConfig(
        mongo_uri="mongodb://localhost:27017",
        quant_ml_db="quant_ml_test_shadow",
        mongo_db="finance",
        data_db_name="quant_data",
        parquet_root=str(tmp_path),
        s3_bucket="",
        s3_prefix="quant_ml",
        runtime="local",
        worker_pool="",
    )

    cols = {
        COLLECTION_JOBS: _MemColl(),
        COLLECTION_RESULTS: _MemColl(),
        COLLECTION_PREDICTIONS: _MemColl(),
    }
    job_id = "job-shadow-test-001"
    cols[COLLECTION_JOBS].insert_one(
        {
            "job_id": job_id,
            "params": {
                "index_code": "csi1000",
                "model_kind": "ridge",
                "parquet_path": result["parquet_path"],
                "snapshot_id": result["snapshot_id"],
            },
        }
    )
    cols[COLLECTION_RESULTS].insert_one({**result, "job_id": job_id})

    summary = run_shadow_batch({"source_job_id": job_id}, cfg, collections=cols)
    assert summary["n_predictions"] > 0
    assert summary["model_id"].startswith("promo-csi1000-")
    count = cols[COLLECTION_PREDICTIONS].count_documents({"model_id": summary["model_id"]})
    assert count == summary["n_predictions"]

    summary2 = run_shadow_batch({"source_job_id": job_id}, cfg, collections=cols)
    assert summary2["n_predictions"] == summary["n_predictions"]
    assert cols[COLLECTION_PREDICTIONS].count_documents({"model_id": summary["model_id"]}) == count
