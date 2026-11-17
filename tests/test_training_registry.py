from quant_model_platform.db import COLLECTION_MODELS
from quant_model_platform.training.registry import register_trained_model


class _Models:
    def __init__(self, docs=None):
        self.docs = dict(docs or {})

    def find_one(self, query, projection=None):
        return self.docs.get(query.get("model_id"))

    def insert_one(self, doc):
        self.docs[doc["model_id"]] = doc

    def update_one(self, query, update):
        doc = self.docs.get(query.get("_id"))
        if doc:
            doc.update(update.get("$set") or {})


def _job(model_id="registry-model"):
    return {
        "job_id": "job-1",
        "name": "clean-csi1000-raw5-test",
        "user_id": "u1",
        "params": {
            "model_id": model_id,
            "index_code": "csi1000",
            "model_kind": "lightgbm",
            "feature_names": ["momentum_20d"],
            "train_days": 500,
            "test_days": 20,
            "purge_days": 60,
            "start_date": "20150130",
            "end_date": "20241231",
        },
    }


def _result():
    return {
        "result_id": "result-1",
        "artifact_uri": "/data/artifact.joblib",
        "artifact_hash": "hash-1",
        "snapshot_id": "snap-1",
        "parquet_path": "/data/panel.parquet",
        "model_kind": "lightgbm",
        "promotion": {"passed": False, "reason": "insufficient_universes"},
        "code_version": "0.1.0",
    }


def test_register_new_training_model_in_research_state():
    models = _Models()
    model_id = register_trained_model(
        {COLLECTION_MODELS: models},
        _job(),
        _result(),
    )
    assert model_id == "registry-model"
    doc = models.docs["registry-model"]
    assert doc["state"] == "research"
    assert doc["accepted"] is False
    assert doc["artifacts"][0]["result_id"] == "result-1"
    assert doc["training_jobs"][0]["job_id"] == "job-1"
    assert doc["validation_runs"][0]["verdict"] == "fail"
    assert doc["operational_gate"]["passed"] is False
    assert doc["state_history"][0]["actor"] == "training_worker"


def test_register_rerun_preserves_existing_live_state_and_portfolio_validation():
    existing = {
        "_id": "registry-model",
        "model_id": "registry-model",
        "state": "live",
        "accepted": True,
        "accepted_reasons": ["live"],
        "state_flags": {"live_active": True, "shadow_enabled": False},
        "artifacts": [],
        "training_jobs": [],
        "validation_runs": [],
        "latest_validation": {
            "protocol": "portfolio_oos",
            "verdict": "weak",
            "result_id": "portfolio-1",
        },
        "state_history": [],
    }
    models = _Models({"registry-model": existing})
    register_trained_model({COLLECTION_MODELS: models}, _job(), _result())
    assert models.docs["registry-model"]["state"] == "live"
    assert models.docs["registry-model"]["latest_validation"]["protocol"] == "portfolio_oos"
    assert len(models.docs["registry-model"]["artifacts"]) == 1


def test_register_without_model_id_is_noop():
    models = _Models()
    assert register_trained_model({COLLECTION_MODELS: models}, _job(model_id=""), _result()) is None
    assert models.docs == {}
