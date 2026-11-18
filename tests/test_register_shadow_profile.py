import pytest

from quant_model_platform.tools import register_shadow_profile as tool


class _Profiles:
    def __init__(self):
        self.docs = []

    def update_one(self, query, update, upsert=False):
        for doc in self.docs:
            if doc.get("profile_id") == query.get("profile_id"):
                doc.update(update.get("$set") or {})
                return
        doc = dict(update.get("$set") or {})
        doc["profile_id"] = query.get("profile_id")
        doc.update(update.get("$setOnInsert") or {})
        self.docs.append(doc)


def _patch(monkeypatch, merged):
    profiles = _Profiles()
    monkeypatch.setattr(
        tool,
        "get_ml_collections",
        lambda: {"model_shadow_profiles": profiles},
    )
    monkeypatch.setattr(tool, "ensure_ml_indexes", lambda collections: None)
    monkeypatch.setattr(
        tool,
        "merge_job_into_shadow_params",
        lambda params, collections: dict(merged),
    )
    return profiles


def test_register_profile_defaults_to_saved_artifact(monkeypatch):
    profiles = _patch(
        monkeypatch,
        {
            "source_job_id": "job-1",
            "model_id": "model-1",
            "index_code": "csi1000",
            "artifact_uri": "/data/artifact.joblib",
            "parquet_path": "/data/panel.parquet",
            "model_kind": "lightgbm",
        },
    )
    tool.main(["--profile-id", "p1", "--source-job-id", "job-1", "--model-id", "model-1"])
    doc = profiles.docs[0]
    assert doc["enabled"] is True
    assert doc["params"]["use_saved_artifact"] is True
    assert doc["params"]["artifact_uri"] == "/data/artifact.joblib"


def test_register_profile_can_opt_out_of_saved_artifact(monkeypatch):
    profiles = _patch(
        monkeypatch,
        {
            "source_job_id": "job-1",
            "model_id": "model-1",
            "index_code": "csi1000",
            "artifact_uri": "/data/artifact.joblib",
        },
    )
    tool.main([
        "--profile-id",
        "p1",
        "--source-job-id",
        "job-1",
        "--no-use-saved-artifact",
    ])
    assert profiles.docs[0]["params"]["use_saved_artifact"] is False


def test_register_profile_requires_artifact_when_using_saved_artifact(monkeypatch):
    _patch(
        monkeypatch,
        {"source_job_id": "job-1", "model_id": "model-1", "index_code": "csi1000"},
    )
    with pytest.raises(SystemExit):
        tool.main(["--profile-id", "p1", "--source-job-id", "job-1"])
