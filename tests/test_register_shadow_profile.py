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


def test_register_profile_merges_walk_forward_params(monkeypatch):
    """A walk-forward profile carries its own recipe; the job only lends features.

    The source job exists to supply feature names and booster params -- its
    artifact is deliberately bypassed. If ``--params-json`` were applied before
    the job merge, or dropped, the profile would silently fall back to the
    full-sample artifact and the shadow would score a different model.
    """
    profiles = _patch(
        monkeypatch,
        {
            "source_job_id": "job-1",
            "model_id": "job-derived",
            "index_code": "csi1000",
            "artifact_uri": "/data/artifact.joblib",
            "parquet_path": "/data/job-panel.parquet",
            "feature_names": ["momentum_20d", "volatility_20d"],
            "model_params": {"num_leaves": 15},
        },
    )
    tool.main([
        "--profile-id", "p-wf",
        "--source-job-id", "job-1",
        "--model-id", "wf-model",
        "--no-use-saved-artifact",
        "--params-json",
        '{"train_mode":"walk_forward_expanding","train_end_offset_days":21,'
        '"parquet_path":"/data/snapshots/csi1000_current.parquet",'
        '"snapshot_id":"csi1000-current"}',
    ])
    params = profiles.docs[0]["params"]
    assert profiles.docs[0]["model_id"] == "wf-model"
    assert params["train_mode"] == "walk_forward_expanding"
    assert params["train_end_offset_days"] == 21
    # Overrides win over the job's values; the job's features survive.
    assert params["parquet_path"] == "/data/snapshots/csi1000_current.parquet"
    assert params["feature_names"] == ["momentum_20d", "volatility_20d"]
    assert params["use_saved_artifact"] is False


def test_register_profile_rejects_non_object_params_json(monkeypatch):
    _patch(
        monkeypatch,
        {
            "source_job_id": "job-1",
            "model_id": "model-1",
            "index_code": "csi1000",
            "artifact_uri": "/data/artifact.joblib",
        },
    )
    with pytest.raises(SystemExit):
        tool.main([
            "--profile-id", "p1",
            "--source-job-id", "job-1",
            "--params-json", '["not", "an", "object"]',
        ])
