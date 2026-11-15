from types import SimpleNamespace

import pandas as pd
import pytest

from quant_model_platform.shadow.batch import assert_model_features
from quant_model_platform.training.models import TrainedModel
from quant_model_platform.workers import shadow_loop


def _model(features):
    return TrainedModel(
        kind="lightgbm", feature_names=list(features), artifact={}, params={}
    )


def test_assert_model_features_passes_when_all_populated():
    day = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert_model_features(day, _model(["a", "b"]), "m")


def test_assert_model_features_rejects_missing_column():
    day = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="not populated"):
        assert_model_features(day, _model(["a", "b"]), "m")


def test_assert_model_features_rejects_all_nan_column():
    day = pd.DataFrame({"a": [1.0], "b": [float("nan")]})
    with pytest.raises(ValueError, match="not populated"):
        assert_model_features(day, _model(["a", "b"]), "m")


def test_shadow_loop_continues_after_one_profile_fails(monkeypatch):
    collections = {}
    monkeypatch.setattr(shadow_loop, "get_ml_collections", lambda config: collections)
    monkeypatch.setattr(
        shadow_loop,
        "load_enabled_shadow_profiles",
        lambda cols: [{"profile_id": "bad"}, {"profile_id": "good"}],
    )
    monkeypatch.setattr(
        shadow_loop,
        "resolve_profile_params",
        lambda profile, cols: {"model_id": profile["profile_id"]},
    )
    calls = []

    def fake_run(params, config, collections=None):
        calls.append(params["model_id"])
        if params["model_id"] == "bad":
            raise RuntimeError("boom")
        return {"model_id": "good"}

    monkeypatch.setattr(shadow_loop, "run_shadow_batch", fake_run)

    with pytest.raises(RuntimeError, match="1 shadow profile"):
        shadow_loop.run_all_configured_shadow_batches(SimpleNamespace())

    assert calls == ["bad", "good"]


def test_shadow_loop_filters_profiles_by_model_id(monkeypatch):
    collections = {}
    monkeypatch.setattr(shadow_loop, "get_ml_collections", lambda config: collections)
    monkeypatch.setattr(
        shadow_loop,
        "load_enabled_shadow_profiles",
        lambda cols: [
            {"profile_id": "live", "model_id": "research-csi1000-clean-raw5"},
            {"profile_id": "other", "model_id": "promo-hs300"},
        ],
    )
    monkeypatch.setattr(
        shadow_loop,
        "resolve_profile_params",
        lambda profile, cols: {"model_id": profile["model_id"]},
    )
    calls = []
    monkeypatch.setattr(
        shadow_loop,
        "run_shadow_batch",
        lambda params, config, collections=None: calls.append(params["model_id"]) or {},
    )

    shadow_loop.run_all_configured_shadow_batches(
        SimpleNamespace(), profile_ids=["research-csi1000-clean-raw5"]
    )

    assert calls == ["research-csi1000-clean-raw5"]


def test_shadow_loop_unknown_profile_filter_fails_loudly(monkeypatch):
    collections = {}
    monkeypatch.setattr(shadow_loop, "get_ml_collections", lambda config: collections)
    monkeypatch.setattr(
        shadow_loop,
        "load_enabled_shadow_profiles",
        lambda cols: [{"profile_id": "live", "model_id": "research-csi1000-clean-raw5"}],
    )
    monkeypatch.setattr(shadow_loop, "run_shadow_batch", lambda *a, **k: {})

    with pytest.raises(RuntimeError, match="no enabled shadow profile matches"):
        shadow_loop.run_all_configured_shadow_batches(
            SimpleNamespace(), profile_ids=["missing"]
        )
