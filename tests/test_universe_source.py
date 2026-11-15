from quant_model_platform.dataset.features import (
    RAW_FEATURE_REGISTRY,
    feature_registry,
    registry_names,
)
from quant_model_platform.dataset.production import select_snapshot_markers


def test_snapshot_published_after_trade_date_is_not_usable():
    """A weight snapshot published on 11-02 must not enter the 10-30 panel."""
    markers = [
        ("20260930", "20261030"),
        ("20261030", "20261102"),
    ]
    assert select_snapshot_markers(["20261030", "20261102", "20261103"], markers) == [
        ("20261030", "20260930", "20261030"),
        ("20261102", "20261030", "20261102"),
        ("20261103", "20261030", "20261102"),
    ]


def test_snapshot_uses_most_recent_publication():
    markers = [
        ("20261030", "20261030"),
        ("20261030", "20261102"),
        ("20260930", "20261030"),
    ]
    assert select_snapshot_markers(["20261102"], markers) == [
        ("20261102", "20261030", "20261102")
    ]


def test_raw_registry_has_no_rule_scores():
    names = registry_names(feature_registry("raw"))
    assert [f.name for f in RAW_FEATURE_REGISTRY] == names
    assert "growth_score" not in names
    assert "cycle_score" not in names
    assert {"momentum_20d", "volatility_20d", "turnover_20d", "log_market_cap"} <= set(
        names
    )


def test_legacy_registry_keeps_rule_scores():
    names = registry_names(feature_registry("legacy"))
    assert {"growth_score", "cycle_score", "value_score", "quality_score"} <= set(names)
