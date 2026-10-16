from pathlib import Path

import pandas as pd

from quant_model_platform.dataset.builder import build_synthetic_dataset
from quant_model_platform.dataset.features import (
    DEFAULT_FEATURE_REGISTRY,
    annotate_population,
    registry_document,
)
from quant_model_platform.dataset.manifest import build_manifest, content_hash, read_parquet_snapshot


def test_snapshot_hash_reproducible(tmp_path: Path):
    panel1, m1 = build_synthetic_dataset(parquet_dir=tmp_path)
    panel2 = read_parquet_snapshot(tmp_path / "csi1000_synthetic.parquet")
    h1 = content_hash(panel1, extra={"snapshot": m1.snapshot_id})
    h2 = content_hash(panel2, extra={"snapshot": m1.snapshot_id})
    assert h1 == h2
    assert m1.row_count == len(panel1)


def test_registry_records_only_what_the_frame_populates():
    """The production path fills 4 of the 14 registry features; the manifest must say so.

    `pit` is a declared policy about the source, so an entry keeps it whether or
    not the column arrived -- `populated` is the observation that separates the
    two, and without it the manifest claimed 14 point-in-time features.
    """
    frame = pd.DataFrame({"growth_score": [1.0, 2.0], "momentum_20d": [None, None]})
    docs = {d["name"]: d for d in registry_document(frame=frame)}
    assert docs["growth_score"]["populated"] is True
    assert docs["growth_score"]["fill_rate"] == 1.0
    assert docs["momentum_20d"]["populated"] is False
    assert docs["momentum_20d"]["fill_rate"] == 0.0
    assert docs["momentum_20d"]["pit"] is True
    assert docs["momentum_20d"]["source"] == "prices"


def test_registry_without_a_frame_stays_a_plan():
    docs = registry_document()
    assert len(docs) == len(DEFAULT_FEATURE_REGISTRY)
    assert "populated" not in docs[0]


def test_a_partly_filled_feature_reports_its_rate():
    frame = pd.DataFrame({"growth_score": [1.0, None, 3.0, 4.0]})
    docs = annotate_population([{"name": "growth_score"}, {"name": "absent"}], frame)
    assert docs[0]["fill_rate"] == 0.75
    assert docs[0]["populated"] is True
    assert docs[1]["populated"] is False


def test_a_panel_with_the_raw_factors_reports_them_populated():
    """The B6a snapshot fills all five; a manifest still saying populated=False
    would describe the plan rather than the data it was built from."""
    from quant_model_platform.dataset.raw_factors import RAW_FEATURE_NAMES

    frame = pd.DataFrame({"growth_score": [1.0] * 100})
    for name in RAW_FEATURE_NAMES:
        frame[name] = [float("nan")] + [1.0] * 99

    docs = {d["name"]: d for d in registry_document(frame=frame)}

    for name in RAW_FEATURE_NAMES:
        assert docs[name]["populated"] is True
        assert docs[name]["fill_rate"] > 0.9


def test_turnover_is_declared_against_the_source_it_actually_comes_from():
    """It is a 20-day mean of financial_daily_basic.turnover_rate, not a price."""
    docs = {d["name"]: d for d in registry_document()}

    assert docs["turnover_20d"]["source"] == "fundamentals"
    assert docs["momentum_20d"]["source"] == "prices"


def test_manifest_carries_the_population_of_its_own_frame(tmp_path: Path):
    frame = pd.DataFrame({"growth_score": [1.0, 2.0], "value_score": [None, None]})
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path, index=False)
    manifest = build_manifest(
        frame,
        index_code="hs300",
        start_date="20250101",
        end_date="20250102",
        parquet_path=path,
    )
    docs = {d["name"]: d for d in manifest.feature_registry}
    assert docs["growth_score"]["populated"] is True
    assert docs["value_score"]["populated"] is False
