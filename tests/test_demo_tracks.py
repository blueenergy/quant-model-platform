import pandas as pd
import pytest

from quant_model_platform.db import COLLECTION_PREDICTIONS, COLLECTION_RESULTS
from quant_model_platform.demo.backfill_pit import pit_train_dates, predict_cross_section_as_of, truncate_panel
from quant_model_platform.demo.ranking import top_n_symbols
from quant_model_platform.demo.track_a import baseline_score_rows, growth_cycle_composite
from quant_model_platform.demo.track_b import (
    cross_section_rank_correlation,
    load_shadow_predictions,
    resolve_model_scores,
    scores_by_day,
    training_evidence,
)
from quant_model_platform.protocol import LABEL_HORIZON_DAYS, MAX_PURGE_DAYS


def _pit_panel(days, symbols=("AAA", "BBB")):
    """Two names over ``days``, labelled everywhere but the final day."""
    rows = []
    for day in days:
        for code, level in zip(symbols, (1.0, -1.0)):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "growth_score": level,
                    "label_fwd_20d": 0.01 * level if day < days[-1] else None,
                }
            )
    return pd.DataFrame(rows)


def test_growth_cycle_blend_60_40():
    # 0.6 * 100 + 0.4 * 50
    assert growth_cycle_composite(100, 50) == 80


def test_baseline_uses_stored_column_then_blend():
    docs = [
        {"symbol": "AAA", "score_date": "20250102", "composite_growth_cycle_score": 90, "growth_score": 1, "cycle_score": 1},
        {"symbol": "BBB", "score_date": "20250102", "growth_score": 100, "cycle_score": 0},
    ]
    rows = baseline_score_rows(docs, score_column="composite_growth_cycle_score")
    by_symbol = {row["ts_code"]: row["score"] for row in rows}
    assert by_symbol["AAA"] == 90
    assert by_symbol["BBB"] == 60
    assert top_n_symbols(rows, score_key="score", n=1) == ["AAA"]


def test_baseline_defaults_to_the_self_computed_blend():
    """The contract scores track A itself: production has no composite column."""
    docs = [{"symbol": "AAA", "score_date": "20250102", "composite_growth_cycle_score": 90, "growth_score": 100, "cycle_score": 50}]
    assert baseline_score_rows(docs)[0]["score"] == 80


def test_shadow_live_skips_backfill():
    live = [{"ts_code": "AAA", "trade_date": "20250120", "score": 1.5}]
    rows = resolve_model_scores(live, panel=None, params={}, as_of="20250120")
    assert rows[0]["score_source"] == "shadow_live"
    assert rows[0]["score"] == 1.5


def test_shadow_live_rows_for_other_dates_are_not_this_day():
    live = [{"ts_code": "AAA", "trade_date": "20250119", "score": 1.5}]
    with pytest.raises(ValueError, match="no shadow scores on 20250120"):
        resolve_model_scores(live, panel=None, params={}, as_of="20250120")


def test_one_stray_live_row_cannot_displace_a_backfilled_day():
    """The boundary is a date, not a row count (DEMO doc section 1.1)."""
    days = [f"2025{i:04d}" for i in range(1, 201)]
    panel = _pit_panel(days)
    as_of = days[-2]
    stray = [{"ts_code": "AAA", "trade_date": as_of, "score": 9.9}]
    rows = resolve_model_scores(
        stray,
        panel,
        {"model_kind": "ridge", "feature_names": ["growth_score"]},
        as_of,
        shadow_live_from=days[-1],
    )
    assert {row["score_source"] for row in rows} == {"backfill_pit"}
    assert len(rows) == 2


def test_a_live_day_never_falls_back_to_a_refit():
    with pytest.raises(ValueError, match="refusing to backfill"):
        resolve_model_scores([], panel=None, params={}, as_of="20250120", shadow_live_from="20250101")


def test_truncate_panel_drops_future_dates():
    panel = pd.DataFrame(
        {
            "trade_date": ["20250101", "20250102", "20250103"],
            "ts_code": ["A", "A", "A"],
            "label_fwd_20d": [0.1, 0.2, 0.3],
        }
    )
    cut = truncate_panel(panel, "20250102")
    assert set(cut["trade_date"]) == {"20250101", "20250102"}


def test_pit_train_dates_exclude_label_horizon_and_purge():
    days = [f"202501{i:02d}" for i in range(1, 11)]
    allowed = pit_train_dates(days, days[-1], horizon=2, purge=3)
    # as_of index 9, max train index 9-2-1-3=3 -> days[0:4]
    assert allowed[-1] == days[3]
    assert days[-1] not in allowed


def test_default_purge_is_the_protocol_embargo():
    """Contract section 6: the training window ends at or before T - horizon - purge."""
    days = [f"2025{i:04d}" for i in range(1, 401)]
    allowed = pit_train_dates(days, days[-1])
    gap = days.index(days[-1]) - days.index(allowed[-1])
    assert gap >= LABEL_HORIZON_DAYS + MAX_PURGE_DAYS - 1
    assert gap <= LABEL_HORIZON_DAYS + MAX_PURGE_DAYS + 1


def test_an_unpurged_refit_is_the_live_recipe():
    """purge=0 is the shadow recipe: every labelled row, labelled rows end at T-21."""
    days = [f"2025{i:04d}" for i in range(1, 401)]
    allowed = pit_train_dates(days, days[-1], purge=0)
    assert len(allowed) == len(pit_train_dates(days, days[-1])) + MAX_PURGE_DAYS


def test_backfill_predicts_only_signal_day():
    dates = [f"20250{i:03d}" for i in range(101, 131)]
    rows = []
    for day in dates:
        for code, level in (("AAA", 1.0), ("BBB", -1.0)):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "growth_score": level,
                    "label_fwd_20d": 0.01 * level if day < dates[-1] else None,
                }
            )
    panel = pd.DataFrame(rows)
    as_of = dates[-1]
    scored = predict_cross_section_as_of(
        panel,
        {"model_kind": "ridge", "feature_names": ["growth_score"]},
        as_of,
        horizon=5,
        purge=5,
    )
    assert set(scored["trade_date"].astype(str).str[:8]) == {as_of}
    assert set(scored["score_source"]) == {"backfill_pit"}
    assert len(scored) == 2


class _FakePredictions:
    """Minimal stand-in for quant_ml.model_predictions."""

    def __init__(self, docs):
        self.docs = list(docs)
        self.queries = []

    def find(self, query, projection=None):
        self.queries.append(query)
        out = []
        for doc in self.docs:
            if not all(_matches(doc, key, want) for key, want in query.items()):
                continue
            out.append(doc)
        return out


def _matches(doc, key, want):
    value = doc.get(key)
    if isinstance(want, dict):
        if "$gte" in want and not str(value)[:8] >= want["$gte"]:
            return False
        if "$lte" in want and not str(value)[:8] <= want["$lte"]:
            return False
        return True
    return value == want


def test_live_rows_are_read_only_for_the_named_model():
    collections = {
        COLLECTION_PREDICTIONS: _FakePredictions(
            [
                {"shadow": True, "model_id": "promo-hs300-abc", "index_code": "hs300", "trade_date": "20261014", "ts_code": "AAA", "score": 1.0},
                {"shadow": True, "model_id": "other", "index_code": "hs300", "trade_date": "20261014", "ts_code": "BBB", "score": 2.0},
                {"shadow": False, "model_id": "promo-hs300-abc", "index_code": "hs300", "trade_date": "20261014", "ts_code": "CCC", "score": 3.0},
                {"shadow": True, "model_id": "promo-hs300-abc", "index_code": "hs300", "trade_date": "20260930", "ts_code": "DDD", "score": 4.0},
            ]
        )
    }
    rows = load_shadow_predictions(
        collections,
        model_id="promo-hs300-abc",
        index_code="hs300",
        start="20261001",
        end="20261014",
    )
    assert [row["ts_code"] for row in rows] == ["AAA"]
    assert rows[0]["score_source"] == "shadow_live"
    assert scores_by_day(rows) == {"20261014": {"AAA": 1.0}}


def test_overlap_spearman_catches_a_disagreeing_rank():
    live = [{"ts_code": s, "trade_date": "20261014", "score": float(i)} for i, s in enumerate("ABCDEF")]
    same = [{"ts_code": s, "trade_date": "20261014", "score": float(i)} for i, s in enumerate("ABCDEF")]
    report = cross_section_rank_correlation(live, same)
    assert report["mean_spearman"] == 1.0
    assert report["passes"] is True
    assert report["n_scored"] == 1


def test_overlap_spearman_flags_a_thin_or_reversed_overlap():
    live = [{"ts_code": s, "trade_date": "20261014", "score": float(i)} for i, s in enumerate("ABCDEF")]
    reversed_rows = [{"ts_code": s, "trade_date": "20261014", "score": float(-i)} for i, s in enumerate("ABCDEF")]
    assert cross_section_rank_correlation(live, reversed_rows)["passes"] is False
    thin = reversed_rows[:2]
    report = cross_section_rank_correlation(live, thin)
    assert report["shared_days"] == 1
    assert report["n_scored"] == 0
    assert report["passes"] is False
    assert report["days"][0]["spearman"] is None


def test_overlap_ignores_days_only_one_side_has():
    live = [{"ts_code": "AAA", "trade_date": "20261014", "score": 1.0}]
    backfill = [{"ts_code": "AAA", "trade_date": "20261013", "score": 1.0}]
    report = cross_section_rank_correlation(live, backfill)
    assert report["shared_days"] == 0
    assert report["days"] == []
    assert report["passes"] is False


class _FakeResults:
    """Minimal stand-in for quant_ml.model_training_results."""

    def __init__(self, docs):
        self.docs = list(docs)
        self.queries = []

    def find_one(self, query, projection=None, sort=None):
        self.queries.append((query, sort))
        for doc in self.docs:
            if all(doc.get(key) == want for key, want in query.items()):
                return doc
        return None


def _training_result(**overrides):
    doc = {
        "job_id": "job-1",
        "result_id": "res-1",
        "created_at": "2026-10-01T00:00:00",
        "model_kind": "ridge",
        "code_version": "sha-00a4f45",
        "snapshot_id": "snap-1",
        "artifact_hash": "deadbeef",
        "promotion": {"passed": False, "reason": "ic_below_baseline", "model_mean_ic": 0.0234},
        "permutation_p": 0.31,
        "baseline_ic": {"mean_ic": 0.0493},
        "fold_reports": [
            {"model": {"mean_ic": 0.02, "avg_turnover": 0.5, "n_days": 100},
             "baseline": {"mean_ic": 0.05, "avg_turnover": 0.4, "n_days": 100}},
            {"model": {"mean_ic": 0.03, "avg_turnover": 0.3, "n_days": 80},
             "baseline": {"mean_ic": 0.07, "avg_turnover": 0.2, "n_days": 80}},
        ],
    }
    doc.update(overrides)
    return doc


def test_training_evidence_quotes_the_verdict_it_was_given():
    """The gate judged this run; the demo repeats the judgement, it does not retry it."""
    collections = {COLLECTION_RESULTS: _FakeResults([_training_result()])}

    block = training_evidence(collections, job_id="job-1")

    assert block["promotion"] == {"passed": False, "reason": "ic_below_baseline", "model_mean_ic": 0.0234}
    assert block["artifact_hash"] == "deadbeef"
    assert block["result_id"] == "res-1"
    assert block["comparable_to_primary"] is False
    # 272 fold rows already live in Mongo; copying them into a one-shot output
    # directory is the duplication the aggregate-into-JSON rule exists to avoid.
    assert "fold_reports" not in block  # the 272 raw rows stay in Mongo


def test_training_evidence_reduces_the_folds_the_way_the_gate_does():
    """Its mean has to be the number the gate compared, not a neighbour of it."""
    collections = {COLLECTION_RESULTS: _FakeResults([_training_result()])}

    folds = training_evidence(collections, job_id="job-1")["folds"]

    assert folds["model"]["mean_ic"] == pytest.approx(0.025)
    assert folds["baseline"]["mean_ic"] == pytest.approx(0.06)
    assert folds["model_minus_baseline"] == pytest.approx(-0.035)
    assert folds["model"]["sd_ic"] == pytest.approx(0.0070711, abs=1e-6)
    assert folds["model"]["share_positive"] == 1.0
    assert folds["model"]["n_days_summed"] == 180
    # The pooled baseline the gate compared against is quoted as it stands, not
    # recomputed: the fold mean above and this number are two different averages
    # of nearly the same data, which is why the note has to say so.
    assert training_evidence(collections, job_id="job-1")["baseline_ic"] == {"mean_ic": 0.0493}


@pytest.mark.parametrize(
    "collections, job_id",
    [
        ({COLLECTION_RESULTS: _FakeResults([_training_result()])}, ""),
        ({COLLECTION_RESULTS: _FakeResults([_training_result()])}, "job-unknown"),
        ({COLLECTION_RESULTS: _FakeResults([])}, "job-1"),
        ({}, "job-1"),
    ],
)
def test_training_evidence_degrades_instead_of_raising(collections, job_id):
    """A missing audit trail must not stop the measurement it annotates."""
    assert training_evidence(collections, job_id=job_id) is None


def test_training_evidence_skips_folds_with_no_ic():
    collections = {
        COLLECTION_RESULTS: _FakeResults(
            [_training_result(fold_reports=[{"model": {"mean_ic": None}, "baseline": {"mean_ic": 0.05}}])]
        )
    }

    folds = training_evidence(collections, job_id="job-1")["folds"]

    assert folds["model"]["n_folds"] == 0
    assert folds["model"]["mean_ic"] is None
    assert folds["model"]["sd_ic"] is None
    assert folds["model_minus_baseline"] is None  # not a number invented from one side
    assert folds["baseline"]["mean_ic"] == pytest.approx(0.05)
