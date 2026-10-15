import csv
import json
import random
from pathlib import Path

import pytest

from quant_model_platform.demo.portfolio import (
    COST_PER_SIDE,
    ROUND_TRIP_COST_RATE,
    compare_tracks,
    equity_stats,
    one_way_turnover,
    simulate_track,
)
from quant_model_platform.tools.demo_hs300_20d_top20 import main


def test_turnover_counts_replaced_names():
    assert one_way_turnover([], ["A", "B"]) == 1.0
    assert one_way_turnover(["A", "B"], ["A", "C"]) == 0.5


def test_costs_follow_the_frozen_round_trip_convention():
    rows = simulate_track(
        ["20250101", "20250121"],
        {"20250101": [{"ts_code": "AAA", "score": 1}], "20250121": [{"ts_code": "BBB", "score": 1}]},
        {"20250101": {"AAA": 0.0}, "20250121": {"BBB": 0.0}},
        top_n=1,
        track="t",
    )
    # Opening the book pays one side; swapping every name pays both.
    assert rows[0]["cost"] == pytest.approx(COST_PER_SIDE)
    assert rows[1]["cost"] == pytest.approx(ROUND_TRIP_COST_RATE)
    assert ROUND_TRIP_COST_RATE == pytest.approx(0.0027)


def test_missing_cross_section_holds_the_book_at_zero_turnover():
    dates = ["20250101", "20250121", "20250210"]
    scores = {"20250101": [{"ts_code": "AAA", "score": 1.0}]}
    forwards = {day: {"AAA": 0.02} for day in dates}
    rows = simulate_track(dates, scores, forwards, top_n=1, track="t")

    assert [row["holdings"] for row in rows] == [["AAA"], ["AAA"], ["AAA"]]
    assert [row["cross_section"] for row in rows] == ["selected", "carried", "carried"]
    assert rows[1]["one_way_turnover"] == 0.0
    assert rows[1]["cost"] == 0.0
    # The book is still held, so the period still earns its return.
    assert rows[1]["gross_return"] == pytest.approx(0.02)
    assert rows[1]["net_return"] == pytest.approx(0.02)


def test_carried_book_does_not_recharge_the_opening_side():
    dates = ["20250101", "20250121", "20250210"]
    scores = {"20250101": [{"ts_code": "AAA", "score": 1.0}]}
    forwards = {day: {"AAA": 0.0} for day in dates}
    rows = simulate_track(dates, scores, forwards, top_n=1, track="t")

    # A carried book is not a new book: only the first period is an opening.
    assert [row["cost"] for row in rows] == pytest.approx([COST_PER_SIDE, 0.0, 0.0])


def test_incomplete_forward_window_stays_out_of_the_nav():
    dates = ["20250101", "20250121"]
    scores = {day: [{"ts_code": "AAA", "score": 1.0}] for day in dates}
    forwards = {"20250101": {"AAA": 0.05}}
    rows = simulate_track(dates, scores, forwards, top_n=1, track="t")

    assert rows[1]["pricing"] == "none"
    assert rows[1]["net_return"] is None
    # The completed period compounds as itself, not as a flat zero, and the
    # unfinished one is reported rather than hidden.
    assert equity_stats(rows)["total_return"] == pytest.approx(0.05 - COST_PER_SIDE)

    report = compare_tracks(dates, scores, scores, forwards, top_n=1, hold_days=20)
    assert report["track_a"]["periods"] == 1
    assert report["track_a"]["dropped_periods"] == 1


def test_partial_pricing_is_averaged_over_the_priced_subset_and_flagged():
    rows = simulate_track(
        ["20250101"],
        {"20250101": [{"ts_code": "AAA", "score": 2}, {"ts_code": "BBB", "score": 1}]},
        {"20250101": {"AAA": 0.04}},
        top_n=2,
        track="t",
    )
    assert rows[0]["pricing"] == "partial"
    assert rows[0]["n_holdings"] == 2
    assert rows[0]["n_priced"] == 1
    assert rows[0]["gross_return"] == pytest.approx(0.04)


def test_model_beats_baseline_when_forwards_say_so():
    dates = ["20250101", "20250121"]
    scores_a = {
        "20250101": [{"ts_code": "AAA", "score": 2}, {"ts_code": "BBB", "score": 1}],
        "20250121": [{"ts_code": "AAA", "score": 2}, {"ts_code": "BBB", "score": 1}],
    }
    scores_b = {
        "20250101": [{"ts_code": "CCC", "score": 2}, {"ts_code": "DDD", "score": 1}],
        "20250121": [{"ts_code": "CCC", "score": 2}, {"ts_code": "DDD", "score": 1}],
    }
    forwards = {
        "20250101": {"AAA": 0.0, "BBB": 0.0, "CCC": 0.10, "DDD": 0.10},
        "20250121": {"AAA": 0.0, "BBB": 0.0, "CCC": 0.10, "DDD": 0.10},
    }
    report = compare_tracks(dates, scores_a, scores_b, forwards, top_n=2, hold_days=20)

    assert report["track_b"]["total_return"] > report["track_a"]["total_return"]
    assert report["comparison"]["avg_overlap_ratio"] == 0.0
    assert report["track_a"]["max_drawdown"] <= 0.0
    assert report["track_a"]["score_column"] == "growth_cycle_60:40"


def test_score_column_label_follows_the_parameter():
    dates = ["20250101"]
    scores = {dates[0]: [{"ts_code": "AAA", "score": 1.0}]}
    forwards = {dates[0]: {"AAA": 0.0}}
    report = compare_tracks(dates, scores, scores, forwards, top_n=1, score_column_a="composite_score")
    assert report["track_a"]["score_column"] == "composite_score"


def _noisy_pool(days, names, *, premium: float = 0.0):
    """A pool whose forward returns are demeaned, optionally plus a premium."""
    generator = random.Random(7)
    scores, forwards = {}, {}
    for day in days:
        raw = [generator.uniform(-0.1, 0.1) for _ in names]
        mean = sum(raw) / len(raw)
        scores[day] = [{"ts_code": name, "score": generator.random()} for name in names]
        forwards[day] = {name: value - mean + premium for name, value in zip(names, raw)}
    return scores, forwards


def test_random_control_track_reports_zero_excess():
    """The harness self-check: R sees the same pool and pays the same costs."""
    days = [f"2025{i:04d}" for i in range(1, 121)]
    scores, forwards = _noisy_pool(days, [f"S{i:03d}" for i in range(50)])

    report = compare_tracks(days, scores, scores, forwards, top_n=20, hold_days=1)

    assert report["track_r"]["periods"] == len(days)
    assert report["track_r"]["carried_periods"] == 0
    assert report["control_r"]["passed"] is True
    assert abs(report["control_r"]["t_stat"]) <= 3.0
    # R is not free: it pays the same costs as every other track.
    assert report["track_r"]["total_return"] < report["track_r"]["gross_mean_period_return"] * len(days)


def test_random_control_flags_a_pool_premium():
    """The check has teeth: a pool the harness fails to demean must trip it."""
    days = [f"2025{i:04d}" for i in range(1, 121)]
    scores, forwards = _noisy_pool(days, [f"S{i:03d}" for i in range(50)], premium=0.01)

    report = compare_tracks(days, scores, scores, forwards, top_n=20, hold_days=1)

    assert report["control_r"]["passed"] is False
    assert report["control_r"]["mean_period_return"] == pytest.approx(0.01, abs=0.002)


def _bundle(days, names):
    """Rules track ranks the pool backwards, model track forwards; both share a
    pool whose per-day forward returns are already demeaned, as the label is."""
    scores_a, scores_b, forwards = {}, {}, {}
    for index, day in enumerate(days):
        scores_a[day] = [{"ts_code": name, "score": float(len(names) - i)} for i, name in enumerate(names)]
        scores_b[day] = [{"ts_code": name, "score": float(i)} for i, name in enumerate(names)]
        raw = [((i * 7 + index * 3) % 11 - 5) / 1000.0 for i in range(len(names))]
        mean = sum(raw) / len(raw)
        forwards[day] = {name: value - mean for name, value in zip(names, raw)}
    return {"trading_days": days, "scores_a": scores_a, "scores_b": scores_b, "forward_returns": forwards}


def test_cli_writes_summary(tmp_path: Path):
    days = [f"2025{i:04d}" for i in range(1, 261)]
    bundle = _bundle(days, [f"S{i:02d}" for i in range(30)])
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    out = tmp_path / "out"

    code = main(["--bundle", str(path), "--years", "1", "--top-n", "10", "--rebalance-days", "20", "--out-dir", str(out)])

    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "equity"
    assert summary["window_trading_days"] == 252
    assert summary["rebalance_count"] == 13
    assert "total_return" in summary["track_a"]
    assert summary["track_r"]["periods"] == 13
    assert summary["control_r"]["t_stat"] is not None
    assert summary["cost"]["round_trip_bps"] == 27.0
    assert (out / "equity_curve.csv").is_file()
    assert (out / "rebalances.csv").is_file()


def _write_bundle(tmp_path: Path, days):
    bundle = _bundle(days, [f"S{i:02d}" for i in range(30)])
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_cli_years_zero_uses_the_whole_history(tmp_path: Path):
    """A 252-day window is 12 periods: too few to say anything. Widening must be free."""
    days = [f"2025{i:04d}" for i in range(1, 261)]
    out = tmp_path / "out"

    code = main(["--bundle", str(_write_bundle(tmp_path, days)), "--years", "0", "--out-dir", str(out)])

    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["window_trading_days"] == 260
    assert summary["rebalance_count"] == 13


def test_cli_names_the_run_by_its_real_window_not_a_label(tmp_path: Path):
    days = [f"2025{i:04d}" for i in range(1, 261)]
    out = tmp_path / "out"

    code = main(["--bundle", str(_write_bundle(tmp_path, days)), "--years", "0", "--out-dir", str(out)])

    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["demo_id"] == f"demo-hs300-20d-top10-{days[0]}-{days[-1]}"


def test_cli_top_n_defaults_to_the_contract_primary(tmp_path: Path):
    days = [f"2025{i:04d}" for i in range(1, 261)]
    out = tmp_path / "out"

    code = main(["--bundle", str(_write_bundle(tmp_path, days)), "--years", "1", "--out-dir", str(out)])

    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["top_n"] == 10


def test_cli_writes_the_provenance_of_its_own_outputs(tmp_path: Path):
    days = [f"2025{i:04d}" for i in range(1, 261)]
    out = tmp_path / "out"

    code = main(["--bundle", str(_write_bundle(tmp_path, days)), "--years", "1", "--out-dir", str(out)])

    assert code == 0
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["protocol"]["max_purge_days"] == 60
    assert meta["protocol"]["label_horizon_days"] == 20
    assert meta["window"]["trading_days"] == 252
    assert meta["requested"]["top_n"] == 10
    assert {entry["name"] for entry in meta["outputs"]} == {
        "summary.json",
        "equity_curve.csv",
        "rebalances.csv",
        "metrics_by_date.csv",
    }


def test_cli_writes_the_primary_block(tmp_path: Path):
    """The metric the contract calls primary, end to end through the CLI."""
    days = [f"2025{i:04d}" for i in range(1, 261)]
    out = tmp_path / "out"

    code = main(["--bundle", str(_write_bundle(tmp_path, days)), "--years", "1", "--out-dir", str(out)])

    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    primary = summary["primary"]
    # 13 rebalances on the 252-day window: few enough that the MDE is huge, which
    # is the honest thing for this fixture to report.
    assert primary["power"]["n_dates"] == primary["grid"]["n_dates"] == 13
    assert primary["power"]["effective_n"] == 13
    assert primary["grid"]["non_overlapping"] is True
    assert primary["pool"]["basis"] == "day_cross_section"
    assert primary["control"]["passed"] is True
    assert primary["tracks"]["baseline"]["rank_ic"]["n_dates"] == 13

    with (out / "metrics_by_date.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 13
    assert {row["track"] for row in rows} == {"baseline", "model", "control_random"}
    # The CSV is written from the same rows the summary is derived from, so the
    # two cannot disagree -- if this drifts, one of them grew a second source.
    baseline_excess = [
        float(row["excess_gross"]) for row in rows if row["track"] == "baseline" and row["excess_gross"] != ""
    ]
    assert len(baseline_excess) == 13
    assert sum(baseline_excess) / len(baseline_excess) == pytest.approx(
        primary["tracks"]["baseline"]["topk_excess_gross_vs_pool"]["mean"]
    )
    assert not [row for row in rows if "None" in row.values()]


def test_cli_degrades_without_a_source_job(tmp_path: Path):
    """No source job means no training evidence to quote -- and no failure."""
    days = [f"2025{i:04d}" for i in range(1, 261)]
    out = tmp_path / "out"

    code = main(["--bundle", str(_write_bundle(tmp_path, days)), "--years", "1", "--out-dir", str(out)])

    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_training"] is None


def test_cli_errors_instead_of_shrinking_the_window(tmp_path: Path, capsys):
    bundle = {
        "trading_days": ["20250101", "20250102"],
        "scores_a": {},
        "scores_b": {},
        "forward_returns": {},
    }
    path = tmp_path / "short.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    assert main(["--bundle", str(path), "--years", "2"]) == 2
    assert "only 2 available" in capsys.readouterr().err
