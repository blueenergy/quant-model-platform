import pandas as pd

from quant_model_platform.demo.backfill_pit import pit_train_dates, predict_cross_section_as_of, truncate_panel
from quant_model_platform.demo.ranking import top_n_symbols
from quant_model_platform.demo.track_a import baseline_score_rows, growth_cycle_composite
from quant_model_platform.demo.track_b import resolve_model_scores


def test_growth_cycle_blend_60_40():
    # 0.6 * 100 + 0.4 * 50
    assert growth_cycle_composite(100, 50) == 80


def test_baseline_uses_stored_column_then_blend():
    docs = [
        {"symbol": "AAA", "score_date": "20250102", "composite_growth_cycle_score": 90, "growth_score": 1, "cycle_score": 1},
        {"symbol": "BBB", "score_date": "20250102", "growth_score": 100, "cycle_score": 0},
    ]
    rows = baseline_score_rows(docs)
    by_symbol = {row["ts_code"]: row["score"] for row in rows}
    assert by_symbol["AAA"] == 90
    assert by_symbol["BBB"] == 60
    assert top_n_symbols(rows, score_key="score", n=1) == ["AAA"]


def test_shadow_live_skips_backfill():
    live = [{"ts_code": "AAA", "trade_date": "20250120", "score": 1.5}]
    rows = resolve_model_scores(live, panel=None, params={}, as_of="20250120")
    assert rows[0]["score_source"] == "shadow_live"
    assert rows[0]["score"] == 1.5


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


def test_pit_train_dates_exclude_label_horizon():
    days = [f"202501{i:02d}" for i in range(1, 11)]
    allowed = pit_train_dates(days, days[-1], horizon=2)
    # as_of index 9, max train index 9-2-1=6 -> days[0:7]
    assert allowed[-1] == days[6]
    assert days[-1] not in allowed


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
    )
    assert set(scored["trade_date"].astype(str).str[:8]) == {as_of}
    assert set(scored["score_source"]) == {"backfill_pit"}
    assert len(scored) == 2
