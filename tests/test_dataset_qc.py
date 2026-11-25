from __future__ import annotations

import pandas as pd

from quant_model_platform.dataset.qc import check_panel_quality


def _panel(dates=None, symbols=None):
    dates = dates or ["20260105", "20260106", "20260107"]
    symbols = symbols or ["000001.SZ", "000002.SZ"]
    rows = []
    for day in dates:
        for symbol in symbols:
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": symbol,
                    "momentum_20d": 0.01,
                    "momentum_60d": 0.02,
                    "volatility_20d": 0.03,
                    "turnover_20d": 0.04,
                    "log_market_cap": 10.0,
                    "label_fwd_20d": 0.05,
                }
            )
    return pd.DataFrame(rows)


def _check(panel, *, first=None, expected=None, features=None, label="label_fwd_20d"):
    return check_panel_quality(
        panel,
        index_code="csi1000",
        start_date=str(panel["trade_date"].min()),
        end_date=str(panel["trade_date"].max()),
        first_factor_dates=first if first is not None else {
            "000001.SZ": "20150105",
            "000002.SZ": "20150105",
        },
        expected_dates=expected if expected is not None else sorted(panel["trade_date"].unique()),
        feature_names=features
        if features is not None
        else ["momentum_20d", "momentum_60d", "volatility_20d", "turnover_20d", "log_market_cap"],
        label_col=label,
    )


def test_clean_panel_passes():
    report = _check(_panel())
    assert report["ok"] is True
    assert report["duplicate_keys"] == 0
    assert report["missing_factor_rows"] == 0
    assert report["missing_expected_dates"] == []


def test_duplicate_keys_fail():
    panel = pd.concat([_panel(), _panel().head(1)], ignore_index=True)
    report = _check(panel)
    assert report["ok"] is False
    assert report["duplicate_keys"] == 1
    assert any("duplicate" in error for error in report["errors"])


def test_factor_gap_inside_the_panel_fails():
    report = _check(_panel(), first={"000001.SZ": "20260106", "000002.SZ": "20150105"})
    assert report["ok"] is False
    assert report["missing_factor_rows"] == 1
    assert any("adjustment factor" in error for error in report["errors"])


def test_missing_calendar_date_fails():
    report = _check(_panel(), expected=["20260105", "20260106", "20260107", "20260108"])
    assert report["ok"] is False
    assert report["missing_expected_dates"] == ["20260108"]


def test_all_null_label_fails():
    panel = _panel()
    panel["label_fwd_20d"] = None
    report = _check(panel)
    assert report["ok"] is False
    assert any("entirely null" in error for error in report["errors"])


def test_2015_boundary_is_reported_as_a_known_artifact():
    dates = [f"2015{m:02d}{d:02d}" for m in (1, 2, 3) for d in (5, 6, 7, 8, 9)]
    panel = _panel(dates=dates, symbols=["000001.SZ"])
    panel["momentum_60d"] = 2.5
    report = _check(panel, expected=dates)
    assert report["ok"] is True
    assert report["known_artifact"]["rows"] == len(panel)
    assert report["known_artifact"]["momentum_60d_median"] == 2.5
    assert any("known 2015" in warning for warning in report["warnings"])


def test_coverage_drop_is_a_warning_not_an_error():
    dates = [f"202601{d:02d}" for d in range(1, 26)]
    rows = []
    for i, day in enumerate(dates):
        symbols = ["000001.SZ"] * (100 if i < 24 else 40)
        for j, symbol in enumerate(symbols):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": f"{j:06d}.SZ",
                    "momentum_20d": 0.01,
                    "momentum_60d": 0.02,
                    "volatility_20d": 0.03,
                    "turnover_20d": 0.04,
                    "log_market_cap": 10.0,
                    "label_fwd_20d": 0.05,
                }
            )
    panel = pd.DataFrame(rows)
    first = {code: "20150105" for code in panel["ts_code"].unique()}
    report = _check(panel, first=first, expected=dates)
    assert report["ok"] is True
    assert report["daily_count"]["anomalies"]
    assert any("coverage dropped" in warning for warning in report["warnings"])
