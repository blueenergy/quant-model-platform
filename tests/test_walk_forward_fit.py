"""Point-in-time guarantees for the walk-forward deployment fit.

The deployed artifact is the one thing that cannot be re-derived after the
fact: if it was fit on labels the market had not produced yet, every backtest
that used it is optimistic. These tests pin the cutoff arithmetic and the
"the full-sample fit was not used" property.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_model_platform.training.walk_forward import (
    DEFAULT_TRAIN_END_OFFSET_DAYS,
    resolve_train_end,
    signal_rows,
    walk_forward_fit,
)

FEATURES = ["momentum_20d", "momentum_60d", "volatility_20d", "turnover_20d", "log_market_cap"]


def _panel(n_dates: int = 120, n_symbols: int = 5, start: str = "20200101") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=n_dates).strftime("%Y%m%d")
    rows = []
    for date in dates:
        for i in range(n_symbols):
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": f"{i:06d}.SZ",
                    "label_fwd_20d": float(i) + (dates.get_loc(date) % 7) * 0.01,
                    **{f: 0.1 * i + 0.01 * j for j, f in enumerate(FEATURES)},
                }
            )
    return pd.DataFrame(rows)


def _params() -> dict:
    return {
        "model_kind": "ridge",
        "label_col": "label_fwd_20d",
        "feature_names": list(FEATURES),
        "model_params": {"alpha": 1.0},
    }


def test_train_end_is_the_label_horizon_before_the_signal_date():
    panel = _panel()
    dates = sorted(panel["trade_date"].unique())
    signal = dates[-1]
    train_end = resolve_train_end(dates, signal, DEFAULT_TRAIN_END_OFFSET_DAYS)
    # The label of a row dated d ends at the open of d + 21 sessions, so the
    # last training row must sit 21 sessions back, not 20.
    assert train_end == dates[-1 - DEFAULT_TRAIN_END_OFFSET_DAYS]


def test_train_end_counts_sessions_not_calendar_days():
    dates = pd.bdate_range(start="20200101", periods=40).strftime("%Y%m%d")
    # A holiday inside the window must not move the cutoff.
    assert resolve_train_end(dates, dates[-1], 5) == dates[-6]


def test_fit_never_trains_on_rows_after_the_train_end():
    panel = _panel()
    dates = sorted(panel["trade_date"].unique())
    signal = dates[-1]
    fit = walk_forward_fit(panel, _params(), signal_date=signal)
    train_end = resolve_train_end(dates, signal, DEFAULT_TRAIN_END_OFFSET_DAYS)
    assert fit.train_end == train_end
    # The signal date itself is never a training row.
    assert signal > fit.train_end


def test_training_rows_carry_no_label_from_after_the_signal_date():
    """The cutoff must be tight enough that the last label ends at T's open.

    A row dated ``train_end`` has a label window that ends at the open of the
    signal date; anything later would carry the move being traded.
    """
    panel = _panel()
    dates = sorted(panel["trade_date"].unique())
    signal = dates[-1]
    fit = walk_forward_fit(panel, _params(), signal_date=signal)
    assert fit.train_end == dates[-1 - DEFAULT_TRAIN_END_OFFSET_DAYS]
    assert fit.n_train_dates <= len(dates) - DEFAULT_TRAIN_END_OFFSET_DAYS


def test_fit_requires_enough_history():
    panel = _panel(n_dates=10)
    with pytest.raises(ValueError, match="not enough history"):
        walk_forward_fit(panel, _params(), signal_date=sorted(panel["trade_date"].unique())[-1])


def test_fit_rejects_a_signal_date_outside_the_panel():
    panel = _panel()
    with pytest.raises(ValueError, match="not in the panel"):
        walk_forward_fit(panel, _params(), signal_date="20991231")


def test_incomplete_feature_rows_are_dropped_by_default():
    panel = _panel()
    first = panel["trade_date"].min()
    mask = (panel["trade_date"] == first) & (panel["ts_code"] == "000000.SZ")
    panel.loc[mask, "turnover_20d"] = None
    fit = walk_forward_fit(panel, _params(), signal_date=sorted(panel["trade_date"].unique())[-1])
    complete = panel.dropna(subset=FEATURES)
    assert fit.n_train_rows == len(complete[complete["trade_date"] <= fit.train_end])


def test_signal_rows_drops_incomplete_rows():
    panel = _panel()
    signal = sorted(panel["trade_date"].unique())[-1]
    panel.loc[(panel["trade_date"] == signal) & (panel["ts_code"] == "000001.SZ"), "volatility_20d"] = None
    rows = signal_rows(panel, signal_date=signal, feature_names=FEATURES)
    assert "000001.SZ" not in set(rows["ts_code"])
    assert len(rows) == 4


def test_fit_reports_its_provenance():
    panel = _panel()
    signal = sorted(panel["trade_date"].unique())[-1]
    fit = walk_forward_fit(panel, _params(), signal_date=signal)
    meta = fit.as_dict()
    assert meta["train_mode"] == "walk_forward_expanding"
    assert meta["signal_date"] == signal
    assert meta["train_end"] < signal
    assert meta["n_train_rows"] > 0
    assert meta["feature_names"] == list(FEATURES)


def test_offset_is_configurable():
    """A larger offset is a more conservative cutoff, not a looser one.

    ``offset`` counts the sessions between the training cutoff and the signal
    date, so a bigger offset ends training earlier and trains on less data.
    """
    panel = _panel()
    dates = sorted(panel["trade_date"].unique())
    signal = dates[-1]
    near = walk_forward_fit(panel, _params(), signal_date=signal, train_end_offset_days=5)
    far = walk_forward_fit(panel, _params(), signal_date=signal, train_end_offset_days=40)
    assert near.train_end == dates[-6]
    assert far.train_end == dates[-41]
    assert near.n_train_rows > far.n_train_rows
    assert near.n_train_dates > far.n_train_dates
