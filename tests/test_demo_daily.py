"""The daily NAV is only worth having if it prices the same strategy.

Two things are pinned here. First, that the chained daily value of a period
equals that period's row -- the extra resolution must not buy any extra
freedom, or the drawdown is describing a different book than the returns are.
Second, that the statistics see what the rebalance grid stepped over.
"""

import json

import pandas as pd
import pytest

from quant_model_platform.demo.daily import daily_block, drawdown_stats
from quant_model_platform.demo.portfolio import simulate_track
from quant_model_platform.demo.sources import forward_returns_by_signal_date


def _bar(day, open_, close):
    """An ordinary bar: high and low apart, so it is never a one-price limit."""
    return {
        "trade_date": day,
        "open": open_,
        "close": close,
        "high": max(open_, close) * 1.01,
        "low": min(open_, close) * 0.99,
    }


def _limit_down(day, price):
    return {"trade_date": day, "open": price, "close": price, "high": price, "low": price}


def _days(periods, start="20240102"):
    return pd.bdate_range(start, periods=periods).strftime("%Y%m%d").tolist()


def _ramp(days, *, start=10.0, step=0.1, skip=()):
    rows = []
    price = start
    for day in days:
        if day in skip:
            price += step
            continue
        rows.append(_bar(day, price, price + step))
        price += step
    return rows


def _rows_for(prices, signals, *, top_n=2, horizon=2):
    forwards = forward_returns_by_signal_date(prices, signals, horizon=horizon)
    scores = {
        day: [{"symbol": symbol, "score": float(len(prices) - i)} for i, symbol in enumerate(sorted(prices))]
        for day in signals
    }
    return simulate_track(signals, scores, forwards, top_n=top_n, track="baseline"), forwards


def _pool_rows(rows, prices, forwards):
    """The benchmark as a book like any other: hold the whole cross-section."""
    out = []
    for row in rows:
        labels = forwards.get(row["trade_date"]) or {}
        holdings = sorted(prices)
        priced = [float(labels[symbol]) for symbol in holdings if labels.get(symbol) is not None]
        gross = (sum(priced) / len(priced)) if priced else None
        out.append(
            {
                "trade_date": row["trade_date"],
                "holdings": holdings,
                "cross_section": "selected",
                "cost": 0.0,
                "gross_return": gross,
                "net_return": gross,
            }
        )
    return out


def _setup(periods=14, every=4, horizon=2):
    days = _days(periods)
    prices = {symbol: _ramp(days, start=10.0 + i) for i, symbol in enumerate(("AAA", "BBB", "CCC"))}
    signals = days[::every]
    rows, forwards = _rows_for(prices, signals, horizon=horizon)
    return days, prices, signals, forwards, rows


def _block(days, prices, forwards, rows, *, trading_days=None, relative_to="pool"):
    return daily_block(
        {"baseline": rows, "pool": _pool_rows(rows, prices, forwards)},
        prices,
        forwards,
        horizon=2,
        trading_days=days if trading_days is None else trading_days,
        relative_to=relative_to,
    )


def test_the_chained_daily_value_of_a_period_equals_its_row():
    days, prices, signals, forwards, rows = _setup()
    report = _block(days, prices, forwards, rows)["summary"]["reconcile"]
    assert report["gross_passed"] is True
    assert report["gross_max_abs_gap"]["baseline"] < 1e-12
    assert report["gross_max_abs_gap"]["pool"] < 1e-12


def test_the_daily_nav_compounds_exactly_the_period_returns():
    days, prices, signals, forwards, rows = _setup()
    summary = _block(days, prices, forwards, rows)["summary"]
    expected = 1.0
    for row in rows:
        if row["gross_return"] is not None:
            expected *= 1.0 + row["gross_return"]
    assert summary["tracks"]["baseline"]["total_return"] == pytest.approx(expected - 1.0, abs=1e-12)


def test_a_limit_down_exit_delays_the_sale_and_still_reconciles():
    days = _days(10)
    prices = {"AAA": [_bar(day, 10.0, 10.0) for day in days]}
    prices["AAA"][3] = _limit_down(days[3], 9.0)  # the scheduled exit bar cannot be sold
    rows, forwards = _rows_for(prices, [days[0]])
    report = _block(days, prices, forwards, rows)["summary"]
    assert report["reconcile"]["gross_passed"] is True
    # the sale slips to the next session that is not a one-price limit-down
    assert report["coverage"]["overlap_days"]["baseline"] == 0
    assert report["tracks"]["baseline"]["n_days"] > 0


def test_a_delayed_exit_past_the_next_entry_is_reported_as_an_overlap():
    days = _days(12)
    prices = {"AAA": [_bar(day, 10.0, 10.0) for day in days]}
    # Three sessions locked limit-down, each 10% under the last close: a bar that
    # merely *stays* at the limit has a move of zero and is not a limit-down, so
    # the delay only compounds while the price keeps falling.
    for index, price in zip((3, 4, 5), (9.0, 8.1, 7.29)):
        prices["AAA"][index] = _limit_down(days[index], price)
    signals = [days[0], days[4]]
    rows, forwards = _rows_for(prices, signals, top_n=1)
    summary = _block(days, prices, forwards, rows)["summary"]
    # the first period is still holding on the day the second one buys
    assert summary["coverage"]["overlap_days"]["baseline"] == 1
    assert summary["reconcile"]["gross_passed"] is True


def test_a_symbol_whose_next_bar_is_late_is_cash_until_its_own_entry():
    days = _days(10)
    prices = {
        "AAA": _ramp(days, skip=(days[1], days[2])),  # suspended right after the signal
        "BBB": _ramp(days, start=20.0),
    }
    rows, forwards = _rows_for(prices, [days[0]])
    summary = _block(days, prices, forwards, rows)["summary"]
    assert summary["reconcile"]["gross_passed"] is True
    # AAA enters on its own resumption bar, so the book is a day older than the grid
    assert summary["grid"]["first"] == days[0]


def test_a_period_with_no_price_holds_the_curve_flat_instead_of_zeroing_it():
    days, prices, signals, forwards, rows = _setup()
    summary = _block(days, prices, forwards, rows)["summary"]
    assert summary["coverage"]["periods_without_a_book"]["baseline"] >= 1
    assert summary["reconcile"]["gross_passed"] is True


def test_a_carried_period_is_counted_as_one():
    days, prices, signals, forwards, rows = _setup()
    rows[1]["cross_section"] = "carried"
    summary = _block(days, prices, forwards, rows)["summary"]
    assert summary["coverage"]["carried_periods"]["baseline"] == 1


def test_the_relative_series_is_the_book_over_the_benchmark():
    days, prices, signals, forwards, rows = _setup()
    block = _block(days, prices, forwards, rows)
    row = block["rows"][5]
    assert row["baseline_vs_pool"] == pytest.approx(row["baseline_gross"] / row["pool_gross"])
    assert "pool_vs_pool" not in row


def test_the_grid_keeps_the_days_the_last_exit_lands_on():
    days, prices, signals, forwards, rows = _setup()
    summary = _block(days, prices, forwards, rows, trading_days=days[:8])["summary"]
    assert summary["grid"]["n_days"] > 8
    assert summary["grid"]["days_beyond_window"] > 0


def test_the_multiplicative_cost_stays_within_its_stated_bound():
    days, prices, signals, forwards, rows = _setup()
    report = _block(days, prices, forwards, rows)["summary"]["reconcile"]
    gap = report["net_gap_vs_period_row"]["baseline"]
    bound = report["net_gap_bound"]["baseline"]
    assert gap <= bound + 1e-15
    # the pool pays nothing, so its two conventions cannot disagree at all
    assert report["net_gap_vs_period_row"]["pool"] == pytest.approx(0.0, abs=1e-15)


def test_a_book_with_no_priced_period_has_no_daily_series():
    days = _days(5)
    prices = {"AAA": _ramp(days)}
    rows = [
        {
            "trade_date": days[0],
            "holdings": ["AAA"],
            "cross_section": "selected",
            "cost": 0.0,
            "gross_return": None,
            "net_return": None,
        }
    ]
    assert daily_block({"baseline": rows}, prices, {}, horizon=2, trading_days=days) is None


def test_the_summary_serializes_without_nan_or_infinity():
    days, prices, signals, forwards, rows = _setup()
    text = json.dumps(_block(days, prices, forwards, rows)["summary"])
    assert "NaN" not in text
    assert "Infinity" not in text


def test_drawdown_stats_see_a_trough_between_two_grid_points():
    stats = drawdown_stats([1.0, 1.0, 0.5, 1.0])
    assert stats["max_drawdown"] == pytest.approx(-0.5)
    assert stats["peak_to_trough_days"] == 2
    assert stats["time_underwater_share"] == pytest.approx(0.25)
    assert stats["worst_day"] == pytest.approx(-0.5)


def test_downside_deviation_counts_only_the_downside():
    stats = drawdown_stats([1.0, 1.1, 0.99, 1.089])
    assert stats["downside_deviation"] == pytest.approx(0.1 * 252**0.5)
    assert stats["worst_day"] == pytest.approx(-0.1)


def test_drawdown_stats_of_nothing_is_empty():
    assert drawdown_stats([]) == {}
