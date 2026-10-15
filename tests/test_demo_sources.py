"""The demo's own forward-return code has to match ``ScoreBacktester``.

These are the places where a plausible-looking reimplementation drifts: which
bar is the entry, when the validator refuses the trade, and what an unfinished
forward window means. Each case is a rule the production validation applies, so
a failure here is a demo number that cannot be reconciled with it.
"""

import pandas as pd

from quant_model_platform.demo.sources import (
    attach_hfq_factor,
    clean_price_rows,
    forward_return,
    forward_returns_by_signal_date,
    score_cross_sections,
)


def _bar(date, open_, close, *, high=None, low=None):
    return {
        "trade_date": date,
        "open": open_,
        "close": close,
        "high": open_ if high is None else high,
        "low": open_ if low is None else low,
    }


def _flat(periods, price=10.0, start="20240102"):
    """A run of ordinary bars, one session apart, all priced the same."""
    days = pd.bdate_range(start, periods=periods).strftime("%Y%m%d").tolist()
    return [_bar(day, price, price) for day in days], days


def test_entry_is_the_first_session_strictly_after_the_signal_date():
    rows, days = _flat(6)
    rows[2]["open"] = 20.0  # the signal day's own bar must never be the entry
    assert forward_return(rows, days[2], 2) == 0.0  # entry days[3], exit days[5], both 10


def test_the_signal_day_itself_can_be_an_entry_when_it_is_absent_from_the_history():
    """A score date with no bar still enters on the next session it does have."""
    rows = [_bar("20240102", 10.0, 10.0), _bar("20240108", 10.0, 10.0), _bar("20240109", 12.0, 12.0)]
    assert forward_return(rows, "20240103", 1) == 12.0 / 10.0 - 1


def test_a_one_price_limit_up_entry_bar_refuses_the_trade():
    rows = [
        _bar("20240102", 10.0, 10.0),
        _bar("20240103", 11.0, 11.0),  # +10% on one price: unbuyable
        _bar("20240104", 12.0, 12.0),
    ]
    assert forward_return(rows, "20240102", 1) is None


def test_a_resumption_one_price_limit_up_is_refused_even_without_a_signal_bar():
    """A deliberate divergence from ``ScoreBacktester`` (see ``sources``).

    The symbol was suspended on the signal date, so SBT's price window starts
    at the resumption bar, has no previous close, and skips the limit check --
    it will price a one-price limit-up that cannot be bought at the open. The
    demo's history reaches back past the suspension and refuses.
    """
    rows = [
        _bar("20241025", 10.0, 10.0),
        _bar("20241111", 11.0, 11.0),  # resumption: one price, +10%
        _bar("20241112", 12.0, 12.0),
    ]
    assert forward_return(rows, "20241105", 1) is None


def test_an_ordinary_up_bar_is_still_buyable():
    rows = [
        _bar("20240102", 10.0, 10.0),
        _bar("20240103", 10.5, 10.6, high=10.8, low=10.4),  # +6%, traded through
        _bar("20240104", 11.0, 11.0),
    ]
    assert forward_return(rows, "20240102", 1) == 11.0 / 10.5 - 1


def test_a_one_price_limit_down_exit_bar_delays_the_exit():
    rows = [
        _bar("20240102", 10.0, 10.0),
        _bar("20240103", 10.0, 10.0),
        _bar("20240104", 9.0, 9.0),  # target bar, limit down: cannot sell
        _bar("20240105", 9.2, 9.2),
    ]
    assert forward_return(rows, "20240102", 1) == 9.2 / 10.0 - 1


def test_the_exit_delay_gives_up_after_five_sessions():
    rows = [_bar("20240102", 10.0, 10.0), _bar("20240103", 10.0, 10.0)]
    price = 10.0
    for day in ("20240104", "20240105", "20240108", "20240109", "20240110", "20240111"):
        price *= 0.9
        rows.append(_bar(day, price, price))  # six limit-down bars in a row
    assert forward_return(rows, "20240102", 1) is None


def test_an_unfinished_forward_window_has_no_return_instead_of_zero():
    rows = [_bar("20240102", 10.0, 10.0), _bar("20240103", 10.0, 10.0)]
    assert forward_return(rows, "20240102", 20) is None


def test_rows_without_a_positive_close_are_dropped_before_indexing():
    rows = clean_price_rows(
        [
            {"trade_date": "20240102", "open": 10.0, "close": 10.0},
            {"trade_date": "20240103", "open": 99.0, "close": None},  # suspended print
            {"trade_date": "20240104", "open": 11.0, "close": 11.0},
            {"trade_date": "20240105", "open": 12.0, "close": 12.0},
        ]
    )
    assert [row["trade_date"] for row in rows] == ["20240102", "20240104", "20240105"]
    assert forward_return(rows, "20240102", 1) == 12.0 / 11.0 - 1


def test_a_missing_open_falls_back_to_the_close():
    rows = clean_price_rows(
        [
            {"trade_date": "20240102", "open": 10.0, "close": 10.0},
            {"trade_date": "20240103", "open": None, "close": 11.0},
            {"trade_date": "20240104", "open": 12.0, "close": 12.0},
        ]
    )
    assert forward_return(rows, "20240102", 1) == 12.0 / 11.0 - 1


def test_returns_are_keyed_by_signal_date_and_omit_the_unpriced():
    rows, days = _flat(3)
    other = [_bar("20240102", 5.0, 5.0)]  # no exit bar: unpriced, must be absent
    out = forward_returns_by_signal_date({"A": rows, "B": other}, [days[0]], horizon=1)
    assert out == {days[0]: {"A": 0.0}}


def test_attach_hfq_factor_takes_the_last_factor_at_or_before_each_date():
    prices = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": "20231220", "open": 10.0, "close": 10.0},
            {"symbol": "A", "trade_date": "20240103", "open": 10.0, "close": 10.0},
            {"symbol": "A", "trade_date": "20240104", "open": 5.0, "close": 5.0},
        ]
    )
    factors = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": "20231219", "adj_factor": 2.0},
            {"symbol": "A", "trade_date": "20240104", "adj_factor": 4.0},
        ]
    )
    out = attach_hfq_factor(prices, factors).set_index("trade_date")
    # 20240103 has no factor of its own and inherits 2.0 from 20231219.
    assert out.loc["20240103", "open"] == 20.0
    assert out.loc["20240104", "open"] == 20.0


def test_a_symbol_without_any_factor_is_left_alone():
    prices = pd.DataFrame([{"symbol": "B", "trade_date": "20240103", "open": 7.0, "close": 7.0}])
    factors = pd.DataFrame([{"symbol": "A", "trade_date": "20240103", "adj_factor": 3.0}])
    assert attach_hfq_factor(prices, factors)["open"].tolist() == [7.0]


def _doc(symbol, date, growth, cycle):
    return {"symbol": symbol, "score_date": date, "growth_score": growth, "cycle_score": cycle}


def test_a_cross_section_below_the_minimum_is_dropped():
    docs = [_doc(f"00000{i}.SZ", "20240102", 0.2 * i, 0.2 * i) for i in range(4)]
    assert score_cross_sections(docs) == {}


def test_a_cross_section_with_a_single_distinct_score_is_dropped():
    docs = [_doc(f"00000{i}.SZ", "20240102", 0.5, 0.5) for i in range(6)]
    assert score_cross_sections(docs) == {}


def test_a_usable_cross_section_carries_the_blend_and_the_source_label():
    docs = [_doc(f"00000{i}.SZ", "20240102", 0.2 * i, 0.2 * i) for i in range(6)]
    out = score_cross_sections(docs)
    assert list(out) == ["20240102"]
    assert len(out["20240102"]) == 6
    assert out["20240102"][0]["score_source"] == "baseline_growth_cycle"
    assert out["20240102"][1]["score"] == 0.2
