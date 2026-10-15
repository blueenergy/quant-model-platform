import pytest

from quant_model_platform.demo.calendar import rebalance_signal_dates, window_trading_days
from quant_model_platform.demo.ranking import top_n_symbols


def test_window_last_252():
    seq = [f"{20240101 + i}" for i in range(300)]
    window = window_trading_days(seq, years=1)
    assert len(window) == 252
    assert window[0] == seq[-252]
    assert window[-1] == seq[-1]


def test_window_errors_when_history_is_shorter_than_requested():
    seq = [f"{20240101 + i}" for i in range(300)]
    with pytest.raises(ValueError, match="only 300 available"):
        window_trading_days(seq, years=2)


def test_window_deduplicates_before_counting():
    seq = [f"{20240101 + i}" for i in range(300)] + ["20240101"] * 10
    assert len(window_trading_days(seq, years=1)) == 252


def test_rebalance_every_20():
    days = [f"202501{i:02d}" for i in range(1, 61)]
    signals = rebalance_signal_dates(days, interval=20)
    assert signals == [days[0], days[20], days[40]]


def test_top_n_ties_break_by_symbol():
    rows = [
        {"ts_code": "B", "score": 2.0},
        {"ts_code": "A", "score": 2.0},
        {"ts_code": "C", "score": 1.0},
    ]
    assert top_n_symbols(rows, score_key="score", n=2) == ["A", "B"]
