from quant_model_platform.demo.calendar import rebalance_signal_dates, window_trading_days
from quant_model_platform.demo.ranking import top_n_symbols


def test_window_last_252():
    days = [f"2024{m:02d}01" for m in range(1, 13)]
    days += [f"2025{i:04d}"[-4:] for i in range(300)]
    # simpler: sequential fake yyyymmdd not needed if we pass 300 strings
    seq = [f"{20240101 + i}" for i in range(300)]
    window = window_trading_days(seq, years=1)
    assert len(window) == 252
    assert window[0] == seq[-252]
    assert window[-1] == seq[-1]


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
