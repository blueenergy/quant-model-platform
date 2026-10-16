import pandas as pd
import pytest

from quant_model_platform.dataset.labels import forward_open_to_open_return


def test_next_open_forward_return_uses_future_opens():
    prices = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "ts_code": "000001.SZ", "open": 10.0},
            {"trade_date": "2024-01-03", "ts_code": "000001.SZ", "open": 11.0},
            {"trade_date": "2024-01-04", "ts_code": "000001.SZ", "open": 12.0},
            {"trade_date": "2024-01-05", "ts_code": "000001.SZ", "open": 13.0},
        ]
    )
    ret = forward_open_to_open_return(prices, horizon=1)
    # Signal on 2024-01-02 -> enter 11, exit open after 1 day from entry = 12 -> 12/11-1
    assert abs(ret.iloc[0] - (12.0 / 11.0 - 1.0)) < 1e-9


# --- the label must not depend on the order Mongo happened to return --------


def _panel(dates, symbols):
    return pd.DataFrame([{"trade_date": d, "ts_code": s} for d in dates for s in symbols])


def _prices(dates, symbols):
    """Distinct, drifting prices so every (date, symbol) has its own label."""
    rows = []
    for j, s in enumerate(symbols):
        for i, d in enumerate(dates):
            rows.append({"trade_date": d, "ts_code": s, "open": 10.0 + i + 100.0 * j})
    return pd.DataFrame(rows)


DATES = [f"2024{m:02d}{d:02d}" for m in (1, 2, 3, 4) for d in range(1, 11)]
SYMBOLS = ["000001.SZ", "600000.SH"]


def test_the_label_does_not_change_when_the_price_rows_are_shuffled():
    """``load_price_history`` returns Mongo's order, which is not sorted.

    A positional assignment survived only because that order once matched the
    sort inside ``forward_open_to_open_return``; the day it stops matching,
    every row silently gets another row's return.
    """
    from quant_model_platform.dataset.labels import attach_label_column

    panel = _panel(DATES, SYMBOLS)
    prices = _prices(DATES, SYMBOLS)

    reference = attach_label_column(panel, prices, horizon=1)["label_fwd_20d"]
    for seed in (0, 7, 99):
        shuffled = prices.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        got = attach_label_column(panel, shuffled, horizon=1)["label_fwd_20d"]
        pd.testing.assert_series_equal(got, reference)


def test_the_label_does_not_change_when_the_symbols_are_interleaved():
    """Date-major order is the shape Mongo actually returns."""
    from quant_model_platform.dataset.labels import attach_label_column

    panel = _panel(DATES, SYMBOLS)
    prices = _prices(DATES, SYMBOLS)
    date_major = prices.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    pd.testing.assert_series_equal(
        attach_label_column(panel, date_major, horizon=1)["label_fwd_20d"],
        attach_label_column(panel, prices, horizon=1)["label_fwd_20d"],
    )


def test_the_label_matches_a_hand_computed_forward_return():
    """Order-invariance alone is not enough; the values must be the right ones."""
    from quant_model_platform.dataset.labels import attach_label_column

    panel = _panel(DATES, SYMBOLS)
    prices = _prices(DATES, SYMBOLS)
    date_major = prices.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    got = attach_label_column(panel, date_major, horizon=1, neutralize="none")
    got = got.set_index(["trade_date", "ts_code"])["label_fwd_20d"]

    one = date_major[date_major["ts_code"] == "000001.SZ"].reset_index(drop=True)
    # Entry is the next bar's open, exit the one after; prices step by 1.0.
    expected = (one["open"].shift(-2) / one["open"].shift(-1) - 1.0).tolist()
    for i, date in enumerate(DATES[:-2]):
        assert got.loc[(date, "000001.SZ")] == pytest.approx(expected[i])
    assert got.loc[(DATES[0], "600000.SH")] != got.loc[(DATES[0], "000001.SZ")]
