"""Raw factors must be trailing, correctly windowed, and honest about gaps.

These are pure DataFrame tests; nothing here touches Mongo. The price inputs
are always forward-adjusted closes, which is the contract the production
loader has to honour (see tests/test_production_prices.py for that half).
"""

import numpy as np
import pandas as pd
import pytest

from quant_model_platform.dataset.raw_factors import (
    DAILY_BASIC_FACTORS,
    PRICE_FACTORS,
    RAW_FEATURE_NAMES,
    WARMUP_OBSERVATIONS,
    derive_daily_basic_factors,
    derive_price_factors,
)


def _sessions(n, start=1):
    """n distinct YYYYMMDD strings, so nothing depends on real calendars."""
    return [f"2024{m:02d}{d:02d}" for m, d in ((1 + i // 28, 1 + i % 28) for i in range(n))]


def _prices(closes, symbol="000001.SZ", start=1):
    return pd.DataFrame(
        {
            "trade_date": _sessions(len(closes), start=start),
            "ts_code": symbol,
            "close": [float(c) for c in closes],
        }
    )


def _basic(caps, turnovers, symbol="000001.SZ"):
    return pd.DataFrame(
        {
            "trade_date": _sessions(len(caps)),
            "ts_code": symbol,
            "circ_mv": [float(c) for c in caps],
            "turnover_rate": [float(t) for t in turnovers],
        }
    )


# --- shape -------------------------------------------------------------------


def test_the_two_derivations_cover_exactly_the_declared_factor_names():
    prices = _prices(range(1, 71))
    basic = _basic([1000] * 25, [1.0] * 25)

    assert set(derive_price_factors(prices).columns) == {"trade_date", "ts_code", *PRICE_FACTORS}
    assert set(derive_daily_basic_factors(basic).columns) == {
        "trade_date",
        "ts_code",
        *DAILY_BASIC_FACTORS,
    }
    assert tuple(PRICE_FACTORS) + tuple(DAILY_BASIC_FACTORS) == tuple(RAW_FEATURE_NAMES)


def test_a_missing_column_is_loud_rather_than_a_silent_nan_column():
    with pytest.raises(ValueError, match="close"):
        derive_price_factors(pd.DataFrame({"trade_date": [], "ts_code": []}))
    with pytest.raises(ValueError, match="circ_mv"):
        derive_daily_basic_factors(
            pd.DataFrame({"trade_date": [], "ts_code": [], "turnover_rate": []})
        )


def test_an_empty_frame_yields_the_declared_columns_not_a_keyerror():
    empty_prices = pd.DataFrame({"trade_date": [], "ts_code": [], "close": []})
    empty_basic = pd.DataFrame(
        {"trade_date": [], "ts_code": [], "circ_mv": [], "turnover_rate": []}
    )

    assert list(derive_price_factors(empty_prices).columns) == ["trade_date", "ts_code", *PRICE_FACTORS]
    assert list(derive_daily_basic_factors(empty_basic).columns) == [
        "trade_date",
        "ts_code",
        *DAILY_BASIC_FACTORS,
    ]


# --- definitions -------------------------------------------------------------


def test_momentum_is_the_return_over_the_window():
    closes = [100.0 + i for i in range(70)]
    prices = _prices(closes)
    got = pd.concat([prices, derive_price_factors(prices).drop(columns=["trade_date", "ts_code"])], axis=1)

    expected_20 = closes[30] / closes[10] - 1.0
    expected_60 = closes[65] / closes[5] - 1.0
    assert got["momentum_20d"].iloc[30] == pytest.approx(expected_20)
    assert got["momentum_60d"].iloc[65] == pytest.approx(expected_60)


def test_momentum_is_nan_until_the_window_is_covered():
    got = derive_price_factors(_prices([100.0 + i for i in range(70)]))

    assert got["momentum_20d"].iloc[:20].isna().all()
    assert got["momentum_20d"].iloc[20:].notna().all()
    assert got["momentum_60d"].iloc[:60].isna().all()
    assert got["momentum_60d"].iloc[60:].notna().all()


def test_volatility_is_the_annualised_trailing_std_of_daily_returns():
    closes = [100.0, 101.0, 99.0, 102.0, 98.0] * 15
    prices = _prices(closes)
    got = derive_price_factors(prices)["volatility_20d"]

    returns = pd.Series(closes).pct_change()
    expected = returns.iloc[1:21].std(ddof=1) * np.sqrt(252.0)
    assert got.iloc[20] == pytest.approx(expected)


def test_a_short_window_is_nan_not_a_short_sample_estimate():
    """Twelve observations must not silently produce a '60-day' momentum."""
    got = derive_price_factors(_prices([100.0 + i for i in range(12)]))

    assert got["momentum_20d"].isna().all()
    assert got["momentum_60d"].isna().all()
    assert got["volatility_20d"].isna().all()


def test_turnover_is_the_trailing_mean_and_size_is_log_yuan():
    basic = _basic([1000.0] * 25, [1.0 + i for i in range(25)])
    got = derive_daily_basic_factors(basic)

    assert got["turnover_20d"].iloc[:19].isna().all()
    assert got["turnover_20d"].iloc[19] == pytest.approx(np.mean(range(1, 21)))
    # 1000 万元 == 1e7 元
    assert got["log_market_cap"].iloc[0] == pytest.approx(np.log(1e7))


def test_a_non_positive_market_cap_is_nan_rather_than_minus_infinity():
    got = derive_daily_basic_factors(_basic([0.0, -5.0, 1000.0], [1.0, 1.0, 1.0]))

    assert got["log_market_cap"].iloc[:2].isna().all()
    assert np.isfinite(got["log_market_cap"].iloc[2])


# --- per-symbol isolation ----------------------------------------------------


def test_symbols_are_windows_independently():
    """A second symbol's history must not bleed into the first one's shift."""
    two = pd.concat(
        [_prices([100.0 + i for i in range(70)], symbol="000001.SZ"),
         _prices([10.0 * (i + 1) for i in range(70)], symbol="600000.SH")],
        ignore_index=True,
    )
    got = derive_price_factors(two)

    a = got[got["ts_code"] == "000001.SZ"]["momentum_20d"].reset_index(drop=True)
    b = got[got["ts_code"] == "600000.SH"]["momentum_20d"].reset_index(drop=True)
    assert a.iloc[30] == pytest.approx((130.0 / 110.0) - 1.0)
    assert b.iloc[30] == pytest.approx((310.0 / 110.0) - 1.0)


def test_out_of_order_rows_are_sorted_before_the_window_is_applied():
    shuffled = _prices([100.0 + i for i in range(70)]).sample(frac=1.0, random_state=7)
    got = derive_price_factors(shuffled).reset_index(drop=True)
    reference = derive_price_factors(_prices([100.0 + i for i in range(70)]))

    pd.testing.assert_series_equal(
        got["momentum_20d"], reference["momentum_20d"], check_names=False
    )


def test_a_duplicated_index_does_not_scramble_the_windows():
    """Index labels repeat here; positional grouping must still hold."""
    prices = _prices([100.0 + i for i in range(70)])
    prices.index = [i % 3 for i in range(70)]

    got = derive_price_factors(prices)
    reference = derive_price_factors(_prices([100.0 + i for i in range(70)]))
    pd.testing.assert_series_equal(
        got["momentum_20d"].reset_index(drop=True),
        reference["momentum_20d"].reset_index(drop=True),
        check_names=False,
    )


# --- the reason close has to be adjusted -------------------------------------


def test_a_split_inside_the_window_is_not_a_fake_loss_once_close_is_adjusted():
    """Raw closes halve at a 2:1 split; hfq-adjusted ones must keep the return flat.

    hfq anchors on the earliest price, so the *post-split* bars carry the
    cumulative factor -- the economic return across the split is zero.
    """
    split_at = 30
    raw = [10.0] * 40
    raw[split_at:] = [5.0] * 10
    adjusted = [c * (2.0 if i >= split_at else 1.0) for i, c in enumerate(raw)]

    fake = derive_price_factors(_prices(raw))["momentum_20d"]
    real = derive_price_factors(_prices(adjusted))["momentum_20d"]

    # The window straddling the split: unadjusted says -50%, adjusted says flat.
    assert fake.iloc[split_at + 5] == pytest.approx(-0.5)  # the bug
    assert real.iloc[split_at + 5] == pytest.approx(0.0)  # the fix


# --- warm-up contract --------------------------------------------------------


def test_the_declared_warmup_covers_every_window():
    longest = 60
    assert WARMUP_OBSERVATIONS >= longest
    for window in (20, 60):
        closes = [100.0 + i for i in range(longest + window)]
        got = derive_price_factors(_prices(closes))
        # The first row past the warm-up must already carry a value.
        assert pd.notna(got[f"momentum_{window}d"].iloc[longest])
