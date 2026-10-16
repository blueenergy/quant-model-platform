"""Trailing raw factors, so a model can train on inputs instead of on scores.

The registry (``dataset/features.py``) has always declared these names; the
production panel only ever filled the four ``*_score`` columns. A score is a
weighted blend somebody else chose, so a model trained on scores inherits
those weights unexamined. These are the raw quantities instead.

Every factor here looks strictly backwards from the signal date and is
therefore PIT-safe by construction -- no announcement dates, no restatements.

Two things to know when reading the definitions:

* ``shift`` counts *observations*, not calendar sessions. A symbol halted for a
  week reaches back further in wall-clock time than one that traded every day.
  This matches ``stock-scoring-system/analysis/core/entry_stock_factors.py``,
  which does ``close / close.shift(20) - 1.0`` the same way.
* ``close`` must already be forward-adjusted. ``volume_price.close`` is the raw
  traded price, so a split inside the window reads as a huge fake loss --
  exactly the bug ``add8d17`` fixed for the label. Adjustment happens in
  ``dataset/production.py``, not here.

A window that is not fully covered stays NaN rather than being computed from a
short sample: a 60-day momentum silently computed off 12 observations would
look like data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = (20, 60)
VOLATILITY_WINDOW = 20
TURNOVER_WINDOW = 20

PRICE_FACTORS = ("momentum_20d", "momentum_60d", "volatility_20d")
DAILY_BASIC_FACTORS = ("turnover_20d", "log_market_cap")

#: Every column these derivations are responsible for. Single source of truth
#: for the panel builder, the tests, and the training-job feature lists.
RAW_FEATURE_NAMES = PRICE_FACTORS + DAILY_BASIC_FACTORS

#: Observations of history required before the first factor value is defined.
#: The warm-up the caller loads must cover at least this many sessions.
WARMUP_OBSERVATIONS = max(*MOMENTUM_WINDOWS, VOLATILITY_WINDOW, TURNOVER_WINDOW)

_TRADING_DAYS = 252.0
_KEYS = ("trade_date", "ts_code")


def _prepare(frame: pd.DataFrame, columns: Sequence[str], what: str) -> pd.DataFrame:
    """Keys plus ``columns``, one row per symbol in chronological order.

    The index is reset so every ``groupby`` result below lines up positionally:
    index-based alignment would silently scramble values on duplicate labels.
    """
    missing = [c for c in (*_KEYS, *columns) if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} frame is missing column(s): {missing}")
    out = frame.loc[:, [*_KEYS, *columns]].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _empty(keys: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    out = keys.reset_index(drop=True)
    for name in names:
        out[name] = pd.Series(dtype="float64")
    return out


def _rolling_std_by_symbol(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    """Trailing sample std per symbol, NaN until ``window`` observations exist."""
    rolled = values.groupby(codes, sort=False).rolling(window, min_periods=window)
    return rolled.std(ddof=1).reset_index(level=0, drop=True)


def _rolling_mean_by_symbol(values: pd.Series, codes: pd.Series, window: int) -> pd.Series:
    rolled = values.groupby(codes, sort=False).rolling(window, min_periods=window)
    return rolled.mean().reset_index(level=0, drop=True)


def derive_price_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """Momentum and volatility from forward-adjusted closes.

    ``prices`` must carry ``trade_date``, ``ts_code`` and an adjusted ``close``.
    """
    out = _prepare(prices, ("close",), "price")
    if out.empty:
        return _empty(prices.loc[:, list(_KEYS)], PRICE_FACTORS)

    closes = out["close"]
    codes = out["ts_code"]
    grouped = closes.groupby(codes, sort=False)

    for window in MOMENTUM_WINDOWS:
        out[f"momentum_{window}d"] = closes / grouped.shift(window) - 1.0

    # Written as an explicit ratio rather than pct_change() so a missing close
    # propagates as NaN instead of being silently forward-filled.
    daily = closes / grouped.shift(1) - 1.0
    out["volatility_20d"] = (
        _rolling_std_by_symbol(daily, codes, VOLATILITY_WINDOW) * np.sqrt(_TRADING_DAYS)
    )
    return out.loc[:, [*_KEYS, *PRICE_FACTORS]]


def derive_daily_basic_factors(basic: pd.DataFrame) -> pd.DataFrame:
    """Turnover and size from ``quant_data.financial_daily_basic``.

    That collection is already daily and keyed by (ts_code, trade_date), so no
    announcement-date backfill is involved -- take the row as of the signal date.

    ``circ_mv`` is quoted in 万元; ``log_market_cap`` is the natural log of the
    same quantity in 元. The unit choice only shifts the column by a constant,
    which any standardiser removes, but 元 keeps the number interpretable.
    """
    out = _prepare(basic, ("circ_mv", "turnover_rate"), "daily basic")
    if out.empty:
        return _empty(basic.loc[:, list(_KEYS)], DAILY_BASIC_FACTORS)

    # A non-positive market cap is a missing sentinel, not a company worth
    # nothing; log() would turn it into -inf and poison the standardiser.
    cap = out["circ_mv"].where(out["circ_mv"] > 0)
    out["log_market_cap"] = np.log(cap * 1e4)
    out["turnover_20d"] = _rolling_mean_by_symbol(
        out["turnover_rate"], out["ts_code"], TURNOVER_WINDOW
    )
    return out.loc[:, [*_KEYS, *DAILY_BASIC_FACTORS]]
