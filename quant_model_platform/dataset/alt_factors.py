"""Alternative-channel factors: analyst revisions and unlock overhang.

Both channels cleared the novelty gate against the four production scores
(|daily Spearman| < 0.35 over the full window; llm-wiki B6e-1, 2026-10-17):
`analyst_rev_60d` max 0.171 (cycle), `unlock_overhang_60d` max 0.080 (value).

PIT discipline:

* ``earnings_report_rc.report_date`` is the research report's publication day,
  so the consensus series is known from that day onward -- the daily grid is
  built by forward-fill from publication, never from the forecast period.
* ``share_float`` rows are announced at ``ann_date``; an unlock scheduled for
  ``float_date`` only counts for days ``t`` with ``ann_date <= t``. The 60-day
  window sum is computed exactly via two cumulative sums (see
  ``_overhang_series``) rather than a shifted rolling window, which pandas 3
  no longer does reliably on unsorted/offset indices.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pymongo import MongoClient

log = logging.getLogger(__name__)

_ANALYST_COLLECTION = "earnings_report_rc"
_ANALYST_PROJECTION = {"_id": 0, "ts_code": 1, "report_date": 1, "eps": 1}
_UNLOCK_COLLECTION = "share_float"
_UNLOCK_PROJECTION = {"_id": 0, "ts_code": 1, "ann_date": 1, "float_date": 1, "float_ratio": 1}
_CHIP_COLLECTION = "cyq_perf"
_CHIP_PROJECTION = {"_id": 0, "symbol": 1, "trade_date": 1, "cost_15pct": 1,
                    "cost_50pct": 1, "cost_85pct": 1, "weight_avg": 1}
_SYMBOL_CHUNK = 250

ANALYST_FACTORS = ("analyst_rev_60d",)
UNLOCK_FACTORS = ("unlock_overhang_60d",)
CHIP_FACTORS = ("cyq_conc", "cyq_skew")
ALT_FEATURE_NAMES = ANALYST_FACTORS + UNLOCK_FACTORS + CHIP_FACTORS

#: Consensus smoothing window and its minimum sample, in calendar days.
_REV_WINDOW_DAYS = 60
_REV_MIN_PERIODS = 30
#: Extra lookback so the first panel day already has a full 60-day window and
#: a value 60 days before it.
_REV_LOOKBACK_DAYS = 130
#: The overhang horizon: unlocks scheduled inside this many calendar days.
_OVERHANG_WINDOW_DAYS = 60


def _parse_day(value: Any) -> Optional[datetime]:
    text = "".join(ch for ch in str(value or "") if ch.isdigit())[:8]
    if len(text) != 8:
        return None
    try:
        return datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return None


def _daily_grid(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(start_date, end_date, freq="D")


def _chunks(symbols: Sequence[str], size: int = _SYMBOL_CHUNK) -> List[List[str]]:
    unique = sorted({s for s in symbols if s})
    return [unique[i : i + size] for i in range(0, len(unique), size)]


def _revision_series(events: pd.DataFrame, grid: pd.DatetimeIndex) -> Optional[np.ndarray]:
    """60-day consensus level and its 60-day revision on a daily grid.

    ``events`` is (date, eps) with one row per publication day; the level is
    the forward-filled daily consensus, smoothed over a trailing 60-day
    window (>=30 observations), and the revision compares it with the same
    quantity 60 calendar days earlier. NaN until the window is full.
    """
    if events.empty:
        return None
    series = events.set_index("date")["eps"].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.reindex(grid, method="ffill")
    values = series.to_numpy(dtype="float64")
    level = pd.Series(values, index=grid).rolling(
        f"{_REV_WINDOW_DAYS}D", min_periods=_REV_MIN_PERIODS
    ).mean().to_numpy()
    prev_pos = np.searchsorted(grid.to_numpy(),
                               (grid - pd.Timedelta(days=_REV_WINDOW_DAYS)).to_numpy(),
                               side="right") - 1
    prev = np.where(prev_pos >= 0, level[np.clip(prev_pos, 0, None)], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rev = np.where(~np.isnan(prev) & (prev != 0),
                       (level - prev) / np.abs(prev), np.nan)
    return rev


def load_analyst_revision(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """``analyst_rev_60d`` per (trade_date, ts_code), NaN without coverage."""
    empty = pd.DataFrame(columns=["trade_date", "ts_code", *ANALYST_FACTORS])
    start_dt = _parse_day(start_date)
    end_dt = _parse_day(end_date)
    if start_dt is None or end_dt is None or not symbols:
        return empty
    lookback = (start_dt - timedelta(days=_REV_LOOKBACK_DAYS)).strftime("%Y%m%d")
    coll = client[data_db_name][_ANALYST_COLLECTION]
    # The internal grid starts at the lookback so the first panel day already
    # has a full 60-day window and its 60-day-ago comparison; only the panel
    # window itself is emitted.
    grid = _daily_grid(lookback, end_dt.strftime("%Y%m%d"))
    emit = grid >= start_dt
    grid_days = grid.strftime("%Y%m%d")[emit]

    frames: List[pd.DataFrame] = []
    for batch in _chunks(symbols):
        # No lower date bound: the forward-fill needs the last report *before*
        # the grid start as its seed, and a symbol whose latest report predates
        # the lookback would otherwise open its window with NaN (2026-10-17:
        # a 2024-start build differed from the 2015-start build on 748 rows
        # for exactly this reason). Full history per chunk is cheap here.
        docs = list(coll.find(
            {"ts_code": {"$in": batch},
             "report_date": {"$lte": end_dt.strftime("%Y%m%d")}},
            _ANALYST_PROJECTION,
        ))
        if not docs:
            continue
        raw = pd.DataFrame(docs)
        raw["date"] = pd.to_datetime(raw["report_date"], format="%Y%m%d", errors="coerce")
        raw["eps"] = pd.to_numeric(raw["eps"], errors="coerce")
        raw = raw.dropna(subset=["date", "eps"])
        if raw.empty:
            continue
        # One consensus per publication day: the mean across contributing orgs.
        per_day = raw.groupby(["ts_code", "date"], as_index=False)["eps"].mean()
        for code, group in per_day.groupby("ts_code", sort=False):
            rev = _revision_series(group, grid)
            if rev is None:
                continue
            frames.append(pd.DataFrame({
                "trade_date": grid_days,
                "ts_code": code,
                "analyst_rev_60d": rev[emit].astype("float64"),
            }))
    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)


def load_chip_factors(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """``cyq_conc`` / ``cyq_skew`` per (trade_date, ts_code) from cyq_perf.

    ``cyq_perf`` is already a daily per-symbol snapshot keyed by trade_date,
    so each row is the PIT value directly -- no event machinery. Queried by
    ``symbol`` (the field the collection's own index covers; ``ts_code`` is
    stored but unindexed), then renamed to the panel's ``ts_code`` join key.

    * ``cyq_conc = (cost_85pct - cost_15pct) / weight_avg`` -- how wide the
      chip distribution is around its mean cost (the "peak" width).
    * ``cyq_skew = (cost_85pct - cost_50pct) / (cost_50pct - cost_15pct)`` --
      asymmetry: trapped supply above vs profit-taking supply below.

    Coverage starts 2018-01-02 (the collection does); a symbol-day with no row
    stays NaN and is carried by the missing flag.
    """
    empty = pd.DataFrame(columns=["trade_date", "ts_code", *CHIP_FACTORS])
    start_dt = _parse_day(start_date)
    end_dt = _parse_day(end_date)
    if start_dt is None or end_dt is None or not symbols:
        return empty
    coll = client[data_db_name][_CHIP_COLLECTION]
    lo, hi = start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")

    frames: List[pd.DataFrame] = []
    for batch in _chunks(symbols):
        docs = list(coll.find(
            {"symbol": {"$in": batch}, "trade_date": {"$gte": lo, "$lte": hi}},
            _CHIP_PROJECTION,
        ))
        if not docs:
            continue
        raw = pd.DataFrame(docs)
        for col in ("cost_15pct", "cost_50pct", "cost_85pct", "weight_avg"):
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        wavg = raw["weight_avg"].where(raw["weight_avg"] > 0)
        raw["cyq_conc"] = (raw["cost_85pct"] - raw["cost_15pct"]) / wavg
        den = raw["cost_50pct"] - raw["cost_15pct"]
        raw["cyq_skew"] = ((raw["cost_85pct"] - raw["cost_50pct"])
                           / den.mask(den.abs() < 1e-6))
        raw = raw.rename(columns={"symbol": "ts_code"})
        raw["trade_date"] = raw["trade_date"].astype(str).str[:8]
        frames.append(raw[["trade_date", "ts_code", *CHIP_FACTORS]])
    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)


def _overhang_series(events: pd.DataFrame, grid: pd.DatetimeIndex) -> np.ndarray:
    """Sum of float_ratio for unlocks with ``ann_date <= t < float_date <= t+60``.

    Two cumulative series on the daily grid, evaluated at ``t``:

    * ``A(t) = sum{r : ann <= t < float}`` -- announced and not yet unlocked.
    * ``D(t) = sum{r : ann <= t < dec}``, ``dec = max(ann, float-60d)``.

    ``A - D`` keeps exactly the events whose unlock sits inside ``(t, t+60]``:
    an event with ``float > t+60`` is still inside ``D`` and cancels; an event
    unlocking within 60 days of its announcement has ``dec == ann`` and never
    enters ``D``. Events announced before the grid start are carried in as a
    scalar so the first grid day is already correct.
    """
    grid_start = grid[0]
    grid_end = grid[-1]
    n = len(grid)
    delta_a = np.zeros(n, dtype="float64")
    delta_d = np.zeros(n, dtype="float64")
    carry_a = 0.0
    carry_d = 0.0
    grid_np = grid.to_numpy()

    def add_wave(delta: np.ndarray, begin: datetime, end: datetime, ratio: float,
                 carry: float) -> float:
        """Square wave ``+ratio`` on ``[begin, end)``, clipped to the grid.

        An event ending at or before the grid start never enters; one that
        began before it is carried in and still gets its end decrement, or the
        carry would persist past the end date.
        """
        if end <= grid_start:
            return carry
        if begin <= grid_start:
            carry += ratio
        else:
            i = int(np.searchsorted(grid_np, np.datetime64(begin), side="left"))
            delta[i] += ratio
        if end <= grid_end:
            j = int(np.searchsorted(grid_np, np.datetime64(end), side="left"))
            delta[j] -= ratio
        return carry

    for ann, flt, ratio in zip(events["ann"], events["flt"], events["ratio"]):
        if ann > grid_end:
            continue
        dec = max(ann, flt - timedelta(days=_OVERHANG_WINDOW_DAYS))
        carry_a = add_wave(delta_a, ann, flt, ratio, carry_a)
        carry_d = add_wave(delta_d, ann, dec, ratio, carry_d)
    a = carry_a + np.cumsum(delta_a)
    d = carry_d + np.cumsum(delta_d)
    # Float noise can push an exact zero a few ulps negative; the quantity is
    # a sum of non-negative ratios by construction.
    return np.maximum(a - d, 0.0)


def load_unlock_overhang(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """``unlock_overhang_60d`` per (trade_date, ts_code).

    Only symbols with at least one announcement produce rows; the caller fills
    the rest with 0.0, because ``share_float`` is market-wide and
    event-complete (2005+), so an absent row means "no scheduled unlock",
    not missing data.
    """
    empty = pd.DataFrame(columns=["trade_date", "ts_code", *UNLOCK_FACTORS])
    start_dt = _parse_day(start_date)
    end_dt = _parse_day(end_date)
    if start_dt is None or end_dt is None or not symbols:
        return empty
    coll = client[data_db_name][_UNLOCK_COLLECTION]
    grid = _daily_grid(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
    grid_days = grid.strftime("%Y%m%d")

    frames: List[pd.DataFrame] = []
    for batch in _chunks(symbols):
        docs = list(coll.find(
            {"ts_code": {"$in": batch},
             "ann_date": {"$lte": end_dt.strftime("%Y%m%d")}},
            _UNLOCK_PROJECTION,
        ))
        if not docs:
            continue
        raw = pd.DataFrame(docs)
        raw["ann"] = pd.to_datetime(raw["ann_date"], format="%Y%m%d", errors="coerce")
        raw["flt"] = pd.to_datetime(raw["float_date"], format="%Y%m%d", errors="coerce")
        raw["ratio"] = pd.to_numeric(raw["float_ratio"], errors="coerce")
        raw = raw.dropna(subset=["ann", "flt", "ratio"])
        if raw.empty:
            continue
        for code, group in raw.groupby("ts_code", sort=False):
            overhang = _overhang_series(group, grid)
            frames.append(pd.DataFrame({
                "trade_date": grid_days,
                "ts_code": code,
                "unlock_overhang_60d": overhang,
            }))
    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True)
