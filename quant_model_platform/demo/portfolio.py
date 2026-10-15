"""Equal-weight 20-day TOP-N equity for the hs300 demo.

Gross period return matches ``ScoreBacktester.calculate_topk_summary`` top-K
mean of ``forward_return_{horizon}d`` (next-open entry and exit). That class
compounds no NAV and charges nothing, so net return, turnover and max drawdown
live here.

Costs follow the frozen contract (docs/DEMO_HS300_20D_TOP20.md section 1.2):
7 bps of fees round trip plus 10 bps of slippage a side, i.e. 13.5 bps a side
and 27 bps to swap a fully changed book. Drawdown is measured on the rebalance
grid, not daily, and the period count is a year's worth at best -- treat the
NAV as communication, never as evidence.
"""

from __future__ import annotations

import math
from statistics import pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence

from quant_model_platform.demo.ranking import random_control_scores, top_n_symbols

#: Fees 7 bps round trip + 10 bps slippage a side.
COST_PER_SIDE = 0.00135
ROUND_TRIP_COST_RATE = 2.0 * COST_PER_SIDE

#: Seed for the random control track: reproducible, arbitrary.
CONTROL_SEED = 20261015

#: |t| of track R's mean period return we are willing to call "zero". R shares
#: the pool, the dates and the cost model with the real tracks, so a large t
#: means the harness leaks or mis-charges, not that randomness paid.
CONTROL_T_MAX = 3.0

#: Label written when track A is scored by the self-computed blend rather than
#: a stored column -- ``stock_scores`` has no composite column in production.
SELF_COMPUTED_SCORE = "growth_cycle_60:40"


def one_way_turnover(previous: Sequence[str], current: Sequence[str]) -> float:
    """Share of the new book that was not held. A fresh book turns over fully."""
    if not current:
        return 0.0
    if not previous:
        return 1.0
    kept = len(set(previous) & set(current))
    return 1.0 - kept / len(current)


def _period_cost(turnover: float, *, opening: bool) -> float:
    sides = 1.0 if opening else 2.0
    return turnover * COST_PER_SIDE * sides


def simulate_track(
    signal_dates: Sequence[str],
    scores_by_date: Mapping[str, Sequence[Dict[str, Any]]],
    forward_returns: Mapping[str, Mapping[str, float]],
    *,
    top_n: int,
    track: str,
) -> List[Dict[str, Any]]:
    """Rebalance chain for one track. A day with no usable cross-section holds.

    A period can come up empty two different ways, and they must not be
    confused: with **no cross-section** the previous book is carried forward --
    no trade, so no turnover and no cost -- while a **forward window that has
    not happened yet** is recorded but priced as ``none``, keeping it out of the
    NAV instead of compounding it as a flat zero.
    """
    previous: List[str] = []
    rows: List[Dict[str, Any]] = []
    for day in signal_dates:
        cross_section = list(scores_by_date.get(day) or [])
        selected = top_n_symbols(cross_section, score_key="score", n=top_n)
        carried = not selected
        holdings = list(previous) if carried else selected
        day_fwd = forward_returns.get(day) or {}
        priced = [float(day_fwd[symbol]) for symbol in holdings if day_fwd.get(symbol) is not None]
        pricing = "none" if not priced else ("full" if len(priced) == len(holdings) else "partial")
        turnover = 0.0 if carried else one_way_turnover(previous, holdings)
        cost = _period_cost(turnover, opening=not previous) if holdings else 0.0
        gross = (sum(priced) / len(priced)) if priced else None
        rows.append(
            {
                "trade_date": day,
                "track": track,
                "holdings": holdings,
                "cross_section": "carried" if carried else "selected",
                "n_scored": len(cross_section),
                "n_holdings": len(holdings),
                "n_priced": len(priced),
                "pricing": pricing,
                "gross_return": gross,
                "one_way_turnover": turnover,
                "cost": cost,
                "net_return": None if gross is None else gross - cost,
            }
        )
        previous = holdings
    return rows


def equity_stats(rows: Sequence[Mapping[str, Any]], *, hold_days: int = 20) -> Dict[str, Any]:
    """Compound ``net_return`` over the rows that realize one.

    Rows priced ``none`` hold the curve flat and stay out of every statistic --
    a period whose forward window has not happened is not a zero-return period.
    Gross moments come along because the pool-relative excess is a gross
    quantity: costs are a deterministic drag every track pays.
    """
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    realized: List[float] = []
    gross: List[float] = []
    curve: List[float] = []
    for row in rows:
        value = row.get("net_return")
        if value is None:
            curve.append(equity)
            continue
        if row.get("gross_return") is not None:
            gross.append(float(row["gross_return"]))
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        realized.append(float(value))
        curve.append(equity)
    return {
        "total_return": equity - 1.0,
        "max_drawdown": max_drawdown,
        "ann_vol": _annualized_vol(realized, hold_days=hold_days),
        "mean_period_return": (sum(realized) / len(realized)) if realized else None,
        "t_stat": _t_stat(realized),
        "gross_mean_period_return": (sum(gross) / len(gross)) if gross else None,
        "gross_t_stat": _t_stat(gross),
        "equity_curve": curve,
    }


def _annualized_vol(realized: Sequence[float], *, hold_days: int) -> Optional[float]:
    if len(realized) < 2 or hold_days <= 0:
        return None
    return pstdev(realized) * math.sqrt(252.0 / hold_days)


def _t_stat(realized: Sequence[float]) -> Optional[float]:
    """Mean over its own standard error; None when there is nothing to test."""
    if len(realized) < 2:
        return None
    spread = pstdev(realized)
    if spread <= 0:
        return None
    return (sum(realized) / len(realized)) / (spread / math.sqrt(len(realized)))


def overlap_ratio(left: Sequence[str], right: Sequence[str]) -> Optional[float]:
    if not left or not right:
        return None
    return len(set(left) & set(right)) / max(len(left), len(right))


def compare_tracks(
    signal_dates: Sequence[str],
    scores_a: Mapping[str, Sequence[Dict[str, Any]]],
    scores_b: Mapping[str, Sequence[Dict[str, Any]]],
    forward_returns: Mapping[str, Mapping[str, float]],
    *,
    top_n: int = 20,
    hold_days: int = 20,
    score_column_a: str = "",
    control_seed: int = CONTROL_SEED,
) -> Dict[str, Any]:
    """Track A (rules), track B (model) and track R (random control).

    R ranks a random permutation of the same pool on the same dates through the
    same cost model, so its pool-relative excess has zero expectation: a
    nonzero one is a harness defect, not a result.
    """
    track_a = simulate_track(signal_dates, scores_a, forward_returns, top_n=top_n, track="baseline")
    track_b = simulate_track(signal_dates, scores_b, forward_returns, top_n=top_n, track="model")
    pool = scores_a or scores_b
    track_r = simulate_track(
        signal_dates,
        random_control_scores(pool, seed=control_seed),
        forward_returns,
        top_n=top_n,
        track="control_random",
    )
    stats_a = equity_stats(track_a, hold_days=hold_days)
    stats_b = equity_stats(track_b, hold_days=hold_days)
    stats_r = equity_stats(track_r, hold_days=hold_days)
    public_a = _public_stats(stats_a, track_a)
    public_b = _public_stats(stats_b, track_b)
    public_r = _public_stats(stats_r, track_r)
    overlaps = [overlap_ratio(a["holdings"], b["holdings"]) for a, b in zip(track_a, track_b)]
    random_overlaps = [overlap_ratio(a["holdings"], r["holdings"]) for a, r in zip(track_a, track_r)]
    known = [value for value in overlaps if value is not None]
    known_random = [value for value in random_overlaps if value is not None]
    return {
        "track_a": {**public_a, "score_column": score_column_a or SELF_COMPUTED_SCORE},
        "track_b": public_b,
        "track_r": public_r,
        "control_r": _control_check(stats_r, public_r, control_seed=control_seed),
        "comparison": {
            "excess_return_b_minus_a": _difference(public_b, public_a, "total_return"),
            "avg_overlap_ratio": (sum(known) / len(known)) if known else None,
            "avg_overlap_baseline_vs_random": (sum(known_random) / len(known_random)) if known_random else None,
        },
        "rebalances": track_a + track_b + track_r,
        "equity_curve": [
            {
                "trade_date": day,
                "baseline": stats_a["equity_curve"][i],
                "model": stats_b["equity_curve"][i],
                "random": stats_r["equity_curve"][i],
                "baseline_pricing": track_a[i]["pricing"],
                "model_pricing": track_b[i]["pricing"],
                "random_pricing": track_r[i]["pricing"],
            }
            for i, day in enumerate(signal_dates)
        ],
    }


def _control_check(stats: Dict[str, Any], public: Dict[str, Any], *, control_seed: int) -> Dict[str, Any]:
    t_stat = stats["gross_t_stat"]
    return {
        "seed": control_seed,
        "t_stat": t_stat,
        "t_max": CONTROL_T_MAX,
        "mean_period_return": stats["gross_mean_period_return"],
        "net_total_return": stats["total_return"],
        "periods": public["periods"],
        "passed": t_stat is not None and abs(t_stat) <= CONTROL_T_MAX,
        "note": (
            "Track R draws random scores from the same pool on the same dates, so its pool-relative "
            "excess (gross) has zero expectation; costs are a deterministic drag it pays like any "
            "other track and are excluded from the null. At ~12 periods a year this catches gross "
            "harness defects (look-ahead, mis-charged turnover), not subtle bias."
        ),
    }


def _difference(left: Dict[str, Any], right: Dict[str, Any], field: str) -> Optional[float]:
    """Only meaningful when both tracks realized at least one period."""
    if not left["periods"] or not right["periods"]:
        return None
    return (left[field] or 0.0) - (right[field] or 0.0)


def _public_stats(stats: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    traded = [row for row in rows if row["pricing"] != "none"]
    turns = [row["one_way_turnover"] for row in traded]
    return {
        "total_return": stats["total_return"],
        "max_drawdown": stats["max_drawdown"],
        "ann_vol": stats["ann_vol"],
        "mean_period_return": stats["mean_period_return"],
        "t_stat": stats["t_stat"],
        "gross_mean_period_return": stats["gross_mean_period_return"],
        "gross_t_stat": stats["gross_t_stat"],
        "avg_turnover": (sum(turns) / len(turns)) if turns else None,
        "periods": len(traded),
        "rebalance_days": len(rows),
        "dropped_periods": len(rows) - len(traded),
        "partial_periods": sum(1 for row in rows if row["pricing"] == "partial"),
        "carried_periods": sum(1 for row in rows if row["cross_section"] == "carried"),
    }
