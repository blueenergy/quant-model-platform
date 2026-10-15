"""Equal-weight 20-day TOP-N equity for the hs300 demo.

Gross period return matches ``ScoreBacktester.calculate_topk_summary`` top-K
mean of ``forward_return_{horizon}d`` (next-open entry and exit). That class
does not compound a rebalance NAV or apply costs, so net return, turnover, and
max drawdown live here. One-way cost is 10 bps; a name swap pays both sides.
"""

from __future__ import annotations

import math
from statistics import pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence

from quant_model_platform.demo.ranking import top_n_symbols

ONE_WAY_COST = 0.001


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
    return turnover * ONE_WAY_COST * sides


def simulate_track(
    signal_dates: Sequence[str],
    scores_by_date: Mapping[str, Sequence[Dict[str, Any]]],
    forward_returns: Mapping[str, Mapping[str, float]],
    *,
    top_n: int,
    track: str,
) -> List[Dict[str, Any]]:
    previous: List[str] = []
    rows: List[Dict[str, Any]] = []
    for day in signal_dates:
        holdings = top_n_symbols(scores_by_date.get(day, []), score_key="score", n=top_n)
        member_returns = []
        day_fwd = forward_returns.get(day, {})
        for symbol in holdings:
            value = day_fwd.get(symbol)
            if value is None:
                continue
            member_returns.append(float(value))
        gross = sum(member_returns) / len(member_returns) if member_returns else None
        opening = not previous
        turnover = one_way_turnover(previous, holdings)
        cost = _period_cost(turnover, opening=opening) if holdings else 0.0
        net = None if gross is None else gross - cost
        rows.append(
            {
                "trade_date": day,
                "track": track,
                "holdings": holdings,
                "n_priced": len(member_returns),
                "gross_return": gross,
                "one_way_turnover": turnover,
                "cost": cost,
                "net_return": net,
            }
        )
        previous = holdings
    return rows


def equity_stats(period_returns: Sequence[Optional[float]], *, hold_days: int = 20) -> Dict[str, Optional[float]]:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    realized: List[float] = []
    curve: List[float] = []
    for value in period_returns:
        if value is None:
            curve.append(equity)
            continue
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        realized.append(float(value))
        curve.append(equity)
    vol = None
    if len(realized) > 1 and hold_days > 0:
        vol = pstdev(realized) * math.sqrt(252.0 / hold_days)
    return {
        "total_return": equity - 1.0,
        "max_drawdown": max_drawdown,
        "ann_vol": vol,
        "equity_curve": curve,
    }


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
) -> Dict[str, Any]:
    track_a = simulate_track(signal_dates, scores_a, forward_returns, top_n=top_n, track="baseline")
    track_b = simulate_track(signal_dates, scores_b, forward_returns, top_n=top_n, track="model")
    stats_a = equity_stats([row["net_return"] for row in track_a], hold_days=hold_days)
    stats_b = equity_stats([row["net_return"] for row in track_b], hold_days=hold_days)
    overlaps = [
        overlap_ratio(a["holdings"], b["holdings"])
        for a, b in zip(track_a, track_b)
    ]
    known = [value for value in overlaps if value is not None]
    return {
        "track_a": {**_public_stats(stats_a, track_a), "score_column": "composite_growth_cycle_score"},
        "track_b": _public_stats(stats_b, track_b),
        "comparison": {
            "excess_return_b_minus_a": (stats_b["total_return"] or 0.0) - (stats_a["total_return"] or 0.0),
            "avg_overlap_ratio": (sum(known) / len(known)) if known else None,
        },
        "rebalances": track_a + track_b,
        "equity_curve": [
            {"trade_date": day, "baseline": stats_a["equity_curve"][i], "model": stats_b["equity_curve"][i]}
            for i, day in enumerate(signal_dates)
        ],
    }


def _public_stats(stats: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    turns = [row["one_way_turnover"] for row in rows]
    return {
        "total_return": stats["total_return"],
        "max_drawdown": stats["max_drawdown"],
        "ann_vol": stats["ann_vol"],
        "avg_turnover": (sum(turns) / len(turns)) if turns else None,
        "periods": len(rows),
    }
