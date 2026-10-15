"""Baseline track: growth/cycle composite on one signal date."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_growth_cycle_weights(spec: str) -> Tuple[float, float]:
    """Parse ``60:40`` into growth and cycle weights that sum to 1."""
    left, _, right = str(spec).partition(":")
    growth = float(left)
    cycle = float(right) if right else 0.0
    if growth > 1 or cycle > 1:
        growth, cycle = growth / 100.0, cycle / 100.0
    total = growth + cycle
    if total <= 0:
        raise ValueError(f"invalid growth/cycle weights: {spec}")
    return growth / total, cycle / total


def growth_cycle_composite(growth: Optional[float], cycle: Optional[float], *, spec: str = "60:40") -> Optional[float]:
    """Same available-weight blend as portfolio research 60:40."""
    growth_w, cycle_w = parse_growth_cycle_weights(spec)
    available = 0.0
    total = 0.0
    if growth is not None:
        available += growth_w
        total += growth_w * float(growth)
    if cycle is not None:
        available += cycle_w
        total += cycle_w * float(cycle)
    if available <= 0:
        return None
    return total / available


def baseline_score_rows(
    docs: Sequence[Dict[str, Any]],
    *,
    score_column: str = "composite_growth_cycle_score",
    weights: str = "60:40",
) -> List[Dict[str, Any]]:
    """One cross-section. Prefer a stored column; otherwise blend growth and cycle."""
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        symbol = str(doc.get("symbol") or doc.get("ts_code") or "").strip()
        trade_date = str(doc.get("score_date") or doc.get("trade_date") or "")[:8]
        if not symbol or len(trade_date) != 8:
            continue
        stored = doc.get(score_column)
        if stored is None and isinstance(doc.get("scores"), dict):
            stored = doc["scores"].get(score_column)
        try:
            score = float(stored) if stored is not None else None
        except (TypeError, ValueError):
            score = None
        if score is None:
            score = growth_cycle_composite(doc.get("growth_score"), doc.get("cycle_score"), spec=weights)
        if score is None:
            continue
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "score": score,
                "score_source": "baseline_growth_cycle",
            }
        )
    return rows
