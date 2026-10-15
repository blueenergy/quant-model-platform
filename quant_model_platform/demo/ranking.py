"""Cross-sectional ranking helpers for the demo."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence


def top_n_symbols(rows: Sequence[Dict[str, Any]], *, score_key: str, n: int) -> List[str]:
    """Return up to n symbols with the highest finite scores (ties: symbol ascending)."""
    scored = []
    for row in rows:
        symbol = str(row.get("ts_code") or row.get("symbol") or "").strip()
        if not symbol:
            continue
        raw = row.get(score_key)
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if score != score:  # NaN
            continue
        scored.append((score, symbol))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [symbol for _, symbol in scored[:n]]


def random_control_scores(
    scores_by_date: Mapping[str, Sequence[Dict[str, Any]]],
    *,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Track R: uniform random scores over the pool the real tracks see.

    Same symbols, same dates, no signal. Its pool-relative excess is therefore a
    zero-effect reference for the harness -- see ``portfolio.compare_tracks``.
    Scores are drawn per (seed, date), so the track is reproducible and does not
    depend on dict ordering.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for day, rows in scores_by_date.items():
        key = str(day)[:8]
        generator = random.Random(f"{seed}:{key}")
        drawn: List[Dict[str, Any]] = []
        for row in rows or []:
            symbol = str(row.get("ts_code") or row.get("symbol") or "").strip()
            if not symbol:
                continue
            drawn.append(
                {
                    "ts_code": symbol,
                    "trade_date": key,
                    "score": generator.random(),
                    "score_source": "control_random",
                }
            )
        out[day] = drawn
    return out
