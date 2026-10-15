"""Cross-sectional ranking helpers for the demo."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


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
