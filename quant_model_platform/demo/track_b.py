"""Model track: live shadow scores, otherwise a PIT backfill for that signal day."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quant_model_platform.demo.backfill_pit import predict_cross_section_as_of


def shadow_score_rows(docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        symbol = str(doc.get("ts_code") or doc.get("symbol") or "").strip()
        trade_date = str(doc.get("trade_date") or "")[:8]
        if not symbol or len(trade_date) != 8:
            continue
        try:
            score = float(doc["score"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "score": score,
                "score_source": "shadow_live",
            }
        )
    return rows


def resolve_model_scores(
    live_docs: Sequence[Dict[str, Any]],
    panel: Optional[pd.DataFrame],
    params: Dict[str, Any],
    as_of: str,
    *,
    shadow_live_from: str = "",
    min_names: int = 1,
) -> List[Dict[str, Any]]:
    """Stored shadow predictions on/after ``shadow_live_from``, PIT backfill before.

    The switch is the declared date, never how many rows happen to exist: one
    stray live row must not displace a whole backfilled cross-section, and a
    live day must not fall back to a refit the live system never ran
    (docs/DEMO_HS300_20D_TOP20.md section 1.1). Without a declared boundary a
    stored prediction is authoritative for its own day, and nothing else is.
    """
    day = str(as_of)[:8]
    live = [row for row in shadow_score_rows(live_docs) if row["trade_date"] == day]
    boundary = str(shadow_live_from)[:8] if shadow_live_from else ""
    if boundary and day >= boundary:
        if len(live) < min_names:
            raise ValueError(
                f"{day} is on/after shadow_live_from={boundary} but has only {len(live)} live "
                "score(s); refusing to backfill a day the live system was supposed to cover"
            )
        return live
    if not boundary and len(live) >= min_names:
        return live
    if panel is None:
        raise ValueError(f"no shadow scores on {day} and no panel for PIT backfill")
    frame = predict_cross_section_as_of(panel, params, day)
    return frame.to_dict(orient="records")
