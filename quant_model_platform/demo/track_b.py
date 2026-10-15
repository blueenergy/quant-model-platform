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
    min_names: int = 1,
) -> List[Dict[str, Any]]:
    """Prefer stored shadow predictions. Backfill only when that day is missing."""
    live = shadow_score_rows(live_docs)
    if len(live) >= min_names:
        return live
    if panel is None:
        raise ValueError(f"no shadow scores on {as_of} and no panel for PIT backfill")
    frame = predict_cross_section_as_of(panel, params, as_of)
    return frame.to_dict(orient="records")
