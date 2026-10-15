"""Model track: live shadow scores, otherwise a PIT backfill for that signal day."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from quant_model_platform.db import COLLECTION_PREDICTIONS
from quant_model_platform.demo.backfill_pit import predict_cross_section_as_of
from quant_model_platform.protocol import MAX_PURGE_DAYS

# Overlap check (docs/DEMO_HS300_20D_TOP20.md section 4): the backfill stands in
# for the live scorer on every date before shadow existed, so the two must rank
# the same names the same way. Below these the long backfilled curve is only a
# plumbing test, not evidence.
MIN_OVERLAP_NAMES = 5
MIN_OVERLAP_SPEARMAN = 0.8


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


def load_shadow_predictions(
    collections: Mapping[str, Any],
    *,
    model_id: str = "",
    index_code: str = "",
    start: str = "",
    end: str = "",
) -> List[Dict[str, Any]]:
    """Live shadow rows out of ``quant_ml.model_predictions``, as the loop wrote them.

    Filtering by ``model_id`` matters: the collection is shared, and a row from
    another profile would otherwise be read as this model's opinion.
    """
    query: Dict[str, Any] = {"shadow": True}
    if model_id:
        query["model_id"] = str(model_id)
    if index_code:
        query["index_code"] = str(index_code)
    window: Dict[str, str] = {}
    if start:
        window["$gte"] = str(start)[:8]
    if end:
        window["$lte"] = str(end)[:8]
    if window:
        query["trade_date"] = window
    rows = list(collections[COLLECTION_PREDICTIONS].find(query, {"_id": 0}))
    return shadow_score_rows(rows)


def scores_by_day(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """``{trade_date: {symbol: score}}``, last write winning per (day, symbol).

    Deliberately source-blind: the overlap check runs it over both the live rows
    and a backfill frame, and must not relabel either one.
    """
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        symbol = str(row.get("ts_code") or row.get("symbol") or "").strip()
        day = str(row.get("trade_date") or "")[:8]
        if not symbol or len(day) != 8:
            continue
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if score != score:  # NaN
            continue
        out.setdefault(day, {})[symbol] = score
    return out


def cross_section_rank_correlation(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    *,
    min_common: int = MIN_OVERLAP_NAMES,
) -> Dict[str, Any]:
    """Spearman between two score sets on the names they both cover, per day.

    Rankings, not levels: the two paths fit different training windows, so their
    scores sit on different scales -- what has to agree is the order.
    """
    left_by_day, right_by_day = scores_by_day(left), scores_by_day(right)
    days: List[Dict[str, Any]] = []
    for day in sorted(set(left_by_day) & set(right_by_day)):
        common = sorted(set(left_by_day[day]) & set(right_by_day[day]))
        entry: Dict[str, Any] = {
            "trade_date": day,
            "n_left": len(left_by_day[day]),
            "n_right": len(right_by_day[day]),
            "n_common": len(common),
            "spearman": None,
        }
        if len(common) >= min_common:
            a = pd.Series([left_by_day[day][s] for s in common], index=common, dtype="float64")
            b = pd.Series([right_by_day[day][s] for s in common], index=common, dtype="float64")
            if a.nunique() > 1 and b.nunique() > 1:
                value = a.corr(b, method="spearman")
                entry["spearman"] = None if value != value else round(float(value), 4)
        days.append(entry)
    scored = [d["spearman"] for d in days if d["spearman"] is not None]
    return {
        "shared_days": len(set(left_by_day) & set(right_by_day)),
        "days": days,
        "n_scored": len(scored),
        "mean_spearman": round(sum(scored) / len(scored), 4) if scored else None,
        "min_spearman": round(min(scored), 4) if scored else None,
        "threshold": MIN_OVERLAP_SPEARMAN,
        "min_common_names": min_common,
        "passes": bool(scored) and min(scored) >= MIN_OVERLAP_SPEARMAN,
    }


def resolve_model_scores(
    live_docs: Sequence[Dict[str, Any]],
    panel: Optional[pd.DataFrame],
    params: Dict[str, Any],
    as_of: str,
    *,
    shadow_live_from: str = "",
    purge: int = MAX_PURGE_DAYS,
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
    frame = predict_cross_section_as_of(panel, params, day, purge=purge)
    return frame.to_dict(orient="records")
