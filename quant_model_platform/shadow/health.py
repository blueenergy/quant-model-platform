"""Refresh registry health from ``quant_ml.model_predictions``.

The shadow loop owns daily prediction writes; after each batch it refreshes
the model registry's health block so the factory page can show stale active
models without issuing N+1 reads on every page load.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List

from quant_model_platform.db import (
    COLLECTION_MODELS,
    COLLECTION_PREDICTION_SETS,
    COLLECTION_PREDICTIONS,
)
from quant_model_platform.timeutil import utcnow

log = logging.getLogger(__name__)

_RECENT_DATES = 20
_BASELINE_DATES = 60
_WINDOW_DATES = _RECENT_DATES + _BASELINE_DATES


def _date_means(preds, model_id: str) -> List[Dict[str, Any]]:
    return list(
        preds.aggregate(
            [
                {"$match": {"model_id": model_id}},
                {
                    "$group": {
                        "_id": "$trade_date",
                        "n": {"$sum": 1},
                        "mean_score": {"$avg": "$score"},
                    }
                },
                {"$sort": {"_id": -1}},
                {"$limit": _WINDOW_DATES},
            ]
        )
    )


def _drift(date_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    means = [float(row.get("mean_score")) for row in date_rows if row.get("mean_score") is not None]
    if not means:
        return {}
    recent = means[:_RECENT_DATES]
    baseline = means[_RECENT_DATES : _RECENT_DATES + _BASELINE_DATES]
    if not recent or not baseline:
        return {"score_mean_recent": sum(recent) / len(recent) if recent else None}
    recent_mean = sum(recent) / len(recent)
    baseline_mean = sum(baseline) / len(baseline)
    variance = sum((value - baseline_mean) ** 2 for value in baseline) / len(baseline)
    baseline_std = math.sqrt(variance)
    return {
        "score_mean_recent": recent_mean,
        "score_mean_baseline": baseline_mean,
        "score_drift": recent_mean - baseline_mean,
        "score_drift_z": (recent_mean - baseline_mean) / baseline_std if baseline_std else None,
    }


def _stale_days(latest_market_date: str | None, last_prediction_date: str | None) -> int | None:
    if not latest_market_date or not last_prediction_date:
        return None
    try:
        return (
            datetime.strptime(str(latest_market_date), "%Y%m%d")
            - datetime.strptime(str(last_prediction_date), "%Y%m%d")
        ).days
    except ValueError:
        return None


def refresh_prediction_set_registry(collections: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute every prediction set's stats and derive model summaries.

    The set registry is the identity source; ``models.prediction_sets`` is a
    denormalized summary that the factory UI reads. Shadow runs append rows
    daily, so both have to be refreshed from ``model_predictions`` rather than
    hand-maintained.
    """
    sets = collections.get(COLLECTION_PREDICTION_SETS)
    preds = collections.get(COLLECTION_PREDICTIONS)
    models = collections.get(COLLECTION_MODELS)
    if sets is None or preds is None or models is None:
        return {"sets": 0, "models": 0, "overlaps": 0}

    summaries: Dict[str, List[Dict[str, Any]]] = {}
    refreshed = 0
    overlaps: List[str] = []
    now = utcnow()
    for doc in sets.find({}):
        set_id = str(doc.get("prediction_set_id") or doc.get("_id") or "")
        model_id = str(doc.get("model_id") or "")
        if not set_id or not model_id:
            continue
        dates = sorted(
            str(value)
            for value in (preds.distinct("trade_date", {"prediction_set_id": set_id}) or [])
            if value
        )
        prediction_count = int(
            preds.count_documents({"prediction_set_id": set_id}) or 0
        )
        segment_rows = list(
            preds.aggregate(
                [
                    {"$match": {"prediction_set_id": set_id}},
                    {
                        "$group": {
                            "_id": {"source": "$source", "window": "$window"},
                            "first_trade_date": {"$min": "$trade_date"},
                            "last_trade_date": {"$max": "$trade_date"},
                            "prediction_count": {"$sum": 1},
                            "dates": {"$addToSet": "$trade_date"},
                        }
                    },
                    {"$sort": {"first_trade_date": 1}},
                ]
            )
        )
        segments = [
            {
                "source": (row.get("_id") or {}).get("source"),
                "window": (row.get("_id") or {}).get("window"),
                "first_trade_date": row.get("first_trade_date"),
                "last_trade_date": row.get("last_trade_date"),
                "date_count": len(row.get("dates") or []),
                "prediction_count": int(row.get("prediction_count") or 0),
            }
            for row in segment_rows
        ]
        first_trade_date = dates[0] if dates else None
        last_trade_date = dates[-1] if dates else None
        sets.update_one(
            {"_id": doc.get("_id")},
            {
                "$set": {
                    "prediction_set_id": set_id,
                    "first_trade_date": first_trade_date,
                    "last_trade_date": last_trade_date,
                    "date_count": len(dates),
                    "prediction_count": prediction_count,
                    "source_segments": segments,
                    "updated_at": now,
                }
            },
        )
        summaries.setdefault(model_id, []).append(
            {
                "prediction_set_id": set_id,
                "kind": doc.get("kind"),
                "status": doc.get("status") or "active",
                "source": segments[0].get("source") if segments else doc.get("source"),
                "window": doc.get("window") or (segments[0].get("window") if segments else None),
                "first_trade_date": first_trade_date,
                "last_trade_date": last_trade_date,
                "date_count": len(dates),
                "prediction_count": prediction_count,
                "source_segments": segments,
            }
        )
        refreshed += 1

    for model_id, rows in summaries.items():
        rows.sort(
            key=lambda row: (
                row["status"] != "active",
                str(row.get("kind") or ""),
                row.get("first_trade_date") or "",
            )
        )
        models.update_one(
            {"$or": [{"_id": model_id}, {"model_id": model_id}]},
            {"$set": {"prediction_sets": rows}},
        )
        for kind in ("research", "shadow"):
            active = [
                row
                for row in rows
                if row.get("kind") == kind and row.get("status") == "active"
            ]
            active.sort(key=lambda row: row.get("first_trade_date") or "")
            for previous, current in zip(active, active[1:]):
                if str(current.get("first_trade_date") or "") <= str(
                    previous.get("last_trade_date") or ""
                ):
                    overlaps.append(
                        f"{model_id}/{kind}: "
                        f"{previous.get('prediction_set_id')} overlaps "
                        f"{current.get('prediction_set_id')}"
                    )
    if overlaps:
        log.warning("overlapping prediction sets: %s", "; ".join(overlaps))
    missing_rows = int(
        preds.count_documents({"prediction_set_id": {"$exists": False}}) or 0
    )
    if missing_rows:
        log.warning("model_predictions rows missing prediction_set_id: %s", missing_rows)
    empty_sets = sum(
        1
        for rows in summaries.values()
        for row in rows
        if int(row.get("prediction_count") or 0) == 0
    )
    return {
        "sets": refreshed,
        "models": len(summaries),
        "overlaps": len(overlaps),
        "missing_rows": missing_rows,
        "empty_sets": empty_sets,
    }


def refresh_model_health(
    collections: Dict[str, Any],
    *,
    latest_market_date: str | None = None,
) -> Dict[str, Any]:
    """Recompute health for every registry model; return active-stale count.

    ``latest_market_date`` should come from ``volume_price`` (the actual
    trading calendar). Falling back to the max prediction date would hide a
    total shadow outage: every model would look "current" relative to the
    stale prediction clock.
    """
    models = collections.get(COLLECTION_MODELS)
    preds = collections.get(COLLECTION_PREDICTIONS)
    if models is None or preds is None:
        return {"checked": 0, "active_stale": 0}
    prediction_set_summary: Dict[str, Any] = {}
    try:
        prediction_set_summary = refresh_prediction_set_registry(collections)
    except Exception:
        log.exception("prediction set registry refresh failed")

    if latest_market_date is None:
        latest_doc = preds.find_one({}, {"trade_date": 1}, sort=[("trade_date", -1)])
        latest_market_date = str((latest_doc or {}).get("trade_date") or "") or None
    latest_market_date = str(latest_market_date or "") or None
    docs = list(models.find({}, {"model_id": 1, "state": 1, "health": 1}))
    active_stale = 0
    now = utcnow()
    for doc in docs:
        model_id = str(doc.get("model_id") or doc.get("_id") or "")
        if not model_id:
            continue
        dates = sorted(str(value) for value in (preds.distinct("trade_date", {"model_id": model_id}) or []) if value)
        date_rows = _date_means(preds, model_id)
        prediction_count = int(preds.count_documents({"model_id": model_id}) or 0)
        state = str(doc.get("state") or "research")
        active = state in ("live", "shadow")
        last_date = dates[-1] if dates else None
        first_date = dates[0] if dates else None
        stale_days = _stale_days(latest_market_date, last_date)
        is_current = bool(last_date and latest_market_date and last_date == latest_market_date)
        if not dates:
            status = "no_predictions"
        elif active:
            status = "current" if is_current else "stale"
        else:
            status = "inactive"
        if active and not is_current:
            active_stale += 1
        drift = _drift(date_rows)
        health = {
            "latest_market_date": latest_market_date,
            "last_prediction_date": last_date,
            "first_prediction_date": first_date,
            "prediction_count": prediction_count,
            "date_count": len(dates),
            "avg_names_per_date": round(prediction_count / len(dates), 1) if dates else None,
            "stale_days": stale_days,
            "is_current": is_current,
            "active": active,
            "status": status,
            **drift,
            "checked_at": now,
        }
        models.update_one({"_id": doc["_id"]}, {"$set": {"health": health}})
    return {
        "checked": len(docs),
        "active_stale": active_stale,
        "latest_market_date": latest_market_date,
        "checked_at": now,
        "prediction_sets": prediction_set_summary,
    }
