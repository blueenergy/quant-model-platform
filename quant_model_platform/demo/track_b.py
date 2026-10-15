"""Model track: live shadow scores, otherwise a PIT backfill for that signal day."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from quant_model_platform.db import COLLECTION_PREDICTIONS, COLLECTION_RESULTS
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


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def fold_summary(folds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One side of a walk-forward run, reduced to what a spread can be read off.

    Reduces exactly the way ``evaluation.gates.passes_promotion_gates`` does --
    drop folds with no ``mean_ic``, then average -- so this block's mean is the
    number the gate judged, not a neighbouring one. The dispersion matters more
    than the mean here: at 20 sessions a fold, a single fold's IC is noise, and
    the spread is what says so.
    """
    ics = [float(fold["mean_ic"]) for fold in folds if fold.get("mean_ic") is not None]
    turns = [float(fold["avg_turnover"]) for fold in folds if fold.get("avg_turnover") is not None]
    return {
        "n_folds": len(ics),
        "mean_ic": _mean(ics),
        "sd_ic": statistics.stdev(ics) if len(ics) > 1 else None,
        "min_ic": min(ics) if ics else None,
        "max_ic": max(ics) if ics else None,
        "share_positive": (sum(1 for value in ics if value > 0) / len(ics)) if ics else None,
        "avg_turnover": _mean(turns),
        "n_days_summed": sum(int(fold.get("n_days") or 0) for fold in folds),
    }


def training_evidence(collections: Mapping[str, Any], *, job_id: str = "") -> Optional[Dict[str, Any]]:
    """What the source job's own training run concluded, quoted rather than redone.

    The demo asks whether the model is worth anything, and the job that produced
    it wrote that answer down. Copying the verdict keeps the demo from quietly
    running a model the platform's own gate rejected, which is exactly what
    happened the first time round.

    Degrades to ``None`` -- no job id, no such job, no result -- like
    ``model_overlap`` does, and never raises: a missing audit trail must not be
    able to stop the measurement it is meant to annotate.
    """
    results = collections.get(COLLECTION_RESULTS)
    if not job_id or results is None:
        return None
    result = results.find_one({"job_id": str(job_id)}, {"_id": 0}, sort=[("created_at", -1)])
    if not result:
        return None
    folds = list(result.get("fold_reports") or [])
    model = fold_summary([fold.get("model") or {} for fold in folds])
    baseline = fold_summary([fold.get("baseline") or {} for fold in folds])
    difference = (
        model["mean_ic"] - baseline["mean_ic"]
        if model["mean_ic"] is not None and baseline["mean_ic"] is not None
        else None
    )
    return {
        "job_id": result.get("job_id"),
        "result_id": result.get("result_id"),
        "created_at": _iso(result.get("created_at")),
        "model_kind": result.get("model_kind"),
        "code_version": result.get("code_version"),
        "snapshot_id": result.get("snapshot_id"),
        "artifact_hash": result.get("artifact_hash"),
        # The verdict as recorded, never recomputed: re-deriving it here would
        # let the gate and the demo disagree about the same run.
        "promotion": result.get("promotion"),
        "permutation_p": result.get("permutation_p"),
        "baseline_ic": result.get("baseline_ic"),
        # Named ``folds``, not ``fold_reports``: that key in Mongo holds the 272
        # raw per-fold documents, and this is a reduction of them.
        "folds": {"model": model, "baseline": baseline, "model_minus_baseline": difference},
        "comparable_to_primary": False,
        "note": (
            "These are the source job's own walk-forward numbers, on its own folds, pool and window; they "
            "are not the demo's primary evidence and will not equal primary.tracks.baseline.rank_ic.mean. "
            "n_days_summed counts daily, overlapping labels, so it is not a sample size either. One "
            "asymmetry worth knowing before reading the margin: the gate compares the model's mean of "
            "per-fold means against baseline_ic.mean_ic, which is pooled over every day, so "
            "folds.baseline.mean_ic and baseline_ic.mean_ic are two different averages of nearly "
            "the same data. Compare like with like where it changes a verdict."
        ),
    }


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
