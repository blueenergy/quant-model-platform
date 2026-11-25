"""Panel quality gates for production snapshots.

The production panel is rebuilt from source every day. This module checks the
result before it is published, so a silent data gap cannot become a training
feature. The checks are split in two:

* hard errors -- duplicate keys, a factor gap inside the panel window, missing
  calendar dates, or an empty label column. These fail the build.
* warnings -- a known historical artifact, a sudden coverage drop, or a
  feature that is mostly null. These are reported but do not fail the build.

The 2015 boundary is deliberate: ``stock_adj_factor`` starts on 2015-01-05, so
the 130-day warm-up before a 2015-01-30 panel has no factor. Those warm-up
prices are used by the 60-session momentum lookback and would otherwise rank
the early-2015 cross-section by adjustment factor. The build keeps that
history for now (research and production share it); the report marks it as a
known artifact instead of pretending it is a fresh regression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

#: How many trading sessions a 60-session feature needs after the first
#: adjustment factor. Used only to describe the known 2015 artifact.
BOUNDARY_LOOKBACK_SESSIONS = 60

#: A daily row count below this share of the trailing median is a coverage
#: warning. The universe changes slowly; a sudden drop is a feed problem.
COVERAGE_DROP_RATIO = 0.7
COVERAGE_BASELINE_DAYS = 20

#: Features above this null share are reported. The raw registry has five
#: features that are nearly complete; legacy builds may legitimately be sparse.
FEATURE_NULL_WARN_RATIO = 0.5


def load_first_factor_dates(
    client,
    *,
    data_db_name: str,
    symbols: Sequence[str],
) -> Dict[str, str]:
    """First date with a non-null adjustment factor, per symbol."""
    unique = sorted({str(s) for s in symbols if s})
    if not unique:
        return {}
    rows = client[data_db_name]["stock_adj_factor"].aggregate(
        [
            {"$match": {"symbol": {"$in": unique}, "adj_factor": {"$ne": None}}},
            {"$group": {"_id": "$symbol", "first_date": {"$min": "$trade_date"}}},
        ]
    )
    return {str(row["_id"]): str(row["first_date"]) for row in rows if row.get("first_date")}


def load_expected_trade_dates(
    client,
    *,
    data_db_name: str,
    start_date: str,
    end_date: str,
) -> List[str]:
    """Open SSE sessions between two dates, ascending."""
    rows = client[data_db_name]["trade_calendar"].find(
        {
            "exchange": "SSE",
            "is_open": {"$in": [1, "1", True]},
            "trade_date": {"$gte": start_date, "$lte": end_date},
        },
        {"_id": 0, "trade_date": 1},
    ).sort("trade_date", 1)
    return [str(row["trade_date"]) for row in rows]


def _json_number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _coverage_anomalies(counts: pd.Series) -> List[Dict[str, Any]]:
    prior = counts.shift(1).rolling(COVERAGE_BASELINE_DAYS, min_periods=10).median()
    flagged = counts[(prior.notna()) & (counts < COVERAGE_DROP_RATIO * prior)]
    return [
        {
            "trade_date": str(day),
            "rows": int(count),
            "trailing_median": float(prior.loc[day]),
        }
        for day, count in flagged.items()
    ]


def _boundary_artifact(
    panel: pd.DataFrame,
    *,
    first_factor_dates: Mapping[str, str],
    expected_dates: Sequence[str],
    start_date: str,
) -> Dict[str, Any]:
    """Describe the early-2015 lookback that crosses the factor start."""
    if not first_factor_dates:
        return {"rows": 0, "dates": [], "momentum_60d_median": None}
    first = min(first_factor_dates.values())
    window = [d for d in sorted(expected_dates) if d >= first][:BOUNDARY_LOOKBACK_SESSIONS]
    if not window:
        return {"rows": 0, "dates": [], "momentum_60d_median": None}
    dates = [d for d in window if d >= start_date]
    if not dates:
        return {"rows": 0, "dates": [], "momentum_60d_median": None}
    subset = panel[panel["trade_date"].isin(dates)]
    median = None
    if "momentum_60d" in subset.columns and not subset.empty:
        median = _json_number(subset["momentum_60d"].median())
    return {
        "rows": int(len(subset)),
        "dates": dates,
        "momentum_60d_median": median,
    }


def check_panel_quality(
    panel: pd.DataFrame,
    *,
    index_code: str,
    start_date: str,
    end_date: str,
    first_factor_dates: Mapping[str, str],
    expected_dates: Sequence[str],
    feature_names: Sequence[str],
    label_col: str,
) -> Dict[str, Any]:
    """Return a JSON-serialisable report; ``ok`` is False on hard errors."""
    errors: List[str] = []
    warnings: List[str] = []

    # The production panel is ~2.7M rows; copying every column doubles the
    # build's peak memory. The gate only needs the keys and the columns it
    # reports on.
    needed = ["trade_date", "ts_code"]
    for name in [label_col, "momentum_60d", *feature_names]:
        if name in panel.columns and name not in needed:
            needed.append(name)
    frame = panel[needed].copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)

    duplicate_keys = int(frame.duplicated(subset=["trade_date", "ts_code"]).sum())
    if duplicate_keys:
        errors.append(f"duplicate (trade_date, ts_code) keys: {duplicate_keys}")

    panel_dates = sorted(frame["trade_date"].unique().tolist())
    expected = [str(d) for d in expected_dates]
    # ``expected`` may start before the panel (the factor-boundary window
    # needs those sessions); continuity is judged only inside the panel window.
    expected_in_window = [d for d in expected if d >= start_date]
    missing_dates = (
        sorted(set(expected_in_window) - set(panel_dates)) if expected_in_window else []
    )
    extra_dates = sorted(set(panel_dates) - set(expected_in_window)) if expected_in_window else []
    if missing_dates:
        errors.append(f"missing trading dates: {len(missing_dates)} ({missing_dates[:5]})")
    if extra_dates:
        warnings.append(f"dates outside the calendar: {len(extra_dates)} ({extra_dates[:5]})")

    first = frame["ts_code"].map(first_factor_dates)
    missing_factor = first.isna() | (frame["trade_date"] < first.astype(str))
    missing_factor_rows = int(missing_factor.sum())
    if missing_factor_rows:
        examples = (
            frame.loc[missing_factor, ["trade_date", "ts_code"]]
            .head(10)
            .to_dict("records")
        )
        errors.append(
            f"rows without an adjustment factor at or before their date: "
            f"{missing_factor_rows} (examples={examples})"
        )

    if label_col not in frame.columns:
        errors.append(f"label column {label_col!r} is missing")
        label_null_ratio = None
    else:
        label_null_ratio = _json_number(frame[label_col].isna().mean())
        if label_null_ratio is not None and label_null_ratio >= 1.0:
            errors.append(f"label column {label_col!r} is entirely null")
        elif label_null_ratio is not None and label_null_ratio > 0.5:
            warnings.append(f"label column {label_col!r} is {label_null_ratio:.1%} null")

    feature_null_ratios: Dict[str, Optional[float]] = {}
    for name in feature_names:
        if name not in frame.columns:
            feature_null_ratios[name] = None
            warnings.append(f"declared feature {name!r} is missing from the panel")
            continue
        ratio = _json_number(frame[name].isna().mean())
        feature_null_ratios[name] = ratio
        if ratio is not None and ratio > FEATURE_NULL_WARN_RATIO:
            warnings.append(f"feature {name!r} is {ratio:.1%} null")

    counts = frame.groupby("trade_date").size().sort_index()
    anomalies = _coverage_anomalies(counts)
    if anomalies:
        warnings.append(
            f"daily coverage dropped below {COVERAGE_DROP_RATIO:.0%} of the "
            f"trailing median on {len(anomalies)} dates"
        )

    boundary = _boundary_artifact(
        frame,
        first_factor_dates=first_factor_dates,
        expected_dates=expected,
        start_date=start_date,
    )
    if boundary["rows"] and boundary["momentum_60d_median"] is not None:
        if boundary["momentum_60d_median"] > 1.0:
            warnings.append(
                "known 2015 adjustment-factor boundary: "
                f"{boundary['rows']} rows, momentum_60d median "
                f"{boundary['momentum_60d_median']:.3f}"
            )

    return {
        "ok": not errors,
        "index_code": index_code,
        "start_date": start_date,
        "end_date": end_date,
        "row_count": int(len(frame)),
        "date_count": int(frame["trade_date"].nunique()),
        "symbol_count": int(frame["ts_code"].nunique()),
        "duplicate_keys": duplicate_keys,
        "missing_factor_rows": missing_factor_rows,
        "missing_expected_dates": missing_dates,
        "extra_dates": extra_dates,
        "daily_count": {
            "min": int(counts.min()) if len(counts) else 0,
            "median": int(counts.median()) if len(counts) else 0,
            "max": int(counts.max()) if len(counts) else 0,
            "anomalies": anomalies,
        },
        "label_col": label_col,
        "label_null_ratio": label_null_ratio,
        "feature_null_ratios": feature_null_ratios,
        "known_artifact": {
            "name": "adj_factor_boundary_2015",
            **boundary,
        },
        "errors": errors,
        "warnings": warnings,
    }
