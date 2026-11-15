"""Materialize training panels from PIT market data.

The default universe source is ``index_constituents`` + ``trade_calendar`` +
PIT ``name_change``; ``stock_scores`` remains available only as a legacy
A/B option and is not read by the default path.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.alt_factors import (
    load_analyst_revision,
    load_chip_factors,
    load_unlock_overhang,
)
from quant_model_platform.dataset.features import (
    FEATURE_REGISTRIES,
    add_missing_flags,
    feature_registry,
    registry_names,
)
from quant_model_platform.dataset.labels import attach_label_column
from quant_model_platform.dataset.manifest import build_manifest, write_parquet_snapshot
from quant_model_platform.protocol import LabelNeutralize
from quant_model_platform.dataset.raw_factors import (
    WARMUP_OBSERVATIONS,
    derive_daily_basic_factors,
    derive_price_factors,
)
from quant_model_platform.dataset.universe import (
    filter_panel_metadata,
    filter_panel_prices,
    winsorize_features,
)

log = logging.getLogger(__name__)

_SCORE_FIELD_MAP = {
    "growth_score": "growth_score",
    "cycle_score": "cycle_score",
    "value_score": "value_score",
    "quality_score": "fundamental_score",
}

#: Adjusted price columns the panel needs: ``open`` for the label, ``close``
#: for the trailing raw factors.
_PRICE_COLUMNS = ("open", "close")
_PRICE_PROJECTION = {"_id": 0, "symbol": 1, "trade_date": 1, "open": 1, "close": 1}
_ADJ_FACTOR_COLLECTION = "stock_adj_factor"
_ADJ_FACTOR_PROJECTION = {"_id": 0, "symbol": 1, "trade_date": 1, "adj_factor": 1}
_DAILY_BASIC_COLLECTION = "financial_daily_basic"
_DAILY_BASIC_PROJECTION = {
    "_id": 0,
    "ts_code": 1,
    "trade_date": 1,
    "circ_mv": 1,
    "turnover_rate": 1,
}
_INDEX_CONSTITUENTS_COLLECTION = "index_constituents"
_TRADE_CALENDAR_COLLECTION = "trade_calendar"
_NAME_CHANGE_COLLECTION = "name_change"
_PANEL_UNIVERSE_SOURCES = ("index_constituents", "stock_scores")
_PANEL_NAME_SOURCES = ("name_change", "stock_info", "index_constituents", "none")
_SYMBOL_CHUNK = 250

#: Panel join key. ``trade_date`` is the normalized ``score_date``; ``ts_code``
#: is whichever of ``symbol`` / ``ts_code`` the source collection stores.
_JOIN_KEYS = ("trade_date", "ts_code")

#: Calendar days of price history loaded before ``start_date`` so that a
#: 60-session window is already full on the panel's first day. 60 sessions is
#: roughly 87 calendar days; 130 leaves room for holidays and suspensions.
_WARMUP_CALENDAR_DAYS = 130


def _normalize_trade_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\D", "", str(value))[:8]
    return text if len(text) == 8 else None


def _extend_end_calendar(end_date: str, extra_days: int = 45) -> str:
    dt = datetime.strptime(end_date, "%Y%m%d")
    return (dt + timedelta(days=extra_days)).strftime("%Y%m%d")


def _extend_start_calendar(start_date: str, extra_days: int = 45) -> str:
    dt = datetime.strptime(start_date, "%Y%m%d")
    return (dt - timedelta(days=extra_days)).strftime("%Y%m%d")


def _flatten_score_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trade_date": _normalize_trade_date(doc.get("score_date")),
        "ts_code": str(doc.get("symbol") or doc.get("ts_code") or "").strip(),
        "name": str(doc.get("name") or ""),
    }
    if not row["trade_date"] or not row["ts_code"]:
        return {}
    scores = doc.get("scores") if isinstance(doc.get("scores"), dict) else {}
    for feat, src in _SCORE_FIELD_MAP.items():
        val = doc.get(src)
        if val is None and scores:
            val = scores.get(src.replace("_score", "")) or scores.get(feat.replace("_score", ""))
        if val is not None:
            try:
                row[feat] = float(val)
            except (TypeError, ValueError):
                row[feat] = float("nan")
    return row


def latest_score_date_for_index(
    client: MongoClient,
    *,
    mongo_db: str,
    index_code: str,
) -> Optional[str]:
    coll = client[mongo_db]["stock_scores"]
    doc = coll.find_one(
        {"index_codes": index_code},
        {"_id": 0, "score_date": 1},
        sort=[("score_date", -1)],
    )
    if not doc:
        return None
    return _normalize_trade_date(doc.get("score_date"))


def latest_price_date(
    client: MongoClient,
    *,
    data_db_name: str,
) -> Optional[str]:
    """Latest trade date available in ``volume_price``.

    This is the panel's date anchor now that the default universe source does
    not read ``stock_scores``. It is independent of the scoring cadence: at
    19:15 the day's bars are already in ``volume_price`` even when the scorer
    has not finished.
    """
    doc = client[data_db_name]["volume_price"].find_one(
        {},
        {"_id": 0, "trade_date": 1},
        sort=[("trade_date", -1)],
    )
    return _normalize_trade_date((doc or {}).get("trade_date"))


def load_stock_score_rows(
    client: MongoClient,
    *,
    mongo_db: str,
    index_code: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    start = _normalize_trade_date(start_date)
    end = _normalize_trade_date(end_date)
    if not start or not end:
        raise ValueError("start_date and end_date must be YYYYMMDD")
    coll = client[mongo_db]["stock_scores"]
    query: Dict[str, Any] = {
        "score_date": {"$gte": start, "$lte": end},
        "index_codes": index_code,
    }
    projection = {
        "_id": 0,
        "symbol": 1,
        "score_date": 1,
        "name": 1,
        "growth_score": 1,
        "cycle_score": 1,
        "value_score": 1,
        "fundamental_score": 1,
        "scores": 1,
    }
    cursor = coll.find(query, projection).sort("score_date", 1)
    try:
        cursor = cursor.batch_size(2000)
    except (TypeError, AttributeError):
        pass
    rows = [_flatten_score_doc(doc) for doc in cursor]
    return [r for r in rows if r]


def _trade_calendar_dates(
    data_db: Any,
    start_date: str,
    end_date: str,
) -> List[str]:
    """Open SSE trading days in ``[start_date, end_date]``."""
    cursor = data_db[_TRADE_CALENDAR_COLLECTION].find(
        {
            "exchange": "SSE",
            "is_open": {"$in": [1, "1"]},
            "trade_date": {"$gte": start_date, "$lte": end_date},
        },
        {"_id": 0, "trade_date": 1},
    ).sort("trade_date", 1)
    dates: List[str] = []
    for row in cursor:
        date = _normalize_trade_date(row.get("trade_date"))
        if date:
            dates.append(date)
    return sorted(set(dates))


def select_snapshot_markers(
    trade_dates: Sequence[str],
    markers: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """PIT snapshot selection: ``(trade_date, weight_trade_date, update_date)``.

    A snapshot is usable on trade date T only when both its weight date and its
    publication date are at/before T. Among usable snapshots, take the most
    recently published one, then the newest weight date. This is what keeps a
    weight snapshot published on 2026-11-02 out of the 2026-10-30 panel while
    still using it on 2026-11-02 itself.
    """
    usable_markers = sorted(
        {(str(weight), str(update)) for weight, update in markers if weight and update}
    )
    out: List[Tuple[str, str, str]] = []
    for trade_date in trade_dates:
        usable = [
            (weight, update)
            for weight, update in usable_markers
            if weight <= trade_date and update <= trade_date
        ]
        if not usable:
            continue
        weight, update = max(usable, key=lambda item: (item[1], item[0]))
        out.append((str(trade_date), weight, update))
    return out


def _load_name_change_names(
    client: MongoClient,
    data_db_name: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """PIT name-change table: ``(ts_code, start_dt, name)`` + first-name fallback."""
    rows = list(
        client[data_db_name][_NAME_CHANGE_COLLECTION].find(
            {},
            {"_id": 0, "ts_code": 1, "start_date": 1, "name": 1},
        )
    )
    if not rows:
        return pd.DataFrame(columns=["ts_code", "start_dt", "name"]), {}
    frame = pd.DataFrame(rows)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["start_dt"] = pd.to_datetime(
        frame["start_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    frame = frame.dropna(subset=["start_dt"]).sort_values(["start_dt", "ts_code"])
    first_name = (
        frame.sort_values("start_dt")
        .groupby("ts_code", sort=False)["name"]
        .first()
        .to_dict()
    )
    return frame[["ts_code", "start_dt", "name"]], first_name


def _apply_universe_names(
    out: pd.DataFrame,
    client: MongoClient,
    data_db_name: str,
    *,
    name_source: str,
) -> pd.DataFrame:
    """Attach the ``name`` column used by the panel's ST filter."""
    if name_source not in _PANEL_NAME_SOURCES:
        raise ValueError(
            f"unknown name source {name_source!r}; choose one of {list(_PANEL_NAME_SOURCES)}"
        )
    if name_source == "none":
        out = out.copy()
        out["name"] = ""
        return out
    if name_source == "stock_info":
        info = pd.DataFrame(
            list(
                client[data_db_name]["stock_info"].find(
                    {}, {"_id": 0, "symbol": 1, "name": 1}
                )
            )
        )
        mapping = (
            dict(
                zip(
                    info["symbol"].astype(str),
                    info["name"].fillna("").astype(str),
                )
            )
            if not info.empty
            else {}
        )
        out = out.copy()
        out["name"] = out["ts_code"].astype(str).map(mapping).fillna("")
        return out
    if name_source == "index_constituents":
        out = out.copy()
        if "name" in out.columns:
            out["name"] = out["name"].fillna("").astype(str)
        else:
            out["name"] = ""
        return out
    # name_change: as-of name, falling back to the symbol's first known name.
    names, first_name = _load_name_change_names(client, data_db_name)
    out = out.copy()
    out["dt"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    out = out.sort_values(["dt", "ts_code"])
    if names.empty:
        out["name"] = out["ts_code"].astype(str).map(first_name).fillna("")
    else:
        out = pd.merge_asof(
            out,
            names,
            left_on="dt",
            right_on="start_dt",
            by="ts_code",
            direction="backward",
        )
        out["name"] = (
            out["name"].fillna(out["ts_code"].astype(str).map(first_name)).fillna("")
        )
    return out.drop(columns=["dt", "start_dt"], errors="ignore")


def load_index_universe_rows(
    client: MongoClient,
    *,
    data_db_name: str,
    index_code: str,
    start_date: str,
    end_date: str,
    name_source: str = "name_change",
) -> List[Dict[str, Any]]:
    """Panel base rows from ``trade_calendar`` + ``index_constituents``.

    Returns the same shape as :func:`load_stock_score_rows`: one dict per
    ``(trade_date, ts_code)`` with a ``name`` column. Membership is PIT: the
    most recently published snapshot whose ``weight_trade_date`` and
    ``update_date`` are both at/before the trade date.

    ``name_source`` selects the name used by the ST filter in
    ``filter_panel_metadata``. The production default is ``name_change``
    (``quant_data.name_change``), which is the true as-of name; the legacy
    panel used ``stock_scores.name``, i.e. the name at scoring/backfill time.
    """
    data_db = client[data_db_name]
    trade_dates = _trade_calendar_dates(data_db, start_date, end_date)
    if not trade_dates:
        return []

    docs = list(
        data_db[_INDEX_CONSTITUENTS_COLLECTION].find(
            {"index_code": index_code, "weight_trade_date": {"$lte": end_date}},
            {
                "_id": 0,
                "weight_trade_date": 1,
                "update_date": 1,
                "symbol": 1,
                "name": 1,
            },
        )
    )
    if not docs:
        return _load_legacy_index_universe_rows(
            client,
            data_db_name=data_db_name,
            index_code=index_code,
            trade_dates=trade_dates,
            end_date=end_date,
            name_source=name_source,
        )

    snap = pd.DataFrame(docs)
    if "name" not in snap.columns:
        snap["name"] = ""
    snap["weight_trade_date"] = snap["weight_trade_date"].map(_normalize_trade_date)
    snap["update_date"] = snap["update_date"].map(_normalize_trade_date)
    snap["symbol"] = snap["symbol"].fillna("").astype(str).str.strip()
    snap = snap.dropna(subset=["weight_trade_date", "update_date"])
    snap = snap[snap["symbol"] != ""]
    if snap.empty:
        return []

    assignments = select_snapshot_markers(
        trade_dates,
        list(zip(snap["weight_trade_date"], snap["update_date"])),
    )
    if not assignments:
        return []

    member_columns = ["weight_trade_date", "update_date", "symbol"]
    if name_source == "index_constituents":
        member_columns.append("name")
    members = snap[member_columns].drop_duplicates()
    assign_frame = pd.DataFrame(
        assignments, columns=["trade_date", "weight_trade_date", "update_date"]
    )
    out = assign_frame.merge(
        members, on=["weight_trade_date", "update_date"], how="inner"
    )
    out = out.rename(columns={"symbol": "ts_code"})
    out = out.drop_duplicates(subset=["trade_date", "ts_code"]).reset_index(drop=True)
    if out.empty:
        return []
    out = _apply_universe_names(
        out, client, data_db_name, name_source=name_source
    )
    return (
        out[["trade_date", "ts_code", "name"]]
        .sort_values(["trade_date", "ts_code"])
        .to_dict("records")
    )


def _load_legacy_index_universe_rows(
    client: MongoClient,
    *,
    data_db_name: str,
    index_code: str,
    trade_dates: Sequence[str],
    end_date: str,
    name_source: str,
) -> List[Dict[str, Any]]:
    """Fallback for docs that predate ``weight_trade_date``: use update_date."""
    import bisect

    data_db = client[data_db_name]
    docs = list(
        data_db[_INDEX_CONSTITUENTS_COLLECTION].find(
            {"index_code": index_code, "update_date": {"$lte": end_date}},
            {"_id": 0, "update_date": 1, "symbol": 1, "name": 1},
        )
    )
    if not docs:
        return []
    snap = pd.DataFrame(docs)
    if "name" not in snap.columns:
        snap["name"] = ""
    snap["update_date"] = snap["update_date"].map(_normalize_trade_date)
    snap["symbol"] = snap["symbol"].fillna("").astype(str).str.strip()
    snap = snap.dropna(subset=["update_date"])
    snap = snap[snap["symbol"] != ""]
    if snap.empty:
        return []
    dates = sorted(set(snap["update_date"]))
    assignments = []
    for trade_date in trade_dates:
        idx = bisect.bisect_right(dates, trade_date) - 1
        if idx >= 0:
            assignments.append((trade_date, dates[idx]))
    if not assignments:
        return []
    member_columns = ["update_date", "symbol"]
    if name_source == "index_constituents":
        member_columns.append("name")
    members = snap[member_columns].drop_duplicates()
    assign_frame = pd.DataFrame(assignments, columns=["trade_date", "update_date"])
    out = assign_frame.merge(members, on="update_date", how="inner")
    out = out.rename(columns={"symbol": "ts_code"})
    out = out.drop_duplicates(subset=["trade_date", "ts_code"]).reset_index(drop=True)
    out = _apply_universe_names(
        out, client, data_db_name, name_source=name_source
    )
    return (
        out[["trade_date", "ts_code", "name"]]
        .sort_values(["trade_date", "ts_code"])
        .to_dict("records")
    )


def _load_adj_factors(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Adjustment factors keyed by (trade_date, ts_code) for a price window.

    Callers should pass a ``start_date`` earlier than the price window so the
    forward fill below always has a seed for each symbol's first row.
    """
    coll = client[data_db_name][_ADJ_FACTOR_COLLECTION]
    unique = sorted({s for s in symbols if s})
    chunks: List[pd.DataFrame] = []
    for i in range(0, len(unique), _SYMBOL_CHUNK):
        batch = unique[i : i + _SYMBOL_CHUNK]
        query = {"symbol": {"$in": batch}, "trade_date": {"$gte": start_date, "$lte": end_date}}
        part = pd.DataFrame(list(coll.find(query, _ADJ_FACTOR_PROJECTION)))
        if not part.empty:
            chunks.append(part)
    if not chunks:
        return pd.DataFrame(columns=["trade_date", "ts_code", "adj_factor"])
    frame = pd.concat(chunks, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
    frame["ts_code"] = frame["symbol"].astype(str)
    frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "ts_code", "adj_factor"])
    # A duplicated key would fan the price join out and double-count rows.
    return frame.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")[
        ["trade_date", "ts_code", "adj_factor"]
    ]


def _resolve_adj_factors(price_keys: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Most recent factor at or before each price date, per symbol.

    Factor rows are interleaved with the price keys before the fill, so a factor
    from before the window seeds each symbol's first rows, and a date the factor
    feed skipped inherits the last known level instead of going NaN.
    """
    seeded = pd.concat(
        [
            factors[["ts_code", "trade_date", "adj_factor"]],
            price_keys[["ts_code", "trade_date"]],
        ],
        ignore_index=True,
    ).sort_values(["ts_code", "trade_date"], kind="mergesort")
    seeded["adj_factor"] = seeded.groupby("ts_code", sort=False)["adj_factor"].ffill()
    filled = seeded.dropna(subset=["adj_factor"]).drop_duplicates(
        subset=["trade_date", "ts_code"], keep="last"
    )
    return price_keys.merge(filled, on=["trade_date", "ts_code"], how="left")["adj_factor"].to_numpy()


def _apply_adj_factor(
    client: MongoClient,
    frame: pd.DataFrame,
    *,
    data_db_name: str,
    start_date: str,
    end_date: str,
    columns: Sequence[str] = ("open",),
) -> pd.DataFrame:
    """Multiply price columns by the forward-adjustment factor.

    ``volume_price`` holds raw traded prices, so a split or a dividend inside
    a forward window reads as a huge fake return. Every label and backtest
    here is a ratio of two prices, so any single multiplicative basis gives
    the same return; hfq (``price * adj_factor``) matches what
    ``ScoreBacktester`` uses via ``adjust="hfq"``.

    ``close`` needs the same treatment as ``open``: a momentum computed off raw
    closes rediscovers the ``add8d17`` bug on the factor side.
    """
    factors = _load_adj_factors(
        client,
        data_db_name=data_db_name,
        symbols=frame["ts_code"].unique().tolist(),
        start_date=_extend_start_calendar(start_date),
        end_date=end_date,
    )
    out = frame.copy()
    out["adj_factor"] = _resolve_adj_factors(out[["ts_code", "trade_date"]], factors)
    missing = out["adj_factor"].isna()
    if missing.any():
        log.warning(
            "no adj_factor for %d/%d price rows (%.3f%%); those prices stay unadjusted",
            int(missing.sum()),
            len(out),
            100.0 * float(missing.mean()),
        )
    basis = out["adj_factor"].fillna(1.0)
    for column in columns:
        out[column] = out[column] * basis
    return out.drop(columns=["adj_factor"])


def load_price_history(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Forward-adjusted opens and closes, one row per (trade_date, ts_code)."""
    empty = pd.DataFrame(columns=["trade_date", "ts_code", *_PRICE_COLUMNS])
    start = _normalize_trade_date(start_date)
    end = _extend_end_calendar(_normalize_trade_date(end_date) or end_date)
    if not start or not end:
        return empty
    coll = client[data_db_name]["volume_price"]
    unique = sorted({s for s in symbols if s})
    chunks: List[pd.DataFrame] = []
    for i in range(0, len(unique), _SYMBOL_CHUNK):
        batch = unique[i : i + _SYMBOL_CHUNK]
        query = {"symbol": {"$in": batch}, "trade_date": {"$gte": start, "$lte": end}}
        part = pd.DataFrame(list(coll.find(query, _PRICE_PROJECTION)))
        if not part.empty:
            chunks.append(part)
    if not chunks:
        return empty
    frame = pd.concat(chunks, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
    frame["ts_code"] = frame["symbol"].astype(str)
    for column in _PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "ts_code", "open"])
    frame = _apply_adj_factor(
        client,
        frame,
        data_db_name=data_db_name,
        start_date=start,
        end_date=end,
        columns=_PRICE_COLUMNS,
    )
    return frame[["trade_date", "ts_code", *_PRICE_COLUMNS]]


def load_daily_basic(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Daily total/circulating market cap and turnover, per (trade_date, ts_code).

    ``quant_data.financial_daily_basic`` is indexed on ``ts_code``, not
    ``symbol`` -- querying it by symbol full-scans ~14M rows, so the symbol set
    is matched against ``ts_code`` directly. Both sides store the suffixed form
    (``600000.SH``), the same as ``volume_price.symbol``.
    """
    empty = pd.DataFrame(columns=["trade_date", "ts_code", "circ_mv", "turnover_rate"])
    start = _normalize_trade_date(start_date)
    end = _extend_end_calendar(_normalize_trade_date(end_date) or end_date)
    if not start or not end:
        return empty
    coll = client[data_db_name][_DAILY_BASIC_COLLECTION]
    unique = sorted({s for s in symbols if s})
    chunks: List[pd.DataFrame] = []
    for i in range(0, len(unique), _SYMBOL_CHUNK):
        batch = unique[i : i + _SYMBOL_CHUNK]
        query = {"ts_code": {"$in": batch}, "trade_date": {"$gte": start, "$lte": end}}
        part = pd.DataFrame(list(coll.find(query, _DAILY_BASIC_PROJECTION)))
        if not part.empty:
            chunks.append(part)
    if not chunks:
        return empty
    frame = pd.concat(chunks, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    # A duplicated key would fan the merge out and double-count rows. The score
    # loader skips this and gets away with it; a new source should not copy that.
    return frame.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")


def _attach_raw_factors(
    client: MongoClient,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Left-join the price and daily-basic factors onto the panel.

    Each join is checked for fan-out: a duplicated key on either side would
    multiply panel rows, and every count downstream would still look plausible.
    """
    basic = load_daily_basic(
        client,
        data_db_name=data_db_name,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    derived = (derive_price_factors(prices), derive_daily_basic_factors(basic))

    out = panel
    for factors in derived:
        if factors.empty:
            continue
        # The registry pre-creates every declared name as NaN; merging onto a
        # column that already exists would produce momentum_20d_x/_y instead.
        overlap = [c for c in factors.columns if c in out.columns and c not in _JOIN_KEYS]
        if overlap:
            out = out.drop(columns=overlap)
        before = len(out)
        out = out.merge(factors, on=list(_JOIN_KEYS), how="left")
        if len(out) != before:
            raise RuntimeError(
                f"raw factor join fanned the panel out: {before} -> {len(out)} rows"
            )
    return out


def _attach_alt_factors(
    client: MongoClient,
    panel: pd.DataFrame,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Left-join the alternative-channel factors (B6e-2) onto the panel.

    Same join discipline as ``_attach_raw_factors``: drop overlapping columns
    first, then refuse to fan the panel out.

    ``analyst_rev_60d`` (FY1) and ``analyst_rev_fy2_60d`` (FY2) stay NaN where
    a symbol has no analyst coverage; the registry's missing flag carries
    that. ``unlock_overhang_60d`` is filled
    with 0.0 after the join: ``share_float`` is market-wide and
    event-complete back to 2005, so an absent row means "no scheduled
    unlock", which is a fact about the stock rather than missing data.
    """
    out = panel
    for factors in (
        load_analyst_revision(
            client,
            data_db_name=data_db_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        ),
        load_unlock_overhang(
            client,
            data_db_name=data_db_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        ),
        load_chip_factors(
            client,
            data_db_name=data_db_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        ),
    ):
        if factors.empty:
            continue
        overlap = [c for c in factors.columns if c in out.columns and c not in _JOIN_KEYS]
        if overlap:
            out = out.drop(columns=overlap)
        before = len(out)
        out = out.merge(factors, on=list(_JOIN_KEYS), how="left")
        if len(out) != before:
            raise RuntimeError(
                f"alt factor join fanned the panel out: {before} -> {len(out)} rows"
            )
    if "unlock_overhang_60d" in out.columns:
        out["unlock_overhang_60d"] = out["unlock_overhang_60d"].fillna(0.0)
    return out


def build_production_panel(
    *,
    index_code: str,
    start_date: str,
    end_date: str,
    parquet_dir: Path,
    config: Optional[PlatformConfig] = None,
    client: Optional[MongoClient] = None,
    use_canonical_snapshot: bool = False,
    label_neutralize: LabelNeutralize = "demean",
    label_transform: str = "rank",
    universe_filter: bool = True,
    min_list_days: int = 250,
    feature_winsor_quantile: float = 0.01,
    label_horizon_days: int = 20,
    universe_source: str = "index_constituents",
    name_source: str = "name_change",
    feature_set: str = "raw",
) -> Tuple[pd.DataFrame, Any]:
    """Build PIT score panel + forward labels; write parquet under parquet_dir.

    ``universe_source`` defaults to ``index_constituents`` + ``trade_calendar``
    (no ``stock_scores`` read); pass ``"stock_scores"`` for the legacy A/B path.
    ``name_source`` selects the ST-filter name (``name_change`` is the PIT
    default). ``feature_set`` selects ``"raw"`` (no rule-score columns) or
    ``"legacy"`` (the original registry).

    ``label_neutralize`` defaults to the production demean; pass
    ``"size_neutral"`` to residualize the label against ``log_market_cap``
    (the B6e-4 style check). Non-default modes are recorded in
    ``source_revision`` so a variant artifact cannot read as production.
    """
    if universe_source not in _PANEL_UNIVERSE_SOURCES:
        raise ValueError(
            f"unknown universe source {universe_source!r}; choose one of {list(_PANEL_UNIVERSE_SOURCES)}"
        )
    if name_source not in _PANEL_NAME_SOURCES:
        raise ValueError(
            f"unknown name source {name_source!r}; choose one of {list(_PANEL_NAME_SOURCES)}"
        )
    if feature_set not in FEATURE_REGISTRIES:
        raise ValueError(
            f"unknown feature set {feature_set!r}; choose one of {sorted(FEATURE_REGISTRIES)}"
        )
    cfg = config or PlatformConfig.from_env()
    own_client = client is None
    mongo = client or MongoClient(cfg.mongo_uri, tz_aware=True)
    try:
        if universe_source == "stock_scores":
            score_rows = load_stock_score_rows(
                mongo,
                mongo_db=cfg.mongo_db,
                index_code=index_code,
                start_date=start_date,
                end_date=end_date,
            )
            if not score_rows:
                raise RuntimeError(
                    f"no stock_scores rows for index={index_code} between {start_date} and {end_date} "
                    f"(db={cfg.mongo_db})"
                )
        else:
            score_rows = load_index_universe_rows(
                mongo,
                data_db_name=cfg.data_db_name,
                index_code=index_code,
                start_date=start_date,
                end_date=end_date,
                name_source=name_source,
            )
            if not score_rows:
                raise RuntimeError(
                    f"no index_constituents/trade_calendar rows for index={index_code} "
                    f"between {start_date} and {end_date} (db={cfg.data_db_name})"
                )
        panel = pd.DataFrame(score_rows)
        if universe_filter:
            info = pd.DataFrame(
                list(
                    mongo[cfg.data_db_name]["stock_info"].find(
                        {"ts_code": {"$in": panel["ts_code"].unique().tolist()}},
                        {"_id": 0, "ts_code": 1, "list_date": 1, "delist_date": 1},
                    )
                )
            )
            panel = filter_panel_metadata(
                panel, stock_info=info, min_list_days=min_list_days
            )
        feature_names = registry_names(feature_registry(feature_set))
        for name in feature_names:
            if name not in panel.columns:
                panel[name] = float("nan")
        symbols = panel["ts_code"].unique().tolist()

        # Prices load from before the panel starts, so a 60-session window is
        # already full on the first signal date. The extra leading rows match no
        # panel row and drop out at the join; a forward-shifted label is
        # unaffected by history that precedes it.
        warmup_start = _extend_start_calendar(start_date, extra_days=_WARMUP_CALENDAR_DAYS)
        prices = load_price_history(
            mongo,
            data_db_name=cfg.data_db_name,
            symbols=symbols,
            start_date=warmup_start,
            end_date=end_date,
        )
        if prices.empty:
            raise RuntimeError(f"no volume_price rows in {cfg.data_db_name} for panel symbols")
        if universe_filter:
            panel = filter_panel_prices(
                panel, price_keys=prices.loc[:, ["trade_date", "ts_code"]]
            )
        panel = _attach_raw_factors(
            mongo,
            panel,
            prices,
            data_db_name=cfg.data_db_name,
            symbols=symbols,
            start_date=warmup_start,
            end_date=end_date,
        )
        # After the raw factors, not before: ``size_neutral`` residualizes the
        # label against ``log_market_cap``, which is a present-but-empty
        # placeholder until _attach_raw_factors fills it. Attached earlier,
        # the label silently degrades to demean on every date -- 2026-10-18's
        # first size_neutral build did exactly that, and the guard's warning
        # is what caught it.
        label_col = f"label_fwd_{int(label_horizon_days)}d"
        panel = attach_label_column(
            panel,
            prices,
            neutralize=label_neutralize,
            label_transform=label_transform,
            horizon=int(label_horizon_days),
            label_col=label_col,
        )
        # Alt-channel loaders carry their own lookback, so they only need the
        # panel window itself (unlike the price factors above).
        panel = _attach_alt_factors(
            mongo,
            panel,
            data_db_name=cfg.data_db_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        panel = winsorize_features(
            panel, feature_names, quantile=feature_winsor_quantile
        )
        # After the factors, not before: flags computed against the NaN
        # placeholders would call every raw factor missing on every row.
        panel = add_missing_flags(panel, feature_names)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        if use_canonical_snapshot:
            path = parquet_dir / f"{index_code}_current.parquet"
        else:
            path = parquet_dir / f"{index_code}_production_{start_date}_{end_date}.parquet"
        write_parquet_snapshot(panel, path)
        # The suffix separates snapshots that share a window but not a feature
        # set. Without it, provenance for the raw-factor panel would read
        # identically to the scores-only one it replaced.
        if universe_source == "index_constituents":
            source_prefix = (
                f"index_constituents:{cfg.data_db_name}:{index_code}:{start_date}-{end_date}"
                f":{name_source}:{feature_set}"
            )
            pit_policy = "index_constituents.weight_trade_date"
        else:
            source_prefix = (
                f"stock_scores:{cfg.mongo_db}:{index_code}:{start_date}-{end_date}"
                f":rawfactors-v1:{feature_set}"
            )
            pit_policy = "stock_scores.index_codes"
        source_revision = (
            source_prefix
            + (f":label-{label_neutralize}" if label_neutralize != "demean" else "")
            + (f":label-transform-{label_transform}" if label_transform != "none" else "")
            + (f":universe-filter-{min_list_days}" if universe_filter else "")
            + (f":winsor-{feature_winsor_quantile}" if feature_winsor_quantile else "")
            + (f":label{int(label_horizon_days)}d" if int(label_horizon_days) != 20 else "")
        )
        manifest = build_manifest(
            panel,
            index_code=index_code,
            start_date=str(start_date)[:8],
            end_date=str(end_date)[:8],
            parquet_path=path,
            source_revision=source_revision,
            pit_policy=pit_policy,
            use_canonical_snapshot=use_canonical_snapshot,
        )
        log.info(
            "production panel rows=%d dates=%d symbols=%d path=%s",
            len(panel),
            panel["trade_date"].nunique(),
            panel["ts_code"].nunique(),
            path,
        )
        return panel, manifest
    finally:
        if own_client:
            mongo.close()


__all__ = [
    "build_production_panel",
    "latest_price_date",
    "latest_score_date_for_index",
    "load_daily_basic",
    "load_index_universe_rows",
    "load_price_history",
    "load_stock_score_rows",
    "select_snapshot_markers",
]
