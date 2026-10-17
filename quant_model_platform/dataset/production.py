"""Materialize training panels from production Mongo (stock_scores + volume_price)."""

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
from quant_model_platform.dataset.features import add_missing_flags, registry_names
from quant_model_platform.dataset.labels import attach_label_column
from quant_model_platform.dataset.manifest import build_manifest, write_parquet_snapshot
from quant_model_platform.dataset.raw_factors import (
    WARMUP_OBSERVATIONS,
    derive_daily_basic_factors,
    derive_price_factors,
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

    ``analyst_rev_60d`` stays NaN where a symbol has no analyst coverage; the
    registry's missing flag carries that. ``unlock_overhang_60d`` is filled
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
) -> Tuple[pd.DataFrame, Any]:
    """Build PIT score panel + forward labels; write parquet under parquet_dir."""
    cfg = config or PlatformConfig.from_env()
    own_client = client is None
    mongo = client or MongoClient(cfg.mongo_uri, tz_aware=True)
    try:
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
        panel = pd.DataFrame(score_rows)
        feature_names = registry_names()
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
        panel = attach_label_column(panel, prices)
        panel = _attach_raw_factors(
            mongo,
            panel,
            prices,
            data_db_name=cfg.data_db_name,
            symbols=symbols,
            start_date=warmup_start,
            end_date=end_date,
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
        source_revision = (
            f"stock_scores:{cfg.mongo_db}:{index_code}:{start_date}-{end_date}:rawfactors-v1"
        )
        manifest = build_manifest(
            panel,
            index_code=index_code,
            start_date=str(start_date)[:8],
            end_date=str(end_date)[:8],
            parquet_path=path,
            source_revision=source_revision,
            pit_policy="stock_scores.index_codes",
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
    "latest_score_date_for_index",
    "load_daily_basic",
    "load_price_history",
    "load_stock_score_rows",
]
