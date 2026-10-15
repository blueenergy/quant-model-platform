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
from quant_model_platform.dataset.features import add_missing_flags, registry_names
from quant_model_platform.dataset.labels import attach_label_column
from quant_model_platform.dataset.manifest import build_manifest, write_parquet_snapshot

log = logging.getLogger(__name__)

_SCORE_FIELD_MAP = {
    "growth_score": "growth_score",
    "cycle_score": "cycle_score",
    "value_score": "value_score",
    "quality_score": "fundamental_score",
}

_PRICE_PROJECTION = {"_id": 0, "symbol": 1, "trade_date": 1, "open": 1}
_SYMBOL_CHUNK = 250


def _normalize_trade_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\D", "", str(value))[:8]
    return text if len(text) == 8 else None


def _extend_end_calendar(end_date: str, extra_days: int = 45) -> str:
    dt = datetime.strptime(end_date, "%Y%m%d")
    return (dt + timedelta(days=extra_days)).strftime("%Y%m%d")


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


def load_open_prices(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start = _normalize_trade_date(start_date)
    end = _extend_end_calendar(_normalize_trade_date(end_date) or end_date)
    if not start or not end:
        return pd.DataFrame(columns=["trade_date", "ts_code", "open"])
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
        return pd.DataFrame(columns=["trade_date", "ts_code", "open"])
    frame = pd.concat(chunks, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].map(_normalize_trade_date)
    frame["ts_code"] = frame["symbol"].astype(str)
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    return frame.dropna(subset=["trade_date", "ts_code", "open"])[["trade_date", "ts_code", "open"]]


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
        panel = add_missing_flags(panel, feature_names)
        symbols = panel["ts_code"].unique().tolist()
        prices = load_open_prices(
            mongo,
            data_db_name=cfg.data_db_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        if prices.empty:
            raise RuntimeError(f"no volume_price rows in {cfg.data_db_name} for panel symbols")
        panel = attach_label_column(panel, prices)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        if use_canonical_snapshot:
            path = parquet_dir / f"{index_code}_current.parquet"
        else:
            path = parquet_dir / f"{index_code}_production_{start_date}_{end_date}.parquet"
        write_parquet_snapshot(panel, path)
        source_revision = f"stock_scores:{cfg.mongo_db}:{index_code}:{start_date}-{end_date}"
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
    "load_open_prices",
    "load_stock_score_rows",
]
