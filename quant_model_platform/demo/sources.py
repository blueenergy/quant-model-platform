"""Local Mongo sources for the demo: member scores, hfq price rows, forward returns.

The forward-return code here is deliberately **independent of
``dataset.labels``** -- the demo is the reference the platform's label gets
reconciled against, so sharing an implementation would make that check
circular. It follows ``ScoreBacktester`` (stock-scoring-system), which is what
the production score validation actually runs:

* prices are hfq (``open * adj_factor``), rows with a non-positive close are
  dropped before any index arithmetic, and a missing open falls back to close;
* entry is the open of the first session **strictly after** the signal date;
* a one-price limit-up entry bar makes the whole (date, symbol) unusable;
* exit is the open ``horizon`` sessions after entry, delayed up to
  ``EXIT_LIMIT_DELAY_DAYS`` sessions while the exit bar is a one-price
  limit-down;
* when no exit exists yet the symbol has **no** forward return -- callers must
  read that as "unpriced", never as a zero.

One deliberate divergence, measured on the hs300 2y window: an entry bar that
is a one-price limit-up is refused **even when the signal date itself has no
bar** for that symbol. ``ScoreBacktester`` cannot apply the rule there, because
its price window starts at the first bar on or after the score date -- a
suspended symbol's resumption bar arrives with no previous close and the check
is skipped -- so it will price a post-suspension one-price limit-up that cannot
be bought at the open. That was 2 of 8850 (date, symbol) pairs, both resumption
bars; :func:`forward_return` refuses them on purpose.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from pymongo import MongoClient

from quant_model_platform.demo.track_a import baseline_score_rows

EXIT_LIMIT_DELAY_DAYS = 5
MIN_NAMES_PER_DATE = 5
LIMIT_MOVE = 0.095
ONE_PRICE_TOLERANCE = 0.001
ADJ_FACTOR_SEED_DAYS = 45
_SYMBOL_CHUNK = 250
_PRICE_PROJECTION = {"_id": 0, "symbol": 1, "trade_date": 1, "open": 1, "high": 1, "low": 1, "close": 1}
_FACTOR_PROJECTION = {"_id": 0, "symbol": 1, "trade_date": 1, "adj_factor": 1}


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _shift_calendar(date: str, days: int) -> str:
    dt = datetime.strptime(str(date)[:8], "%Y%m%d")
    return (dt + timedelta(days=days)).strftime("%Y%m%d")


def _find(collection, query: Dict[str, Any], projection: Dict[str, Any]) -> List[Dict[str, Any]]:
    cursor = collection.find(query, projection)
    try:
        cursor = cursor.batch_size(2000)
    except (TypeError, AttributeError):
        pass
    return list(cursor)


def _chunked(collection, symbols: Sequence[str], *, start: str, end: str, projection: Dict[str, Any]):
    unique = sorted({s for s in symbols if s})
    for i in range(0, len(unique), _SYMBOL_CHUNK):
        batch = unique[i : i + _SYMBOL_CHUNK]
        query = {"symbol": {"$in": batch}, "trade_date": {"$gte": start, "$lte": end}}
        yield from _find(collection, query, projection)


def attach_hfq_factor(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Multiply price columns by the last adjustment factor at or before each date.

    Its own implementation, not the platform's: a shared as-of join would hide
    exactly the class of defect this demo exists to catch.
    """
    by_symbol = {
        str(symbol): group.set_index("trade_date")["adj_factor"].sort_index()
        for symbol, group in factors.groupby("symbol", sort=False)
    }
    frames: List[pd.DataFrame] = []
    for symbol, group in prices.groupby("symbol", sort=False):
        ordered = group.sort_values("trade_date")
        series = by_symbol.get(str(symbol))
        if series is None or series.empty:
            ordered = ordered.assign(adj_factor=1.0)
        else:
            series = series[~series.index.duplicated(keep="last")]
            ordered = ordered.assign(
                adj_factor=series.reindex(ordered["trade_date"].to_numpy(), method="ffill").to_numpy()
            )
        frames.append(ordered)
    out = pd.concat(frames, ignore_index=True) if frames else prices.assign(adj_factor=1.0)
    for column in ("open", "high", "low", "close"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * out["adj_factor"]
    return out.drop(columns=["adj_factor"])


def clean_price_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ScoreBacktester's price cleaning: needs a positive close, open falls back to it."""
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        trade_date = str(row.get("trade_date") or "")[:8]
        close = _to_float(row.get("close"))
        if len(trade_date) != 8 or close is None or close <= 0:
            continue
        open_price = _to_float(row.get("open"))
        cleaned.append(
            {
                "trade_date": trade_date,
                "open": open_price if open_price and open_price > 0 else close,
                "close": close,
                "high": _to_float(row.get("high")),
                "low": _to_float(row.get("low")),
            }
        )
    cleaned.sort(key=lambda item: item["trade_date"])
    return cleaned


def _execution_price(row: Dict[str, Any]) -> Optional[float]:
    price = _to_float(row.get("open"))
    return price if price and price > 0 else _to_float(row.get("close"))


def _one_price_limit(row: Dict[str, Any], previous: Optional[Dict[str, Any]], *, up: bool) -> bool:
    if not previous:
        return False
    open_price = _to_float(row.get("open"))
    high, low = _to_float(row.get("high")), _to_float(row.get("low"))
    close, previous_close = _to_float(row.get("close")), _to_float(previous.get("close"))
    if None in (open_price, high, low, close, previous_close) or not previous_close:
        return False
    one_price = max(open_price, high, low, close) - min(open_price, high, low, close) <= max(
        0.01, close * ONE_PRICE_TOLERANCE
    )
    move = close / previous_close - 1.0
    return bool(one_price and (move >= LIMIT_MOVE if up else move <= -LIMIT_MOVE))


def _find_exit_index(rows: Sequence[Dict[str, Any]], target_idx: int) -> Optional[int]:
    last_idx = min(len(rows) - 1, target_idx + EXIT_LIMIT_DELAY_DAYS)
    for idx in range(target_idx, last_idx + 1):
        previous = rows[idx - 1] if idx > 0 else None
        if not _one_price_limit(rows[idx], previous, up=False):
            return idx
    return None


def forward_return(rows: Sequence[Dict[str, Any]], signal_date: str, horizon: int) -> Optional[float]:
    """Next-open entry, ``horizon``-session exit; None when the rule refuses the trade."""
    day = str(signal_date)[:8]
    entry_idx = next((i for i, row in enumerate(rows) if row["trade_date"] > day), None)
    if entry_idx is None:
        return None
    previous = rows[entry_idx - 1] if entry_idx > 0 else None
    if _one_price_limit(rows[entry_idx], previous, up=True):
        return None
    entry_price = _execution_price(rows[entry_idx])
    if entry_price is None or entry_price <= 0:
        return None
    target_idx = entry_idx + int(horizon)
    if len(rows) <= target_idx:
        return None
    exit_idx = _find_exit_index(rows, target_idx)
    if exit_idx is None:
        return None
    exit_price = _execution_price(rows[exit_idx])
    if exit_price is None or exit_price <= 0:
        return None
    return exit_price / entry_price - 1.0


def forward_returns_by_signal_date(
    prices: Dict[str, Sequence[Dict[str, Any]]],
    signal_dates: Sequence[str],
    *,
    horizon: int,
) -> Dict[str, Dict[str, float]]:
    """``{signal_date: {symbol: return}}``; a symbol with no exit is simply absent."""
    out: Dict[str, Dict[str, float]] = {}
    for day in signal_dates:
        key = str(day)[:8]
        day_returns: Dict[str, float] = {}
        for symbol, rows in prices.items():
            value = forward_return(rows, key, horizon)
            if value is not None:
                day_returns[symbol] = value
        out[key] = day_returns
    return out


def load_price_histories(
    client: MongoClient,
    *,
    data_db_name: str,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Cleaned hfq price rows per symbol, ready for the entry/exit rules."""
    start = str(start_date)[:8]
    end = str(end_date)[:8]
    price_rows = list(
        _chunked(
            client[data_db_name]["volume_price"],
            symbols,
            start=start,
            end=end,
            projection=_PRICE_PROJECTION,
        )
    )
    if not price_rows:
        return {}
    prices = pd.DataFrame(price_rows)
    prices["trade_date"] = prices["trade_date"].astype(str).str[:8]
    factors = pd.DataFrame(
        list(
            _chunked(
                client[data_db_name]["stock_adj_factor"],
                symbols,
                start=_shift_calendar(start, -ADJ_FACTOR_SEED_DAYS),
                end=end,
                projection=_FACTOR_PROJECTION,
            )
        )
    )
    if not factors.empty:
        factors["trade_date"] = factors["trade_date"].astype(str).str[:8]
        factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
        factors = factors.dropna(subset=["adj_factor"])
        prices = attach_hfq_factor(prices, factors)
    histories: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, group in prices.groupby("symbol", sort=False):
        histories[str(symbol)] = clean_price_rows(group.to_dict(orient="records"))
    return histories


def load_member_score_docs(
    client: MongoClient,
    *,
    mongo_db: str,
    index_code: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """``stock_scores`` rows for names that were in ``index_code`` **as of that day**."""
    query = {
        "index_codes": index_code,
        "score_date": {"$gte": str(start_date)[:8], "$lte": str(end_date)[:8]},
    }
    projection = {
        "_id": 0,
        "symbol": 1,
        "score_date": 1,
        "growth_score": 1,
        "cycle_score": 1,
        "scores": 1,
    }
    return _find(client[mongo_db]["stock_scores"], query, projection)


def score_cross_sections(
    docs: Sequence[Dict[str, Any]],
    *,
    score_column: str = "",
    weights: str = "60:40",
    min_names: int = MIN_NAMES_PER_DATE,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-date cross-sections, dropping a day the validator would not rank.

    ``ScoreBacktester`` skips a cross-section below ``min_names_per_date`` or
    with a single distinct score; a day it will not rank is a day the demo must
    not open a fresh book on either (it carries the previous one).
    """
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for row in baseline_score_rows(docs, score_column=score_column, weights=weights):
        by_date.setdefault(row["trade_date"], []).append(row)
    return {
        day: rows
        for day, rows in by_date.items()
        if len(rows) >= min_names and len({row["score"] for row in rows}) >= 2
    }


def build_bundle(
    client: MongoClient,
    *,
    docs: Sequence[Dict[str, Any]],
    signal_dates: Sequence[str],
    data_db_name: str,
    start_date: str,
    end_date: str,
    horizon: int,
    score_column: str = "",
    weights: str = "60:40",
) -> Dict[str, Any]:
    """Bundle payload for an already-chosen window: calendar, track A's scores, returns.

    ``docs`` come from :func:`load_member_score_docs`; the caller needs them
    first because the trading calendar -- and therefore the window and the
    signal dates -- is read off the same rows.
    """
    symbols = sorted({str(doc["symbol"]) for doc in docs if doc.get("symbol")})
    prices = load_price_histories(
        client,
        data_db_name=data_db_name,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "trading_days": sorted({str(doc["score_date"])[:8] for doc in docs if doc.get("score_date")}),
        "scores_a": score_cross_sections(docs, score_column=score_column, weights=weights),
        "forward_returns": forward_returns_by_signal_date(prices, signal_dates, horizon=horizon),
        "prices": prices,
    }
