"""SSE trading calendar from ``quant_data.trade_calendar`` (tushare trade_cal).

The index_prices collection only carries historical dates; the explicit calendar
is needed to schedule future signals / execute dates.
"""
from __future__ import annotations


def trading_days(client, data_db: str, start: str = "20150101", end: str = "20271231") -> list[str]:
    # is_open is int 1 on rows written by the backfill script and the string
    # "1" on rows refreshed from tushare; accept both so a writer flip cannot
    # silently turn this into an empty calendar.
    return sorted(str(r["trade_date"]) for r in client[data_db]["trade_calendar"].find(
        {"exchange": "SSE", "is_open": {"$in": [1, "1"]}, "trade_date": {"$gte": start, "$lte": end}},
        {"_id": 0, "trade_date": 1},
    ))
