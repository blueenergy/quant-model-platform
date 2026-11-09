"""Build the all-A PIT panel used by the production signal.

Universe: every SH/SZ stock with a price row on the date, >= 365 calendar days
since listing, not ST at that date (PIT namechange), and >= 20M CNY 20-day
average turnover. Label: 20-day forward open-to-open return, residualised
against log market cap (size_neutral).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.labels import attach_label_column
from quant_model_platform.dataset.production import (
    _attach_raw_factors,
    add_missing_flags,
    load_price_history,
)
from quant_model_platform.tools.all_a.config import (
    FEATURES,
    LABEL,
    LIQ_MIN,
    PANEL_DIR,
    START,
    WARMUP,
)


def build_panel(end_date: str, out_dir: Path = PANEL_DIR) -> Path:
    cfg = PlatformConfig.from_env()
    client = MongoClient(cfg.mongo_uri, tz_aware=True)
    data_db = cfg.data_db_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"all_a_sn_{START}_{end_date}.parquet"
    if out.exists():
        print(f"panel exists: {out}", flush=True)
        return out

    raw_keys = pd.DataFrame(list(client[data_db]["volume_price"].find(
        {"trade_date": {"$gte": START, "$lte": end_date}, "universe_type": {"$ne": "watchlist_plan_strategy"}},
        {"_id": 0, "symbol": 1, "trade_date": 1, "amount": 1},
    )))
    raw_keys = raw_keys.rename(columns={"symbol": "ts_code"})
    raw_keys["trade_date"] = raw_keys["trade_date"].astype(str)
    raw_keys["amount"] = pd.to_numeric(raw_keys["amount"], errors="coerce")
    raw_keys = raw_keys.drop_duplicates(["trade_date", "ts_code"]).sort_values(["ts_code", "trade_date"])
    raw_keys["amt20"] = raw_keys.groupby("ts_code", sort=False)["amount"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    raw_keys = raw_keys[raw_keys["amt20"] >= LIQ_MIN].drop(columns=["amount", "amt20"])
    print("after liquidity", len(raw_keys), flush=True)

    info = pd.DataFrame(list(client[data_db]["stock_info"].find({}, {"_id": 0, "symbol": 1, "list_date": 1})))
    info["ts_code"] = info["symbol"].astype(str)
    info["list_dt"] = pd.to_datetime(info["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    info = info[["ts_code", "list_dt"]].drop_duplicates("ts_code")
    panel = raw_keys.merge(info, on="ts_code", how="left")
    panel["dt"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d", errors="coerce")
    panel = panel[
        panel["ts_code"].str.match(r"^[036]")
        & panel["list_dt"].notna()
        & (panel["list_dt"] <= panel["dt"] - pd.Timedelta(days=365))
    ].drop(columns=["list_dt"]).sort_values(["dt", "ts_code"])
    print("after metadata", len(panel), flush=True)

    nc = pd.DataFrame(list(client[data_db]["name_change"].find(
        {}, {"_id": 0, "ts_code": 1, "start_date": 1, "name": 1},
    )))
    nc["start_dt"] = pd.to_datetime(nc["start_date"], format="%Y%m%d", errors="coerce")
    nc = nc.dropna(subset=["start_dt", "ts_code"]).sort_values(["start_dt", "ts_code"])
    first_name = nc.sort_values("start_dt").groupby("ts_code", sort=False)["name"].first().to_dict()
    panel = pd.merge_asof(
        panel, nc[["ts_code", "start_dt", "name"]],
        left_on="dt", right_on="start_dt", by="ts_code", direction="backward",
    ).drop(columns=["start_dt"])
    panel["name_pit"] = panel["name"].fillna(panel["ts_code"].map(first_name))
    is_st = panel["name_pit"].astype(str).str.contains("ST", case=True, regex=False) | panel["name_pit"].astype(str).str.contains("退", regex=False)
    panel = panel[~is_st].drop(columns=["name", "name_pit"])
    print("after PIT ST", len(panel), flush=True)

    symbols = sorted(panel["ts_code"].unique())
    prices = load_price_history(client, data_db_name=data_db, symbols=symbols, start_date=WARMUP, end_date=end_date)
    panel = _attach_raw_factors(client, panel, prices, data_db_name=data_db, symbols=symbols, start_date=WARMUP, end_date=end_date)
    panel = attach_label_column(panel, prices, horizon=20, neutralize="size_neutral", label_col=LABEL)
    panel = add_missing_flags(panel, FEATURES)
    panel = panel.drop(columns=["dt"])
    panel.to_parquet(out, index=False)
    print(f"saved {out} {panel.shape}", flush=True)
    return out
