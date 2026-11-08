"""Universe hygiene filters for the training panel.

Kept separate from ``production`` so the pure rules are testable without
Mongo, pyarrow, or the alt-factor loaders.
"""

from __future__ import annotations

import pandas as pd


def filter_panel_metadata(
    panel: pd.DataFrame,
    *,
    stock_info: pd.DataFrame,
    min_list_days: int = 250,
) -> pd.DataFrame:
    """Drop ST, too-new, and delisted rows before any label is attached.

    ``stock_info`` is the current listing record (``ts_code``, ``list_date``,
    ``delist_date``). The ST flag comes from the score document's name at the
    score date; that is the name the scorer saw on the day, not a perfect
    historical ST feed.
    """
    if panel.empty:
        return panel
    out = panel.copy()
    if "name" in out.columns:
        names = out["name"].fillna("").astype(str).str.upper()
        out = out[~names.str.contains("ST", regex=False)]
    info = stock_info.set_index("ts_code") if not stock_info.empty else pd.DataFrame()
    list_dates = (
        out["ts_code"].map(info["list_date"])
        if "list_date" in info.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    delist_dates = (
        out["ts_code"].map(info["delist_date"])
        if "delist_date" in info.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    dates = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
    list_dt = pd.to_datetime(list_dates, format="%Y%m%d", errors="coerce")
    delist_dt = pd.to_datetime(delist_dates, format="%Y%m%d", errors="coerce")
    age_days = (dates - list_dt).dt.days
    keep = age_days.isna() | (age_days >= min_list_days)
    out = out[keep].reset_index(drop=True)
    dates = dates[keep].reset_index(drop=True)
    delist_dt = delist_dt[keep].reset_index(drop=True)
    out = out[delist_dt.isna() | (dates < delist_dt)]
    return out.reset_index(drop=True)


def filter_panel_prices(
    panel: pd.DataFrame,
    *,
    price_keys: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows whose signal date has a traded price (not suspended/no data)."""
    if panel.empty or price_keys.empty:
        return panel
    keys = price_keys.loc[:, ["trade_date", "ts_code"]].drop_duplicates()
    keys["_has_price"] = True
    out = panel.merge(keys, on=["trade_date", "ts_code"], how="left")
    out = out[out["_has_price"].fillna(False)].drop(columns=["_has_price"])
    return out.reset_index(drop=True)
