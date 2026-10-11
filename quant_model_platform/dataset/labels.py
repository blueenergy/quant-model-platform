"""Forward return labels aligned with ScoreBacktester next_open semantics."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from quant_model_platform.protocol import LABEL_HORIZON_DAYS, LabelNeutralize

LabelNeutralizeMode = LabelNeutralize


def forward_open_to_open_return(
    prices: pd.DataFrame,
    *,
    horizon: int = LABEL_HORIZON_DAYS,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    open_col: str = "open",
) -> pd.Series:
    """Return from next session open after signal date to open after horizon sessions."""
    if prices.empty:
        return pd.Series(dtype=float)
    frame = prices[[date_col, code_col, open_col]].copy()
    frame = frame.sort_values([code_col, date_col])
    grouped = frame.groupby(code_col, sort=False)[open_col]
    # Entry: next bar open (T+1), exit: open after horizon trading days from entry bar.
    entry = grouped.shift(-1)
    exit_ = grouped.shift(-(1 + horizon))
    ret = exit_ / entry - 1.0
    return ret


def neutralize_cross_section(
    frame: pd.DataFrame,
    value_col: str,
    *,
    date_col: str = "trade_date",
    size_col: Optional[str] = "log_market_cap",
    mode: LabelNeutralizeMode = "demean",
) -> pd.Series:
    """Pool-neutralize label per trade_date."""
    if mode == "none":
        return frame[value_col]
    if mode == "demean":
        return frame.groupby(date_col, sort=False)[value_col].transform(lambda s: s - s.mean())
    if mode == "size_neutral":
        if size_col is None or size_col not in frame.columns:
            return frame.groupby(date_col, sort=False)[value_col].transform(lambda s: s - s.mean())

        def _residual(group: pd.DataFrame) -> pd.Series:
            y = group[value_col].astype(float)
            x = group[size_col].astype(float)
            mask = y.notna() & x.notna()
            if mask.sum() < 3:
                return y - y.mean()
            coef = np.polyfit(x[mask], y[mask], 1)
            pred = coef[0] * x + coef[1]
            return y - pred

        return frame.groupby(date_col, sort=False, group_keys=False).apply(_residual)
    raise ValueError(f"unknown neutralize mode: {mode}")


def attach_label_column(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int = LABEL_HORIZON_DAYS,
    neutralize: LabelNeutralizeMode = "demean",
    label_col: str = "label_fwd_20d",
) -> pd.DataFrame:
    """Merge forward return label onto feature panel."""
    out = panel.copy()
    raw = forward_open_to_open_return(prices, horizon=horizon)
    tmp = prices[["trade_date", "ts_code"]].copy()
    tmp["label_raw"] = raw.values
    out = out.merge(tmp, on=["trade_date", "ts_code"], how="left")
    out[label_col] = neutralize_cross_section(out, "label_raw", mode=neutralize)
    out.drop(columns=["label_raw"], inplace=True)
    return out
