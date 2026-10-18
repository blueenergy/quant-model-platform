"""Forward return labels aligned with ScoreBacktester next_open semantics."""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from quant_model_platform.protocol import LABEL_HORIZON_DAYS, LabelNeutralize

log = logging.getLogger(__name__)

LabelNeutralizeMode = LabelNeutralize

#: Fewest usable pairs a date needs before its label is residualized on size.
MIN_SIZE_SAMPLES = 3


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
            log.warning("size_neutral: %r is absent; falling back to demean", size_col)
            return frame.groupby(date_col, sort=False)[value_col].transform(lambda s: s - s.mean())
        # Existence is not data. The production panel pre-creates every registry
        # name, so ``log_market_cap`` is a present-but-empty column until a
        # loader fills it, and this branch degrades to a demean without saying
        # so. Nothing passes ``size_neutral`` today, so the warning is for the
        # first caller that does: it should not have to infer from unchanged
        # numbers that its residualization never ran.
        usable = (frame[value_col].notna() & frame[size_col].notna()).groupby(
            frame[date_col], sort=False
        ).sum()
        thin = int((usable < MIN_SIZE_SAMPLES).sum())
        if thin:
            log.warning(
                "size_neutral: %d of %d dates have fewer than %d usable %s samples; "
                "those dates fall back to demean",
                thin,
                int(usable.size),
                MIN_SIZE_SAMPLES,
                size_col,
            )

        def _residual(group: pd.DataFrame) -> pd.Series:
            y = group[value_col].astype(float)
            x = group[size_col].astype(float)
            mask = y.notna() & x.notna()
            if mask.sum() < MIN_SIZE_SAMPLES:
                return y - y.mean()
            coef = np.polyfit(x[mask], y[mask], 1)
            pred = coef[0] * x + coef[1]
            return y - pred

        # Explicit concat rather than groupby.apply: on pandas 3 a
        # DataFrameGroupBy.apply wraps the per-group Series into a DataFrame,
        # which then gets assigned to the label column as a frame -- the
        # caller must receive a Series (b98062e flagged this branch as
        # returning a frame on its then-unreachable path; 2026-10-18 is the
        # first caller). reindex restores the original row order.
        parts = [_residual(group) for _, group in frame.groupby(date_col, sort=False)]
        return pd.concat(parts).reindex(frame.index)
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
    # Assign the Series, not ``raw.values``: ``forward_open_to_open_return``
    # sorts by (code, date) internally and ``prices`` arrives in whatever order
    # Mongo returned, so a positional assignment pastes each return onto a
    # different row. That reads as a plausible label -- same scale, no error --
    # and it is invisible while the two orders happen to agree, which is how it
    # survived until a wider date window moved the query's return order.
    tmp["label_raw"] = raw
    out = out.merge(tmp, on=["trade_date", "ts_code"], how="left")
    out[label_col] = neutralize_cross_section(out, "label_raw", mode=neutralize)
    out.drop(columns=["label_raw"], inplace=True)
    return out
