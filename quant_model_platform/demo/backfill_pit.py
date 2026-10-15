"""Point-in-time refit: train only on dates whose 20d label is known by T."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd

from quant_model_platform.protocol import LABEL_HORIZON_DAYS
from quant_model_platform.training.fit import fit_full_sample_model
from quant_model_platform.training.models import predict_model


def truncate_panel(panel: pd.DataFrame, as_of: str, *, date_col: str = "trade_date") -> pd.DataFrame:
    """Drop every row dated after the signal day."""
    day = str(as_of)[:8]
    out = panel.loc[panel[date_col].astype(str).str[:8] <= day].copy()
    if out.empty:
        raise ValueError(f"no rows on or before {day}")
    if (out[date_col].astype(str).str[:8] > day).any():
        raise RuntimeError("truncate_panel left a future row")
    return out


def pit_train_dates(trading_days: Sequence[str], as_of: str, *, horizon: int = LABEL_HORIZON_DAYS) -> List[str]:
    """Dates whose forward label is realizable using prices only through ``as_of``.

    Label on date index i needs bars through i+1+horizon, so the last trainable
    index is ``as_of_index - horizon - 1``.
    """
    day = str(as_of)[:8]
    days = sorted({str(d)[:8] for d in trading_days if str(d)[:8] <= day})
    if day not in days:
        days.append(day)
        days.sort()
    as_of_idx = days.index(day)
    max_idx = as_of_idx - horizon - 1
    if max_idx < 0:
        return []
    return days[: max_idx + 1]


def predict_cross_section_as_of(
    panel: pd.DataFrame,
    params: Dict[str, Any],
    as_of: str,
    *,
    horizon: int = LABEL_HORIZON_DAYS,
    label_col: str = "label_fwd_20d",
) -> pd.DataFrame:
    """Refit on PIT rows and score the ``as_of`` cross-section only."""
    hist = truncate_panel(panel, as_of)
    allowed = set(pit_train_dates(hist["trade_date"].astype(str).str[:8].unique(), as_of, horizon=horizon))
    if not allowed:
        raise ValueError(f"not enough history before {as_of} for horizon={horizon}")
    train = hist.loc[hist["trade_date"].astype(str).str[:8].isin(allowed)]
    model = fit_full_sample_model(train, params, label_col=label_col)
    day = hist.loc[hist["trade_date"].astype(str).str[:8] == str(as_of)[:8]].copy()
    if day.empty:
        raise ValueError(f"no cross-section on {as_of}")
    day["score"] = predict_model(day, model)
    day["score_source"] = "backfill_pit"
    return day[["trade_date", "ts_code", "score", "score_source"]].reset_index(drop=True)
