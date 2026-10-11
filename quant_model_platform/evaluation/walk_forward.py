"""Purged walk-forward splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Sequence

import pandas as pd

from quant_model_platform.protocol import MAX_PURGE_DAYS


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_dates: List[str]
    test_dates: List[str]
    purge_dates: List[str]


def _sorted_unique_dates(dates: Sequence[str]) -> List[str]:
    return sorted(set(str(d) for d in dates))


def purged_walk_forward_folds(
    trade_dates: Sequence[str],
    *,
    train_days: int,
    test_days: int,
    purge_days: int = MAX_PURGE_DAYS,
    step_days: int | None = None,
) -> List[WalkForwardFold]:
    """Rolling train/test with an embargo (purge) gap between them."""
    dates = _sorted_unique_dates(trade_dates)
    if len(dates) < train_days + purge_days + test_days:
        return []
    step = step_days or test_days
    folds: List[WalkForwardFold] = []
    fold_id = 0
    start = 0
    while True:
        train_end = start + train_days
        purge_end = train_end + purge_days
        test_end = purge_end + test_days
        if test_end > len(dates):
            break
        train_dates = dates[start:train_end]
        purge_dates = dates[train_end:purge_end]
        test_dates = dates[purge_end:test_end]
        folds.append(WalkForwardFold(fold_id=fold_id, train_dates=train_dates, test_dates=test_dates, purge_dates=purge_dates))
        fold_id += 1
        start += step
    return folds


def split_frame_by_dates(frame: pd.DataFrame, dates: Sequence[str], date_col: str = "trade_date") -> pd.DataFrame:
    mask = frame[date_col].isin(list(dates))
    return frame.loc[mask].copy()
