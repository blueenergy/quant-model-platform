"""Point-in-time walk-forward refit for the deployed artifact.

The training job already evaluates fold by fold (purged walk-forward), but the
artifact it published was fit on the whole panel, so it had seen every label
up to the snapshot end -- including the crash-and-bounce of the most recent
quarter. That is what produced the "crash pocket" style drift documented for
``research-csi1000-clean-raw5``.

This module fits the artifact the research replay actually used: for a signal
date T, train only on rows dated at or before ``T - (label_horizon + 1)``,
which is the last row whose 20-day label window ends at or before T's open.
The signal itself is T's close, so a training label that ends at T's open
carries no information about the move being traded.

Everything here is a pure function of a panel: no Mongo, no clock, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quant_model_platform.dataset.features import registry_names
from quant_model_platform.training.models import TrainedModel, train_model

#: Rows are dropped when a declared feature is missing on that row.
#:
#: LightGBM can route missing values natively, so this is a choice, not a
#: limitation. The research replay that validated the strategy dropped them,
#: and a shadow run has to score the same universe the replay scored; keeping
#: the default aligned is what makes parity checkable.
DEFAULT_DROP_INCOMPLETE_FEATURES = True

#: ``label_horizon_days + 1``: the label of a row dated d ends at the open of
#: d + 21 bars, so the last row whose label is fully in the past at signal
#: date T is the one dated T - 21.
DEFAULT_TRAIN_END_OFFSET_DAYS = 21


@dataclass(frozen=True)
class WalkForwardFit:
    """A model fit on the past of one signal date, with its provenance."""

    model: TrainedModel
    signal_date: str
    train_end: str
    n_train_rows: int
    n_train_dates: int
    feature_names: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "train_mode": "walk_forward_expanding",
            "signal_date": self.signal_date,
            "train_end": self.train_end,
            "n_train_rows": self.n_train_rows,
            "n_train_dates": self.n_train_dates,
            "feature_names": list(self.feature_names),
        }


def resolve_train_end(
    trade_dates: Sequence[str],
    signal_date: str,
    offset_days: int = DEFAULT_TRAIN_END_OFFSET_DAYS,
) -> str:
    """Last panel date whose label window ends at or before the signal date.

    ``offset_days`` counts trading sessions, not calendar days: a symbol that
    was halted reaches further back in wall-clock time than one that traded
    every day, and the label window is defined in sessions.
    """
    offset = int(offset_days)
    if offset < 1:
        raise ValueError(f"train_end_offset_days must be >= 1, got {offset_days}")
    dates = sorted({str(d) for d in trade_dates})
    signal = str(signal_date)
    if signal not in dates:
        raise ValueError(f"signal_date={signal} is not in the panel")
    index = dates.index(signal)
    if index < offset:
        raise ValueError(
            f"not enough history for a point-in-time fit at {signal}: "
            f"need {offset} earlier sessions, panel starts at {dates[0]}"
        )
    return dates[index - offset]


def _complete_rows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_col: str,
    *,
    drop_incomplete_features: bool,
) -> pd.DataFrame:
    out = frame
    if label_col in out.columns:
        out = out[out[label_col].notna()]
    if drop_incomplete_features:
        present = [c for c in feature_names if c in out.columns]
        if present:
            out = out.dropna(subset=present)
    return out


def walk_forward_fit(
    panel: pd.DataFrame,
    params: Dict[str, Any],
    *,
    signal_date: str,
    train_end_offset_days: int = DEFAULT_TRAIN_END_OFFSET_DAYS,
    label_col: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    drop_incomplete_features: bool = DEFAULT_DROP_INCOMPLETE_FEATURES,
) -> WalkForwardFit:
    """Fit one model on the data available before ``signal_date``.

    ``panel`` must carry ``trade_date``, ``ts_code`` and the feature columns;
    it may carry rows dated after ``signal_date`` (they are ignored) and it
    should carry the label for the training rows.
    """
    if panel is None or panel.empty:
        raise ValueError("walk-forward fit needs a non-empty panel")
    if "trade_date" not in panel.columns:
        raise ValueError("panel is missing trade_date")

    label = str(label_col or params.get("label_col", "label_fwd_20d"))
    names = list(feature_names or params.get("feature_names") or registry_names())
    kind = str(params.get("model_kind", "ridge"))

    dates = sorted({str(d) for d in panel["trade_date"]})
    train_end = resolve_train_end(dates, signal_date, train_end_offset_days)

    train = panel[panel["trade_date"].astype(str) <= train_end]
    train = _complete_rows(
        train,
        names,
        label,
        drop_incomplete_features=drop_incomplete_features,
    )
    if train.empty:
        raise ValueError(
            f"no training rows for signal_date={signal_date} train_end={train_end}"
        )

    model = train_model(kind, train, names, label, params=params.get("model_params"))
    return WalkForwardFit(
        model=model,
        signal_date=str(signal_date),
        train_end=train_end,
        n_train_rows=int(len(train)),
        n_train_dates=int(train["trade_date"].nunique()),
        feature_names=list(model.feature_names),
    )


def signal_rows(
    panel: pd.DataFrame,
    *,
    signal_date: str,
    feature_names: Sequence[str],
    drop_incomplete_features: bool = DEFAULT_DROP_INCOMPLETE_FEATURES,
) -> pd.DataFrame:
    """The cross-section scored on ``signal_date``.

    Same row filter as the training side, so the replay and the shadow score
    the same universe.
    """
    day = panel[panel["trade_date"].astype(str) == str(signal_date)]
    if day.empty:
        raise ValueError(f"no rows for trade_date={signal_date}")
    day = _complete_rows(
        day,
        feature_names,
        label_col="__none__",
        drop_incomplete_features=drop_incomplete_features,
    )
    if day.empty:
        raise ValueError(f"no scorable rows for trade_date={signal_date}")
    return day
