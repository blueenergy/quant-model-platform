"""Full-sample model fit (same recipe as walk-forward promotion summary)."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd

from quant_model_platform.dataset.features import registry_names
from quant_model_platform.training.models import TrainedModel, train_model


def fit_full_sample_model(
    panel: pd.DataFrame,
    params: Dict[str, Any],
    *,
    label_col: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> TrainedModel:
    label = str(label_col or params.get("label_col", "label_fwd_20d"))
    names = list(feature_names or params.get("feature_names") or registry_names())
    model_kind = str(params.get("model_kind", "ridge"))
    train = panel.dropna(subset=[label])
    if train.empty:
        raise ValueError("no labeled rows for full-sample fit")
    return train_model(model_kind, train, names, label, params=params.get("model_params"))
