"""Rule baseline, linear models, and LightGBM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


ModelKind = str  # rule_baseline | ridge | elasticnet | lightgbm


@dataclass
class TrainedModel:
    kind: ModelKind
    feature_names: List[str]
    artifact: Any
    params: Dict[str, Any]


def _feature_columns(frame: pd.DataFrame, feature_names: Sequence[str]) -> List[str]:
    """Declared names that exist *and* carry data in this frame.

    A column the panel never populated is dropped, not zero-filled: keeping it
    would make the all-finite mask below unsatisfiable, and filling it would
    train on a column of constants that no reader can see.
    """
    return [c for c in feature_names if c in frame.columns and frame[c].notna().any()]


def _feature_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """Features as-is, NaNs included.

    Deliberately not imputed here: the callers' ``np.isfinite`` mask is what
    keeps an incomplete row out of the fit, and filling first made that mask
    unconditionally true.
    """
    cols = _feature_columns(frame, feature_names)
    x = frame[cols].astype(float).values
    return x, cols


def _prediction_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """Fixed-width matrix for scoring: keep every declared feature in place.

    Training drops columns that are empty in the training panel
    (``_feature_columns``), because a constant column carries no information.
    Scoring cannot do that: the saved scaler/booster has a fixed width, so a
    column that is absent or all-NaN in today's snapshot must become NaN and
    stay in place instead of shrinking the matrix. The model callers then
    impute (linear, at the training mean) or let LightGBM handle NaN natively.
    """
    cols = list(feature_names)
    x = frame.reindex(columns=cols).astype(float).to_numpy()
    return x, cols


def train_rule_baseline(frame: pd.DataFrame, feature_names: Sequence[str], *, weight_col: Optional[str] = None) -> TrainedModel:
    """Equal-weight average of z-scored features (per date)."""
    names = _feature_columns(frame, feature_names)
    return TrainedModel(kind="rule_baseline", feature_names=names, artifact={"method": "equal_zscore_mean"}, params={})


def predict_rule_baseline(frame: pd.DataFrame, model: TrainedModel) -> pd.Series:
    names = model.feature_names
    if not names:
        return pd.Series(0.0, index=frame.index)
    # Reindex so a feature that is absent from today's snapshot becomes NaN
    # instead of raising KeyError; NaN z-scores are skipped by mean(axis=1).
    values = frame.reindex(columns=list(names)).astype(float)
    z = values.groupby(frame["trade_date"], sort=False).transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )
    return z.mean(axis=1)


def train_ridge(frame: pd.DataFrame, feature_names: Sequence[str], label_col: str, alpha: float = 1.0) -> TrainedModel:
    x, cols = _feature_matrix(frame, feature_names)
    y = frame[label_col].astype(float).values
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[mask])
    reg = Ridge(alpha=alpha)
    reg.fit(x_train, y[mask])
    return TrainedModel(kind="ridge", feature_names=cols, artifact={"model": reg, "scaler": scaler}, params={"alpha": alpha})


def predict_sklearn_linear(frame: pd.DataFrame, model: TrainedModel) -> pd.Series:
    x, _ = _prediction_matrix(frame, model.feature_names)
    scaler = model.artifact["scaler"]
    reg = model.artifact["model"]
    # Scoring is not fitting: a row missing one input still needs a rank, so
    # impute at the training mean. Post-scaling that is z=0, i.e. the feature
    # contributes nothing instead of dragging the row to an extreme.
    if x.size and not np.isfinite(x).all():
        x = np.where(np.isfinite(x), x, scaler.mean_)
    pred = reg.predict(scaler.transform(x))
    return pd.Series(pred, index=frame.index)


def train_elasticnet(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_col: str,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> TrainedModel:
    x, cols = _feature_matrix(frame, feature_names)
    y = frame[label_col].astype(float).values
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[mask])
    reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    reg.fit(x_train, y[mask])
    return TrainedModel(
        kind="elasticnet",
        feature_names=cols,
        artifact={"model": reg, "scaler": scaler},
        params={"alpha": alpha, "l1_ratio": l1_ratio},
    )


def train_lightgbm(frame: pd.DataFrame, feature_names: Sequence[str], label_col: str, params: Optional[Dict[str, Any]] = None) -> TrainedModel:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    x, cols = _feature_matrix(frame, feature_names)
    y = frame[label_col].astype(float).values
    # LightGBM handles missing feature values natively. Requiring every feature
    # to be present would drop the whole row once the feature set includes a
    # sparse factor (financials, analyst revisions, ...), which is the opposite
    # of what the missing-value split is for. Only the label must exist.
    mask = np.isfinite(y)
    default = {"objective": "regression", "metric": "l2", "verbosity": -1, "num_leaves": 31, "learning_rate": 0.05}
    merged = {**default, **(params or {})}
    # ``num_boost_round`` is an lgb.train argument, not a booster parameter;
    # passing it through ``params`` silently leaves the hard-coded 50 in place.
    num_boost_round = int(merged.pop("num_boost_round", 50))
    train_set = lgb.Dataset(x[mask], label=y[mask], feature_name=cols)
    booster = lgb.train(merged, train_set, num_boost_round=num_boost_round)
    return TrainedModel(kind="lightgbm", feature_names=cols, artifact={"booster": booster}, params=merged)


def predict_model(frame: pd.DataFrame, model: TrainedModel) -> pd.Series:
    if model.kind == "rule_baseline":
        return predict_rule_baseline(frame, model)
    if model.kind in ("ridge", "elasticnet"):
        return predict_sklearn_linear(frame, model)
    if model.kind == "lightgbm":
        x, _ = _prediction_matrix(frame, model.feature_names)
        booster = model.artifact["booster"]
        return pd.Series(booster.predict(x), index=frame.index)
    raise ValueError(f"unknown model kind: {model.kind}")


def train_model(kind: ModelKind, frame: pd.DataFrame, feature_names: Sequence[str], label_col: str, params: Optional[Dict[str, Any]] = None) -> TrainedModel:
    params = params or {}
    if kind == "rule_baseline":
        return train_rule_baseline(frame, feature_names)
    if kind == "ridge":
        return train_ridge(frame, feature_names, label_col, alpha=float(params.get("alpha", 1.0)))
    if kind == "elasticnet":
        return train_elasticnet(
            frame,
            feature_names,
            label_col,
            alpha=float(params.get("alpha", 0.01)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
        )
    if kind == "lightgbm":
        return train_lightgbm(frame, feature_names, label_col, params=params)
    raise ValueError(f"unsupported model kind: {kind}")
