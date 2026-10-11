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


def _feature_matrix(frame: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    cols = [c for c in feature_names if c in frame.columns]
    x = frame[cols].astype(float).fillna(0.0).values
    return x, cols


def train_rule_baseline(frame: pd.DataFrame, feature_names: Sequence[str], *, weight_col: Optional[str] = None) -> TrainedModel:
    """Equal-weight average of z-scored features (per date)."""
    names = [n for n in feature_names if n in frame.columns]
    return TrainedModel(kind="rule_baseline", feature_names=names, artifact={"method": "equal_zscore_mean"}, params={})


def predict_rule_baseline(frame: pd.DataFrame, model: TrainedModel) -> pd.Series:
    names = model.feature_names
    if not names:
        return pd.Series(0.0, index=frame.index)
    z = frame.groupby("trade_date", sort=False)[names].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    return z[names].mean(axis=1)


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
    x, _ = _feature_matrix(frame, model.feature_names)
    scaler = model.artifact["scaler"]
    reg = model.artifact["model"]
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
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    default = {"objective": "regression", "metric": "l2", "verbosity": -1, "num_leaves": 31, "learning_rate": 0.05}
    merged = {**default, **(params or {})}
    train_set = lgb.Dataset(x[mask], label=y[mask], feature_name=cols)
    booster = lgb.train(merged, train_set, num_boost_round=50)
    return TrainedModel(kind="lightgbm", feature_names=cols, artifact={"booster": booster}, params=merged)


def predict_model(frame: pd.DataFrame, model: TrainedModel) -> pd.Series:
    if model.kind == "rule_baseline":
        return predict_rule_baseline(frame, model)
    if model.kind in ("ridge", "elasticnet"):
        return predict_sklearn_linear(frame, model)
    if model.kind == "lightgbm":
        x, _ = _feature_matrix(frame, model.feature_names)
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
