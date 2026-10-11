"""Feature registry for MVP training (30–50 PIT / score columns)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str = "float64"
    pit: bool = True
    missing_flag: bool = True
    source: str = "synthetic"


# MVP registry: extend without breaking manifest hash when adding at end only.
DEFAULT_FEATURE_REGISTRY: List[FeatureSpec] = [
    FeatureSpec("growth_score", source="stock_scores"),
    FeatureSpec("cycle_score", source="stock_scores"),
    FeatureSpec("value_score", source="stock_scores"),
    FeatureSpec("quality_score", source="stock_scores"),
    FeatureSpec("momentum_20d", source="prices"),
    FeatureSpec("momentum_60d", source="prices"),
    FeatureSpec("volatility_20d", source="prices"),
    FeatureSpec("turnover_20d", source="prices"),
    FeatureSpec("log_market_cap", source="fundamentals"),
    FeatureSpec("pe_ttm", source="fundamentals"),
    FeatureSpec("pb", source="fundamentals"),
    FeatureSpec("roe", source="fundamentals"),
    FeatureSpec("revenue_yoy", source="fundamentals"),
    FeatureSpec("profit_yoy", source="fundamentals"),
]


def registry_names(registry: Sequence[FeatureSpec] | None = None) -> List[str]:
    reg = list(registry or DEFAULT_FEATURE_REGISTRY)
    return [f.name for f in reg]


def add_missing_flags(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for name in feature_names:
        if name not in out.columns:
            continue
        flag = f"{name}__missing"
        out[flag] = out[name].isna().astype("int8")
    return out


def registry_document(registry: Sequence[FeatureSpec] | None = None) -> List[Dict[str, object]]:
    reg = registry or DEFAULT_FEATURE_REGISTRY
    return [
        {"name": f.name, "dtype": f.dtype, "pit": f.pit, "missing_flag": f.missing_flag, "source": f.source}
        for f in reg
    ]
