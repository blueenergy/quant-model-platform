"""Feature registry for MVP training (30–50 PIT / score columns)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str = "float64"
    #: Declared point-in-time policy for the feature's *source*, not an
    #: observation about the snapshot. See registry_document().
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


def annotate_population(
    entries: Sequence[Mapping[str, object]], frame: pd.DataFrame
) -> List[Dict[str, object]]:
    """Add ``populated`` / ``fill_rate`` per entry, read off the frame itself."""
    n = int(len(frame))
    out: List[Dict[str, object]] = []
    for entry in entries:
        doc = dict(entry)
        name = str(doc.get("name") or "")
        if name and name in frame.columns and n:
            filled = int(frame[name].notna().sum())
            doc["populated"] = filled > 0
            doc["fill_rate"] = round(filled / n, 4)
        else:
            doc["populated"] = False
            doc["fill_rate"] = 0.0
        out.append(doc)
    return out


def registry_document(
    registry: Sequence[FeatureSpec] | None = None,
    frame: pd.DataFrame | None = None,
) -> List[Dict[str, object]]:
    """The feature registry, as a record when ``frame`` is given and a plan otherwise.

    ``pit`` is a declared policy about a feature's *source*; it does not mean the
    snapshot holds the column. Ten of the fourteen entries are never populated by
    the production path (``dataset/production.py``), so a manifest built without
    the frame describes the plan rather than the data. Passing ``frame`` adds the
    observed fill so the two can be read apart.
    """
    reg = registry or DEFAULT_FEATURE_REGISTRY
    docs: List[Dict[str, object]] = [
        {"name": f.name, "dtype": f.dtype, "pit": f.pit, "missing_flag": f.missing_flag, "source": f.source}
        for f in reg
    ]
    return annotate_population(docs, frame) if frame is not None else docs
