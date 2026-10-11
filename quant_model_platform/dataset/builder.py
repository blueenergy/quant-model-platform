"""Build synthetic or Parquet-backed panels for training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quant_model_platform.dataset.features import DEFAULT_FEATURE_REGISTRY, add_missing_flags, registry_names
from quant_model_platform.dataset.labels import attach_label_column
from quant_model_platform.dataset.manifest import build_manifest, read_parquet_snapshot, write_parquet_snapshot


def synthetic_price_panel(
    dates: Sequence[str],
    codes: Sequence[str],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = {c: 10.0 + rng.random() for c in codes}
    for d in dates:
        for c in codes:
            shock = rng.normal(0, 0.02)
            price[c] = max(0.5, price[c] * (1.0 + shock))
            rows.append({"trade_date": d, "ts_code": c, "open": price[c], "close": price[c] * (1 + shock / 2)})
    return pd.DataFrame(rows)


def synthetic_feature_panel(
    dates: Sequence[str],
    codes: Sequence[str],
    *,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = registry_names()
    rows = []
    for d in dates:
        for c in codes:
            row = {"trade_date": d, "ts_code": c}
            for n in names:
                row[n] = float(rng.normal(0, 1))
            rows.append(row)
    return pd.DataFrame(rows)


def build_synthetic_dataset(
    *,
    index_code: str = "csi1000",
    dates: Optional[Sequence[str]] = None,
    codes: Optional[Sequence[str]] = None,
    parquet_dir: Path,
) -> tuple[pd.DataFrame, object]:
    dates = list(dates or [f"2024-01-{d:02d}" for d in range(2, 32)])
    codes = list(codes or [f"{i:06d}.SZ" for i in range(1, 21)])
    features = add_missing_flags(synthetic_feature_panel(dates, codes), registry_names())
    prices = synthetic_price_panel(dates, codes)
    panel = attach_label_column(features, prices)
    path = parquet_dir / f"{index_code}_synthetic.parquet"
    write_parquet_snapshot(panel, path)
    manifest = build_manifest(
        panel,
        index_code=index_code,
        start_date=str(dates[0]),
        end_date=str(dates[-1]),
        parquet_path=path,
        source_revision="synthetic-v0",
        feature_registry=[{"name": f.name, "source": f.source} for f in DEFAULT_FEATURE_REGISTRY],
    )
    return panel, manifest


def load_dataset_parquet(path: Path) -> pd.DataFrame:
    return read_parquet_snapshot(path)
