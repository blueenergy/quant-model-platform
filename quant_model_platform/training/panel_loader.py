"""Load training panel from job params (parquet / production / synthetic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.builder import build_synthetic_dataset, load_dataset_parquet
from quant_model_platform.dataset.production import build_production_panel
from quant_model_platform.storage.s3 import ensure_local_parquet, publish_local_parquet


def load_training_panel(
    params: Dict[str, Any],
    *,
    parquet_root: Path,
    config: Optional[PlatformConfig] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """Return (panel, snapshot_id, parquet_path)."""
    cfg = config or PlatformConfig.from_env()
    cache_root = parquet_root / "cache"
    index_code = str(params.get("index_code", "csi1000"))
    parquet_path = params.get("parquet_path")

    if parquet_path:
        local_path = ensure_local_parquet(str(parquet_path), cache_root=cache_root)
        panel = load_dataset_parquet(local_path)
        snapshot_id = str(params.get("snapshot_id", "external"))
        return panel, snapshot_id, str(parquet_path)

    if str(params.get("data_source") or "") == "production_scores":
        start_date = str(params.get("start_date") or "")
        end_date = str(params.get("end_date") or "")
        if not start_date or not end_date:
            raise ValueError("production_scores requires start_date and end_date (YYYYMMDD)")
        panel, manifest = build_production_panel(
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
            parquet_dir=parquet_root / "snapshots",
            config=cfg,
        )
        snapshot_id = manifest.snapshot_id
        local_path = parquet_root / "snapshots" / f"{index_code}_production_{start_date}_{end_date}.parquet"
        uri = publish_local_parquet(
            local_path,
            index_code=index_code,
            snapshot_id=snapshot_id,
            config=cfg,
        )
        return panel, snapshot_id, uri

    panel, manifest = build_synthetic_dataset(index_code=index_code, parquet_dir=parquet_root / "snapshots")
    snapshot_id = manifest.snapshot_id
    local_path = parquet_root / "snapshots" / f"{index_code}_synthetic.parquet"
    uri = publish_local_parquet(
        local_path,
        index_code=index_code,
        snapshot_id=snapshot_id,
        config=cfg,
    )
    return panel, snapshot_id, uri
