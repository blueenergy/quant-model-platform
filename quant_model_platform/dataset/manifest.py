"""Snapshot manifests and stable content hashes."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quant_model_platform.dataset.features import registry_document


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(frame: pd.DataFrame, extra: Optional[Dict[str, Any]] = None) -> str:
    """Stable hash over sorted columns and row-wise values (for reproducibility tests)."""
    cols = sorted(frame.columns.tolist())
    subset = frame[cols].sort_values(by=cols, kind="mergesort").reset_index(drop=True)
    payload = {
        "columns": cols,
        "values": subset.astype(str).values.tolist(),
        "extra": extra or {},
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest


@dataclass(frozen=True)
class DatasetManifest:
    snapshot_id: str
    content_hash: str
    index_code: str
    start_date: str
    end_date: str
    feature_registry: List[Dict[str, object]]
    source_revision: str
    pit_policy: str
    parquet_uri: str
    row_count: int
    created_at: datetime

    def to_mongo(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "content_hash": self.content_hash,
            "index_code": self.index_code,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "feature_registry": self.feature_registry,
            "source_revision": self.source_revision,
            "pit_policy": self.pit_policy,
            "parquet_uri": self.parquet_uri,
            "row_count": self.row_count,
            "created_at": self.created_at,
        }


def build_manifest(
    frame: pd.DataFrame,
    *,
    index_code: str,
    start_date: str,
    end_date: str,
    parquet_path: Path,
    source_revision: str = "synthetic",
    pit_policy: str = "v0.1",
    feature_registry: Optional[Sequence[Dict[str, object]]] = None,
) -> DatasetManifest:
    reg = list(feature_registry or registry_document())
    extra = {"index_code": index_code, "start_date": start_date, "end_date": end_date, "registry": reg}
    h = content_hash(frame, extra=extra)
    snapshot_id = str(uuid.uuid4())
    uri = parquet_path.as_uri() if parquet_path.is_absolute() else str(parquet_path)
    return DatasetManifest(
        snapshot_id=snapshot_id,
        content_hash=h,
        index_code=index_code,
        start_date=start_date,
        end_date=end_date,
        feature_registry=reg,
        source_revision=source_revision,
        pit_policy=pit_policy,
        parquet_uri=uri,
        row_count=int(len(frame)),
        created_at=datetime.utcnow(),
    )


def write_parquet_snapshot(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_parquet_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
