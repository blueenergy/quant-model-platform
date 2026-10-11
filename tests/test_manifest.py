from pathlib import Path

import pandas as pd

from quant_model_platform.dataset.builder import build_synthetic_dataset
from quant_model_platform.dataset.manifest import content_hash, read_parquet_snapshot


def test_snapshot_hash_reproducible(tmp_path: Path):
    panel1, m1 = build_synthetic_dataset(parquet_dir=tmp_path)
    panel2 = read_parquet_snapshot(tmp_path / "csi1000_synthetic.parquet")
    h1 = content_hash(panel1, extra={"snapshot": m1.snapshot_id})
    h2 = content_hash(panel2, extra={"snapshot": m1.snapshot_id})
    assert h1 == h2
    assert m1.row_count == len(panel1)
