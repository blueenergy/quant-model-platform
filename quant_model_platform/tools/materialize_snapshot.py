"""CLI: materialize production Parquet snapshot and optional S3 + quant_ml manifest."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.production import build_production_panel
from quant_model_platform.db import COLLECTION_DATASETS, get_ml_collections
from quant_model_platform.queue import clean_for_mongo
from quant_model_platform.storage.s3 import publish_local_parquet


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Materialize ML training snapshot from stock_scores")
    parser.add_argument("--index-code", default="csi1000")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--register-mongo", action="store_true", help="Insert dataset_manifests in quant_ml")
    args = parser.parse_args(argv)

    cfg = PlatformConfig.from_env()
    parquet_dir = Path(cfg.parquet_root) / "snapshots"
    _panel, manifest = build_production_panel(
        index_code=args.index_code,
        start_date=args.start_date,
        end_date=args.end_date,
        parquet_dir=parquet_dir,
        config=cfg,
    )
    local_path = parquet_dir / f"{args.index_code}_production_{args.start_date}_{args.end_date}.parquet"
    uri = publish_local_parquet(
        local_path,
        index_code=args.index_code,
        snapshot_id=manifest.snapshot_id,
        config=cfg,
    )
    print(f"snapshot_id={manifest.snapshot_id}")
    print(f"parquet_path={uri}")
    print(f"rows={manifest.row_count}")

    if args.register_mongo:
        collections = get_ml_collections(config=cfg)
        doc = clean_for_mongo({**manifest.to_mongo(), "parquet_path": uri, "created_at": datetime.utcnow()})
        collections[COLLECTION_DATASETS].insert_one(doc)
        print(f"registered dataset_manifests snapshot_id={manifest.snapshot_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
