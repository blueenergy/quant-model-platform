"""CLI: materialize production Parquet snapshot and optional S3 + quant_ml manifest."""

from __future__ import annotations

import argparse
import logging
from quant_model_platform.timeutil import utcnow
from pathlib import Path

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.channels import PRODUCTION_CURRENT_CHANNEL
from quant_model_platform.dataset.production import build_production_panel, latest_score_date_for_index
from quant_model_platform.db import COLLECTION_DATASETS, get_ml_client, get_ml_collections
from quant_model_platform.queue import clean_for_mongo
from quant_model_platform.storage.s3 import canonical_snapshot_id, publish_canonical_local_parquet, publish_local_parquet

log = logging.getLogger(__name__)


def _resolve_end_date(end_date: str, *, index_code: str, cfg: PlatformConfig) -> str:
    if end_date.lower() != "latest":
        return end_date
    client = get_ml_client(cfg)
    try:
        latest = latest_score_date_for_index(client, mongo_db=cfg.mongo_db, index_code=index_code)
    finally:
        client.close()
    if not latest:
        raise RuntimeError(f"no stock_scores rows for index={index_code}; cannot resolve --end-date latest")
    log.info("resolved end_date=latest -> %s for index=%s", latest, index_code)
    return latest


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Materialize ML training snapshot from stock_scores")
    parser.add_argument("--index-code", default="csi1000")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD or latest")
    parser.add_argument("--canonical", action="store_true", help="Overwrite per-index current.parquet + stable snapshot_id")
    parser.add_argument("--register-mongo", action="store_true", help="Insert/upsert dataset_manifests in quant_ml")
    args = parser.parse_args(argv)

    cfg = PlatformConfig.from_env()
    end_date = _resolve_end_date(args.end_date, index_code=args.index_code, cfg=cfg)
    parquet_dir = Path(cfg.parquet_root) / "snapshots"
    _panel, manifest = build_production_panel(
        index_code=args.index_code,
        start_date=args.start_date,
        end_date=end_date,
        parquet_dir=parquet_dir,
        config=cfg,
        use_canonical_snapshot=args.canonical,
    )
    local_path = parquet_dir / (
        f"{args.index_code}_current.parquet"
        if args.canonical
        else f"{args.index_code}_production_{args.start_date}_{end_date}.parquet"
    )
    if args.canonical:
        uri = publish_canonical_local_parquet(local_path, index_code=args.index_code, config=cfg)
    else:
        uri = publish_local_parquet(
            local_path,
            index_code=args.index_code,
            snapshot_id=manifest.snapshot_id,
            config=cfg,
        )
    print(f"snapshot_id={manifest.snapshot_id}")
    print(f"parquet_path={uri}")
    print(f"rows={manifest.row_count}")
    print(f"end_date={manifest.end_date}")

    if args.register_mongo:
        collections = get_ml_collections(config=cfg)
        doc = clean_for_mongo(
            {
                **manifest.to_mongo(),
                "parquet_path": uri,
                "parquet_uri": uri,
                "updated_at": utcnow(),
            }
        )
        if args.canonical:
            doc["channel"] = PRODUCTION_CURRENT_CHANNEL
            # Mongo rejects the same path in $set and $setOnInsert.
            doc.pop("created_at", None)
            collections[COLLECTION_DATASETS].update_one(
                {"snapshot_id": canonical_snapshot_id(args.index_code)},
                {"$set": doc, "$setOnInsert": {"created_at": utcnow()}},
                upsert=True,
            )
            print(f"upserted dataset_manifests snapshot_id={manifest.snapshot_id} channel={PRODUCTION_CURRENT_CHANNEL}")
        else:
            doc["created_at"] = utcnow()
            collections[COLLECTION_DATASETS].insert_one(doc)
            print(f"registered dataset_manifests snapshot_id={manifest.snapshot_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
