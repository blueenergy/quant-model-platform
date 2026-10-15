"""Register or update a shadow profile bound to a promoted training job."""

from __future__ import annotations

import argparse
import logging
from quant_model_platform.timeutil import utcnow

from quant_model_platform.db import COLLECTION_SHADOW_PROFILES, ensure_ml_indexes, get_ml_collections
from quant_model_platform.queue import clean_for_mongo
from quant_model_platform.shadow.resolve import merge_job_into_shadow_params


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Register model_shadow_profiles from source_job_id")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--model-id", default="", help="Override default promo-{index}-{short_job}")
    parser.add_argument("--disable", action="store_true")
    args = parser.parse_args(argv)

    collections = get_ml_collections()
    ensure_ml_indexes(collections)
    merged = merge_job_into_shadow_params({"source_job_id": args.source_job_id}, collections)
    if args.model_id:
        merged["model_id"] = args.model_id
    doc = clean_for_mongo(
        {
            "profile_id": args.profile_id,
            "source_job_id": args.source_job_id,
            "model_id": merged.get("model_id"),
            "index_code": merged.get("index_code"),
            "enabled": not args.disable,
            "params": {
                "source_job_id": args.source_job_id,
                "model_id": merged.get("model_id"),
                "index_code": merged.get("index_code"),
                "parquet_path": merged.get("parquet_path"),
                "snapshot_id": merged.get("snapshot_id"),
                "model_kind": merged.get("model_kind"),
            },
            "updated_at": utcnow(),
        }
    )
    collections[COLLECTION_SHADOW_PROFILES].update_one(
        {"profile_id": args.profile_id},
        {"$set": doc, "$setOnInsert": {"created_at": utcnow()}},
        upsert=True,
    )
    print(f"profile_id={args.profile_id} model_id={merged.get('model_id')} index_code={merged.get('index_code')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
