"""Register or update a shadow profile bound to a promoted training job."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--use-saved-artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the promoted job's artifact (default); use --no-use-saved-artifact to refit full-sample each run.",
    )
    parser.add_argument("--disable", action="store_true")
    parser.add_argument(
        "--params-json",
        default="",
        help=(
            "JSON object merged over the source job's params, for a profile that "
            "does not deploy the job's artifact (e.g. train_mode=walk_forward_expanding)."
        ),
    )
    args = parser.parse_args(argv)

    collections = get_ml_collections()
    ensure_ml_indexes(collections)
    merged = merge_job_into_shadow_params({"source_job_id": args.source_job_id}, collections)
    if args.params_json:
        extra = json.loads(args.params_json)
        if not isinstance(extra, dict):
            raise SystemExit("--params-json must be a JSON object")
        merged.update(extra)
    if args.model_id:
        merged["model_id"] = args.model_id
    if args.use_saved_artifact and not merged.get("artifact_uri"):
        raise SystemExit(
            f"source job {args.source_job_id} has no artifact_uri; cannot use_saved_artifact"
        )
    params = dict(merged)
    params["use_saved_artifact"] = bool(args.use_saved_artifact)
    doc = clean_for_mongo(
        {
            "profile_id": args.profile_id,
            "source_job_id": args.source_job_id,
            "model_id": merged.get("model_id"),
            "index_code": merged.get("index_code"),
            "enabled": not args.disable,
            "params": params,
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
