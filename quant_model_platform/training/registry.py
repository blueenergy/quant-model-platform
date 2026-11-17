"""Register a completed training job as a research model in ``quant_ml.models``.

Training jobs may optionally carry ``params.model_id``. When present, the
worker creates or updates a registry document in the ``research`` state; it
never demotes an existing live/shadow model because a rerun produced a new
artifact.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from quant_model_platform.db import COLLECTION_MODELS
from quant_model_platform.timeutil import utcnow

log = logging.getLogger(__name__)

_HEX_SUFFIX = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


def _family_id(model_id: str, params: Dict[str, Any]) -> str:
    explicit = str(params.get("family_id") or "").strip()
    if explicit:
        return explicit
    text = str(model_id).strip()
    for prefix in ("research-", "promo-", "live-", "train-"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    parts = text.split("-")
    while parts and (_HEX_SUFFIX.match(parts[-1]) or (len(parts[-1]) == 8 and parts[-1].isdigit())):
        parts.pop()
    return "-".join(parts) or model_id


def _training_gate(
    result: Dict[str, Any],
    job_id: str,
    model_id: str,
    now,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    promotion = result.get("promotion") or {}
    return {
        "validation_id": f"{model_id}:training_gate:{result.get('result_id') or job_id}",
        "protocol": "training_gate",
        "scope": "primary",
        "job_id": job_id,
        "result_id": result.get("result_id"),
        "window": {
            "start": params.get("start_date"),
            "end": params.get("end_date"),
        },
        "metrics": {
            key: promotion.get(key)
            for key in ("model_mean_ic", "baseline_mean_ic", "baseline", "permutation_p", "fdr_alpha", "fold_count")
            if promotion.get(key) is not None
        },
        "verdict": "pass" if promotion.get("passed") else "fail",
        "reason": promotion.get("reason") or ("passed" if promotion.get("passed") else "unknown"),
        "caveats": (
            ["single-pool gate; operational promotion may override"]
            if promotion.get("reason") == "insufficient_universes"
            else []
        ),
        "created_at": now,
    }


def register_trained_model(
    collections: Dict[str, Any],
    job: Dict[str, Any],
    result: Dict[str, Any],
) -> Optional[str]:
    """Create/update the registry model for a completed training job."""
    models = collections.get(COLLECTION_MODELS)
    if models is None:
        return None
    params = dict(job.get("params") or {})
    model_id = str(params.get("model_id") or "").strip()
    if not model_id:
        return None
    job_id = str(job.get("job_id") or "")
    result_id = result.get("result_id")
    if not job_id or not result_id:
        return None
    now = utcnow()
    gate = _training_gate(result, job_id, model_id, now, params)
    artifact = {
        "result_id": result_id,
        "job_id": job_id,
        "role": "training_artifact",
        "artifact_uri": result.get("artifact_uri"),
        "artifact_hash": result.get("artifact_hash"),
        "snapshot_id": result.get("snapshot_id"),
        "parquet_path": result.get("parquet_path"),
        "model_kind": result.get("model_kind") or params.get("model_kind"),
        "training_end": params.get("end_date"),
        "code_version": result.get("code_version"),
        "created_at": now,
    }
    training_job = {
        "job_id": job_id,
        "name": job.get("name"),
        "role": "training_job",
        "link_quality": "exact",
        "status": "completed",
        "result_id": result_id,
        "created_at": job.get("created_at"),
        "completed_at": now,
    }
    recipe = {
        "source_job_id": job_id,
        "source_job_name": job.get("name"),
        "feature_names": params.get("feature_names"),
        "model_kind": params.get("model_kind"),
        "train_days": params.get("train_days"),
        "test_days": params.get("test_days"),
        "purge_days": params.get("purge_days"),
        "label_col": params.get("label_col"),
        "data_source": params.get("data_source"),
        "model_params": params.get("model_params"),
        "start_date": params.get("start_date"),
        "end_date": params.get("end_date"),
    }
    existing = models.find_one({"model_id": model_id})
    if existing:
        artifacts = list(existing.get("artifacts") or [])
        if not any(str(item.get("result_id") or "") == str(result_id) for item in artifacts):
            artifacts.append(artifact)
        linked_jobs = list(existing.get("training_jobs") or [])
        if not any(str(item.get("job_id") or "") == job_id for item in linked_jobs):
            linked_jobs.append(training_job)
        runs = list(existing.get("validation_runs") or [])
        if not any(str(item.get("validation_id") or "") == gate["validation_id"] for item in runs):
            runs.append(gate)
        updates = {
            "state": str(existing.get("state") or "research"),
            "display_name": existing.get("display_name") or job.get("name") or model_id,
            "index_code": params.get("index_code") or existing.get("index_code"),
            "recipe": recipe,
            "artifacts": artifacts,
            "training_jobs": linked_jobs,
            "validation_runs": runs,
            "operational_gate": {
                "passed": gate["verdict"] == "pass",
                "reason": gate["reason"],
                "result_id": gate["result_id"],
                "job_id": job_id,
            },
            "updated_at": now,
        }
        latest = existing.get("latest_validation") or {}
        if not latest or latest.get("protocol") == "training_gate":
            updates["latest_validation"] = {
                "protocol": gate["protocol"],
                "verdict": gate["verdict"],
                "metrics": gate["metrics"],
                "result_id": gate["result_id"],
                "created_at": now,
            }
        models.update_one({"_id": existing["_id"]}, {"$set": updates})
        return model_id

    doc = {
        "_id": model_id,
        "model_id": model_id,
        "display_name": params.get("model_name") or job.get("name") or model_id,
        "family_id": _family_id(model_id, params),
        "index_code": params.get("index_code"),
        "state": "research",
        "accepted": False,
        "accepted_reasons": [],
        "state_flags": {
            "live_active": False,
            "shadow_enabled": False,
            "shadow_profile_any": False,
            "validated": False,
        },
        "recipe": recipe,
        "artifacts": [artifact],
        "training_jobs": [training_job],
        "validation_runs": [gate],
        "latest_validation": {
            "protocol": gate["protocol"],
            "verdict": gate["verdict"],
            "metrics": gate["metrics"],
            "result_id": gate["result_id"],
            "created_at": now,
        },
        "operational_gate": {
            "passed": gate["verdict"] == "pass",
            "reason": gate["reason"],
            "result_id": gate["result_id"],
            "job_id": job_id,
        },
        "prediction_sets": [],
        "shadow_profiles": [],
        "activations": [],
        "state_history": [
            {
                "from": None,
                "to": "research",
                "at": now,
                "actor": "training_worker",
                "reason": "training job completed",
                "evidence": {"job_id": job_id, "result_id": result_id},
            }
        ],
        "health": {"active": False},
        "owner_user_ids": [str(job.get("user_id"))] if job.get("user_id") else [],
        "legacy": {"phase0_backfill": False, "note": "created by training worker"},
        "schema_version": "phase3-v1",
        "created_at": now,
        "updated_at": now,
    }
    models.insert_one(doc)
    log.info("registered trained model model_id=%s job_id=%s state=research", model_id, job_id)
    return model_id
