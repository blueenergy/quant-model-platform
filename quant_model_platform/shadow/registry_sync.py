"""Sync shadow-profile enable/disable into ``quant_ml.models``.

The shadow loop owns the operational profile state. This helper mirrors it
into the model registry so the factory view and consumers do not have to
re-derive state from profiles on every request.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from quant_model_platform.db import COLLECTION_MODELS, COLLECTION_SHADOW_PROFILES
from quant_model_platform.timeutil import utcnow

log = logging.getLogger(__name__)


def _profile_model_id(profile: Dict[str, Any]) -> Optional[str]:
    return str(profile.get("model_id") or (profile.get("params") or {}).get("model_id") or "").strip() or None


def _profile_id(profile: Dict[str, Any]) -> str:
    return str(profile.get("profile_id") or (profile.get("params") or {}).get("profile_id") or "").strip()


def _upsert_embedded_profile(doc: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    profile_id = _profile_id(profile)
    source_job_id = profile.get("source_job_id") or (profile.get("params") or {}).get("source_job_id")
    rows = list(doc.get("shadow_profiles") or [])
    if profile_id and any(str(row.get("profile_id") or "") == profile_id for row in rows):
        rows = [
            {
                **row,
                "enabled": profile.get("enabled") is not False,
                "source_job_id": source_job_id or row.get("source_job_id"),
            }
            if str(row.get("profile_id") or "") == profile_id
            else row
            for row in rows
        ]
    elif profile_id:
        rows.append(
            {
                "profile_id": profile_id,
                "enabled": profile.get("enabled") is not False,
                "source_job_id": source_job_id,
                "created_at": profile.get("created_at"),
                "updated_at": profile.get("updated_at"),
            }
        )
    return rows


def sync_shadow_profile_states(collections: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply shadow/retired state transitions for profile enable/disable."""
    models = collections.get(COLLECTION_MODELS)
    profiles_coll = collections.get(COLLECTION_SHADOW_PROFILES)
    if models is None or profiles_coll is None:
        return []
    transitions: List[Dict[str, Any]] = []
    for profile in profiles_coll.find({}):
        model_id = _profile_model_id(profile)
        if not model_id:
            continue
        doc = models.find_one({"model_id": model_id})
        if not doc:
            continue
        enabled = profile.get("enabled") is not False
        current = str(doc.get("state") or "research")
        target = None
        if enabled and current in ("research", "validated"):
            target = "shadow"
        elif not enabled and current == "shadow":
            target = "retired"
        if not target:
            continue

        now = utcnow()
        embedded = _upsert_embedded_profile(doc, profile)
        reasons = ["shadow"] if target == "shadow" else []
        validated_flag = bool((doc.get("state_flags") or {}).get("validated"))
        updates = {
            "state": target,
            "accepted": bool(reasons),
            "accepted_reasons": reasons,
            "state_flags": {
                "live_active": False,
                "shadow_enabled": target == "shadow",
                "shadow_profile_any": bool(embedded),
                "validated": validated_flag,
            },
            "shadow_profiles": embedded,
            "health": {**dict(doc.get("health") or {}), "active": target == "shadow"},
            "updated_at": now,
            "state_history": list(doc.get("state_history") or [])
            + [
                {
                    "from": current,
                    "to": target,
                    "at": now,
                    "actor": "shadow_loop",
                    "reason": "shadow profile enabled" if enabled else "shadow profile disabled",
                    "evidence": {
                        "profile_id": _profile_id(profile),
                        "enabled": enabled,
                    },
                }
            ],
        }
        models.update_one({"_id": doc["_id"]}, {"$set": updates})
        transitions.append({"model_id": model_id, "from": current, "to": target})
        log.info("registry shadow sync model_id=%s %s -> %s", model_id, current, target)
    return transitions
