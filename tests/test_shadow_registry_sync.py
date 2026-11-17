from quant_model_platform.db import COLLECTION_MODELS, COLLECTION_SHADOW_PROFILES
from quant_model_platform.shadow.registry_sync import sync_shadow_profile_states


class _Profiles:
    def __init__(self, docs):
        self.docs = list(docs)

    def find(self, query):
        return list(self.docs)


class _Models:
    def __init__(self, docs):
        self.docs = {doc["model_id"]: doc for doc in docs}

    def find_one(self, query, projection=None):
        return self.docs.get(query.get("model_id"))

    def update_one(self, query, update):
        doc = self.docs.get(query.get("_id"))
        if doc:
            doc.update(update.get("$set") or {})


def _model(model_id, state):
    return {
        "_id": model_id,
        "model_id": model_id,
        "state": state,
        "accepted": state in ("live", "shadow"),
        "accepted_reasons": ["live"] if state == "live" else ["shadow"] if state == "shadow" else [],
        "state_flags": {"live_active": state == "live", "shadow_enabled": state == "shadow", "shadow_profile_any": state == "shadow", "validated": False},
        "shadow_profiles": [],
        "state_history": [],
        "health": {"active": state in ("live", "shadow")},
    }


def test_shadow_sync_enables_and_retires_profiles():
    profiles = _Profiles(
        [
            {"profile_id": "p1", "model_id": "m-research", "enabled": True},
            {"profile_id": "p2", "model_id": "m-shadow", "enabled": False},
            {"profile_id": "p3", "model_id": "m-live", "enabled": True},
        ]
    )
    models = _Models(
        [
            _model("m-research", "research"),
            _model("m-shadow", "shadow"),
            _model("m-live", "live"),
        ]
    )
    transitions = sync_shadow_profile_states(
        {
            COLLECTION_MODELS: models,
            COLLECTION_SHADOW_PROFILES: profiles,
        }
    )
    assert transitions == [
        {"model_id": "m-research", "from": "research", "to": "shadow"},
        {"model_id": "m-shadow", "from": "shadow", "to": "retired"},
    ]
    research = models.docs["m-research"]
    assert research["state"] == "shadow"
    assert research["accepted"] is True
    assert research["accepted_reasons"] == ["shadow"]
    assert research["state_history"][-1]["actor"] == "shadow_loop"
    retired = models.docs["m-shadow"]
    assert retired["state"] == "retired"
    assert retired["accepted"] is False
    assert models.docs["m-live"]["state"] == "live"
