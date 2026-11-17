from quant_model_platform.db import COLLECTION_MODELS, COLLECTION_PREDICTIONS
from quant_model_platform.shadow.health import refresh_model_health


class _Preds:
    def __init__(self, rows_by_model):
        self.rows_by_model = rows_by_model

    def find_one(self, query, projection=None, sort=None):
        return {"trade_date": "20261116"}

    def distinct(self, field, query=None):
        model_id = (query or {}).get("model_id")
        return sorted({row["trade_date"] for row in self.rows_by_model.get(model_id, [])})

    def count_documents(self, query):
        model_id = (query or {}).get("model_id")
        return len(self.rows_by_model.get(model_id, []))

    def aggregate(self, pipeline):
        model_id = pipeline[0]["$match"]["model_id"]
        rows = {}
        for row in self.rows_by_model.get(model_id, []):
            rows.setdefault(row["trade_date"], []).append(row["score"])
        out = [
            {"_id": date, "n": len(scores), "mean_score": sum(scores) / len(scores)}
            for date, scores in rows.items()
        ]
        return list(reversed(sorted(out, key=lambda item: item["_id"])))


class _Models:
    def __init__(self, docs):
        self.docs = {doc["model_id"]: doc for doc in docs}

    def find(self, query, projection=None):
        return list(self.docs.values())

    def update_one(self, query, update):
        doc = self.docs.get(query.get("_id"))
        if doc:
            doc.update(update.get("$set") or {})


def test_health_refresh_marks_current_stale_and_missing():
    preds = _Preds(
        {
            "m-live": [
                {"trade_date": "20261116", "score": 0.1},
                {"trade_date": "20261113", "score": 0.2},
            ],
            "m-shadow": [
                {"trade_date": "20261113", "score": 0.3},
            ],
        }
    )
    models = _Models(
        [
            {"_id": "m-live", "model_id": "m-live", "state": "live", "health": {}},
            {"_id": "m-shadow", "model_id": "m-shadow", "state": "shadow", "health": {}},
            {"_id": "m-research", "model_id": "m-research", "state": "research", "health": {}},
        ]
    )
    summary = refresh_model_health(
        {
            COLLECTION_MODELS: models,
            COLLECTION_PREDICTIONS: preds,
        }
    )
    assert summary["active_stale"] == 1
    assert models.docs["m-live"]["health"]["status"] == "current"
    assert models.docs["m-live"]["health"]["active"] is True
    assert models.docs["m-shadow"]["health"]["status"] == "stale"
    assert models.docs["m-shadow"]["health"]["stale_days"] == 3
    assert models.docs["m-research"]["health"]["status"] == "no_predictions"
