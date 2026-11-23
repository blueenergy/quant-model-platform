from quant_model_platform.db import (
    COLLECTION_MODELS,
    COLLECTION_PREDICTION_SETS,
    COLLECTION_PREDICTIONS,
)
from quant_model_platform.shadow.health import (
    refresh_model_health,
    refresh_prediction_set_registry,
)


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
        model_id = query.get("_id")
        if model_id is None:
            for clause in query.get("$or") or []:
                if clause.get("_id"):
                    model_id = clause["_id"]
                    break
        doc = self.docs.get(model_id)
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


class _Sets:
    def __init__(self, docs):
        self.docs = docs

    def find(self, query):
        return list(self.docs)

    def update_one(self, query, update):
        for doc in self.docs:
            if doc.get("_id") == query.get("_id"):
                doc.update(update.get("$set") or {})
                return


class _SetPreds:
    def __init__(self, rows):
        self.rows = rows

    def distinct(self, field, query=None):
        set_id = (query or {}).get("prediction_set_id")
        return sorted({row[field] for row in self.rows if row.get("prediction_set_id") == set_id})

    def count_documents(self, query):
        if "prediction_set_id" not in (query or {}):
            return 0
        set_id = query["prediction_set_id"]
        return sum(1 for row in self.rows if row.get("prediction_set_id") == set_id)

    def aggregate(self, pipeline):
        set_id = pipeline[0]["$match"]["prediction_set_id"]
        groups = {}
        for row in self.rows:
            if row.get("prediction_set_id") != set_id:
                continue
            key = (row.get("source"), row.get("window"))
            group = groups.setdefault(
                key,
                {
                    "_id": {"source": row.get("source"), "window": row.get("window")},
                    "first_trade_date": row["trade_date"],
                    "last_trade_date": row["trade_date"],
                    "prediction_count": 0,
                    "dates": set(),
                },
            )
            group["first_trade_date"] = min(group["first_trade_date"], row["trade_date"])
            group["last_trade_date"] = max(group["last_trade_date"], row["trade_date"])
            group["prediction_count"] += 1
            group["dates"].add(row["trade_date"])
        return list(groups.values())


def test_prediction_set_registry_refresh_updates_stats_and_model_summary():
    sets = _Sets(
        [
            {
                "_id": "ps-1",
                "prediction_set_id": "ps-1",
                "model_id": "m1",
                "kind": "research",
                "status": "active",
                "window": "expanding",
            }
        ]
    )
    preds = _SetPreds(
        [
            {"prediction_set_id": "ps-1", "trade_date": "20260101", "source": "replay_v24", "window": "expanding"},
            {"prediction_set_id": "ps-1", "trade_date": "20260201", "source": "replay_v24", "window": "expanding"},
        ]
    )
    models = _Models([{"_id": "m1", "model_id": "m1", "state": "research", "health": {}}])

    summary = refresh_prediction_set_registry(
        {
            COLLECTION_PREDICTION_SETS: sets,
            COLLECTION_PREDICTIONS: preds,
            COLLECTION_MODELS: models,
        }
    )

    assert summary["sets"] == 1
    assert summary["missing_rows"] == 0
    assert summary["overlaps"] == 0
    assert sets.docs[0]["first_trade_date"] == "20260101"
    assert sets.docs[0]["last_trade_date"] == "20260201"
    assert sets.docs[0]["date_count"] == 2
    assert sets.docs[0]["prediction_count"] == 2
    summary_row = models.docs["m1"]["prediction_sets"][0]
    assert summary_row["prediction_set_id"] == "ps-1"
    assert summary_row["prediction_count"] == 2
