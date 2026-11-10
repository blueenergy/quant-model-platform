from quant_model_platform.tools.all_a.calendar import trading_days


class _Collection:
    def __init__(self, docs):
        self.docs = list(docs)

    def find(self, query, projection=None):
        low = query["trade_date"]["$gte"]
        high = query["trade_date"]["$lte"]
        allowed = query["is_open"]["$in"]
        return [
            dict(doc)
            for doc in self.docs
            if doc["trade_date"] >= low
            and doc["trade_date"] <= high
            and doc.get("is_open") in allowed
        ]


class _Database:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        return self.collection


class _Client:
    def __init__(self, collection):
        self.database = _Database(collection)

    def __getitem__(self, name):
        return self.database


def test_trading_days_accepts_int_and_string_open_flags():
    client = _Client(
        _Collection(
            [
                {"trade_date": "20261106", "is_open": 1},
                {"trade_date": "20261107", "is_open": "0"},
                {"trade_date": "20261108", "is_open": "0"},
                {"trade_date": "20261109", "is_open": "1"},
                {"trade_date": "20261110", "is_open": 1},
            ]
        )
    )

    assert trading_days(client, "quant_data", start="20261101", end="20261130") == [
        "20261106",
        "20261109",
        "20261110",
    ]
