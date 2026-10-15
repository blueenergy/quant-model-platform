"""Production opens must come back forward adjusted.

A raw open turns every split or dividend inside a forward window into a fake
return; the platform's label has to agree with ``ScoreBacktester``'s
``adjust="hfq"`` opens.
"""

import pandas as pd
import pytest

from quant_model_platform.dataset.labels import forward_open_to_open_return
from quant_model_platform.dataset.production import load_open_prices

DB = "quant_data"


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, query, projection=None):
        symbols = query["symbol"]["$in"]
        low, high = query["trade_date"]["$gte"], query["trade_date"]["$lte"]
        return [
            {k: v for k, v in doc.items() if not projection or k in projection}
            for doc in self._docs
            if doc["symbol"] in symbols and low <= doc["trade_date"] <= high
        ]


class _FakeDb:
    def __init__(self, collections):
        self._collections = collections

    def __getitem__(self, name):
        return _FakeCollection(self._collections[name])


class _FakeClient:
    def __init__(self, collections):
        self._collections = collections

    def __getitem__(self, name):
        assert name == DB
        return _FakeDb(self._collections)


def _client(*, prices, factors):
    return _FakeClient({"volume_price": prices, "stock_adj_factor": factors})


def _price(date, open_, symbol="A"):
    return {"symbol": symbol, "trade_date": date, "open": open_}


def _factor(date, adj_factor, symbol="A"):
    return {"symbol": symbol, "trade_date": date, "adj_factor": adj_factor}


def test_split_inside_the_forward_window_stops_looking_like_a_loss():
    prices = [
        _price("20240102", 10.0),
        _price("20240103", 10.0),
        _price("20240104", 5.0),
        _price("20240105", 5.0),
    ]
    factors = [_factor("20231220", 1.0), _factor("20240104", 2.0)]
    client = _client(prices=prices, factors=factors)

    loaded = load_open_prices(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240105"
    )
    assert loaded["open"].tolist() == [10.0, 10.0, 10.0, 10.0]

    # Entry on 0103, exit on 0104, the 2:1 split lands in between.
    raw = pd.DataFrame({"trade_date": ["20240102", "20240103", "20240104", "20240105"],
                        "ts_code": "A", "open": [10.0, 10.0, 5.0, 5.0]})
    assert forward_open_to_open_return(raw, horizon=1).iloc[0] == pytest.approx(-0.5)
    assert forward_open_to_open_return(loaded, horizon=1).iloc[0] == pytest.approx(0.0)


def test_factor_from_before_the_window_seeds_the_first_rows():
    prices = [_price("20240102", 10.0), _price("20240103", 10.0)]
    client = _client(prices=prices, factors=[_factor("20231220", 2.0)])

    loaded = load_open_prices(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240103"
    )
    assert loaded["open"].tolist() == [20.0, 20.0]


def test_a_date_the_factor_feed_skipped_inherits_the_last_level():
    prices = [_price("20240102", 10.0), _price("20240103", 10.0), _price("20240104", 5.0)]
    factors = [_factor("20240102", 1.0), _factor("20240104", 2.0)]
    client = _client(prices=prices, factors=factors)

    loaded = load_open_prices(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240104"
    )
    assert loaded["open"].tolist() == [10.0, 10.0, 10.0]


def test_a_duplicated_factor_key_does_not_fan_the_prices_out():
    prices = [_price("20240102", 10.0), _price("20240103", 10.0)]
    factors = [_factor("20240102", 2.0), _factor("20240102", 2.0)]
    client = _client(prices=prices, factors=factors)

    loaded = load_open_prices(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240103"
    )
    assert loaded["open"].tolist() == [20.0, 20.0]


def test_symbols_are_adjusted_independently():
    prices = [_price("20240102", 10.0, "A"), _price("20240102", 10.0, "B")]
    factors = [_factor("20240102", 2.0, "A"), _factor("20240102", 1.0, "B")]
    client = _client(prices=prices, factors=factors)

    loaded = load_open_prices(
        client, data_db_name=DB, symbols=["A", "B"], start_date="20240102", end_date="20240102"
    )
    assert dict(zip(loaded["ts_code"], loaded["open"])) == {"A": 20.0, "B": 10.0}


def test_without_any_factor_the_open_is_left_alone():
    prices = [_price("20240102", 10.0), _price("20240103", 10.0)]
    client = _client(prices=prices, factors=[])

    loaded = load_open_prices(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240103"
    )
    assert loaded["open"].tolist() == [10.0, 10.0]
