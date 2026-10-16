"""Production prices must come back forward adjusted.

A raw price turns every split or dividend inside a forward window into a fake
return; the platform's label has to agree with ``ScoreBacktester``'s
``adjust="hfq"`` opens, and the trailing raw factors need the same treatment
on ``close``.
"""

import pandas as pd
import pytest

from quant_model_platform.dataset.labels import forward_open_to_open_return
from quant_model_platform.dataset.production import (
    _attach_raw_factors,
    load_daily_basic,
    load_price_history,
)
from quant_model_platform.dataset.raw_factors import RAW_FEATURE_NAMES

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


class _TsCodeCollection:
    """``financial_daily_basic`` is indexed on ``ts_code``, not ``symbol``.

    Records every query so a test can assert which key the loader used: the
    wrong one still returns rows here and merely full-scans in production.
    """

    def __init__(self, docs):
        self._docs = docs
        self.queries = []

    def find(self, query, projection=None):
        self.queries.append(query)
        codes = query["ts_code"]["$in"]
        low, high = query["trade_date"]["$gte"], query["trade_date"]["$lte"]
        return [
            {k: v for k, v in doc.items() if not projection or k in projection}
            for doc in self._docs
            if doc["ts_code"] in codes and low <= doc["trade_date"] <= high
        ]


class _FakeDb:
    def __init__(self, collections):
        self._collections = collections

    def __getitem__(self, name):
        collection = self._collections[name]
        return collection if hasattr(collection, "find") else _FakeCollection(collection)


class _FakeClient:
    def __init__(self, collections):
        self._collections = collections

    def __getitem__(self, name):
        assert name == DB
        return _FakeDb(self._collections)


def _client(*, prices, factors, basic=()):
    return _FakeClient(
        {
            "volume_price": prices,
            "stock_adj_factor": factors,
            "financial_daily_basic": _TsCodeCollection(list(basic)),
        }
    )


def _price(date, open_, symbol="A", close=None):
    return {"symbol": symbol, "trade_date": date, "open": open_, "close": open_ if close is None else close}


def _factor(date, adj_factor, symbol="A"):
    return {"symbol": symbol, "trade_date": date, "adj_factor": adj_factor}


def _basic(date, circ_mv, turnover_rate, ts_code="000001.SZ"):
    return {
        "ts_code": ts_code,
        "trade_date": date,
        "circ_mv": circ_mv,
        "turnover_rate": turnover_rate,
    }


def test_split_inside_the_forward_window_stops_looking_like_a_loss():
    prices = [
        _price("20240102", 10.0),
        _price("20240103", 10.0),
        _price("20240104", 5.0),
        _price("20240105", 5.0),
    ]
    factors = [_factor("20231220", 1.0), _factor("20240104", 2.0)]
    client = _client(prices=prices, factors=factors)

    loaded = load_price_history(
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

    loaded = load_price_history(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240103"
    )
    assert loaded["open"].tolist() == [20.0, 20.0]


def test_a_date_the_factor_feed_skipped_inherits_the_last_level():
    prices = [_price("20240102", 10.0), _price("20240103", 10.0), _price("20240104", 5.0)]
    factors = [_factor("20240102", 1.0), _factor("20240104", 2.0)]
    client = _client(prices=prices, factors=factors)

    loaded = load_price_history(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240104"
    )
    assert loaded["open"].tolist() == [10.0, 10.0, 10.0]


def test_a_duplicated_factor_key_does_not_fan_the_prices_out():
    prices = [_price("20240102", 10.0), _price("20240103", 10.0)]
    factors = [_factor("20240102", 2.0), _factor("20240102", 2.0)]
    client = _client(prices=prices, factors=factors)

    loaded = load_price_history(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240103"
    )
    assert loaded["open"].tolist() == [20.0, 20.0]


def test_symbols_are_adjusted_independently():
    prices = [_price("20240102", 10.0, "A"), _price("20240102", 10.0, "B")]
    factors = [_factor("20240102", 2.0, "A"), _factor("20240102", 1.0, "B")]
    client = _client(prices=prices, factors=factors)

    loaded = load_price_history(
        client, data_db_name=DB, symbols=["A", "B"], start_date="20240102", end_date="20240102"
    )
    assert dict(zip(loaded["ts_code"], loaded["open"])) == {"A": 20.0, "B": 10.0}


def test_without_any_factor_the_open_is_left_alone():
    prices = [_price("20240102", 10.0), _price("20240103", 10.0)]
    client = _client(prices=prices, factors=[])

    loaded = load_price_history(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240103"
    )
    assert loaded["open"].tolist() == [10.0, 10.0]


def test_close_is_adjusted_alongside_open():
    """Momentum reads close, so leaving it raw rediscovers the split bug."""
    prices = [
        _price("20240102", 10.0, close=10.2),
        _price("20240103", 5.0, close=5.1),
    ]
    factors = [_factor("20240102", 1.0), _factor("20240103", 2.0)]
    client = _client(prices=prices, factors=factors)

    loaded = load_price_history(
        client, data_db_name=DB, symbols=["A"], start_date="20240102", end_date="20240103"
    )

    assert loaded["open"].tolist() == [10.0, 10.0]
    assert loaded["close"].tolist() == [10.2, 10.2]
    # The intraday basis is a ratio of two adjusted prices, so it survives.
    assert (loaded["close"] / loaded["open"]).tolist() == pytest.approx([1.02, 1.02])


def test_symbols_keep_their_exchange_suffix():
    """stock_scores, volume_price and financial_daily_basic all key on the
    suffixed form; a silent format drift would make every join miss."""
    prices = [_price("20240102", 10.0, "600000.SH"), _price("20240102", 10.0, "000001.SZ")]
    client = _client(prices=prices, factors=[])

    loaded = load_price_history(
        client,
        data_db_name=DB,
        symbols=["600000.SH", "000001.SZ"],
        start_date="20240102",
        end_date="20240102",
    )

    assert sorted(loaded["ts_code"]) == ["000001.SZ", "600000.SH"]


def test_daily_basic_is_queried_by_ts_code_not_symbol():
    """The collection has ~14M rows and an index on ts_code only."""
    client = _client(prices=[], factors=[], basic=[_basic("20240102", 1000.0, 1.5)])
    collection = client[DB]["financial_daily_basic"]

    loaded = load_daily_basic(
        client, data_db_name=DB, symbols=["000001.SZ"], start_date="20240102", end_date="20240102"
    )

    assert collection.queries and "ts_code" in collection.queries[0]
    assert loaded["circ_mv"].tolist() == [1000.0]


def test_a_duplicated_daily_basic_key_does_not_fan_the_load_out():
    basic = [_basic("20240102", 1000.0, 1.5), _basic("20240102", 1000.0, 1.5)]
    client = _client(prices=[], factors=[], basic=basic)

    loaded = load_daily_basic(
        client, data_db_name=DB, symbols=["000001.SZ"], start_date="20240102", end_date="20240102"
    )

    assert len(loaded) == 1


# --- assembling the raw factors onto the panel -------------------------------


def _panel(dates, symbols=("000001.SZ",)):
    rows = [{"trade_date": d, "ts_code": s} for d in dates for s in symbols]
    frame = pd.DataFrame(rows)
    for name in RAW_FEATURE_NAMES:
        frame[name] = float("nan")
    return frame


def _history(dates, closes, symbol="000001.SZ"):
    return pd.DataFrame(
        {
            "trade_date": list(dates),
            "ts_code": symbol,
            "open": [float(c) for c in closes],
            "close": [float(c) for c in closes],
        }
    )


def test_attaching_factors_keeps_the_panel_row_count():
    dates = [f"2024{m:02d}{d:02d}" for m in range(1, 4) for d in range(1, 11)]
    panel = _panel(dates)
    prices = _history(dates, range(100, 100 + len(dates)))
    client = _client(prices=[], factors=[], basic=[])

    joined = _attach_raw_factors(
        client, panel, prices, data_db_name=DB, symbols=["000001.SZ"],
        start_date=dates[0], end_date=dates[-1],
    )

    assert len(joined) == len(panel)
    assert set(RAW_FEATURE_NAMES) <= set(joined.columns)


def test_attaching_factors_does_not_leave_suffixed_placeholder_columns():
    """The registry pre-creates each name as NaN; a naive merge would produce
    ``momentum_20d_x`` / ``momentum_20d_y`` and silently drop the real column."""
    dates = [f"2024{m:02d}{d:02d}" for m in range(1, 4) for d in range(1, 11)]
    panel = _panel(dates)
    prices = _history(dates, range(100, 100 + len(dates)))
    client = _client(prices=[], factors=[], basic=[])

    joined = _attach_raw_factors(
        client, panel, prices, data_db_name=DB, symbols=["000001.SZ"],
        start_date=dates[0], end_date=dates[-1],
    )

    assert not [c for c in joined.columns if c.endswith(("_x", "_y"))]
    assert pd.notna(joined["momentum_20d"].iloc[-1])


def test_warmup_rows_before_the_panel_do_not_add_panel_rows():
    """Prices load from before the panel starts so the first window is full.

    The leading rows match no panel row and must drop out of the join rather
    than extend the panel backwards.
    """
    warmup = [f"2023{11:02d}{d:02d}" for d in range(1, 26)]
    dates = [f"2024{m:02d}{d:02d}" for m in range(1, 4) for d in range(1, 11)]
    panel = _panel(dates)
    prices = _history(warmup + dates, range(50, 50 + len(warmup) + len(dates)))
    client = _client(prices=[], factors=[], basic=[])

    joined = _attach_raw_factors(
        client, panel, prices, data_db_name=DB, symbols=["000001.SZ"],
        start_date=warmup[0], end_date=dates[-1],
    )

    assert len(joined) == len(panel)
    assert joined["trade_date"].min() == dates[0]
    # 25 sessions of warm-up: the 20-session window is already full on day one.
    assert pd.notna(joined["momentum_20d"].iloc[0])


def test_a_duplicated_factor_key_is_loud_rather_than_a_doubled_panel():
    dates = [f"2024{m:02d}{d:02d}" for m in range(1, 4) for d in range(1, 11)]
    panel = _panel(dates)
    prices = _history(dates, range(100, 100 + len(dates)))
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    client = _client(prices=[], factors=[], basic=[])

    with pytest.raises(RuntimeError, match="fanned the panel out"):
        _attach_raw_factors(
            client, panel, prices, data_db_name=DB, symbols=["000001.SZ"],
            start_date=dates[0], end_date=dates[-1],
        )
