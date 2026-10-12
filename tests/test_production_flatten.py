from quant_model_platform.dataset.production import _flatten_score_doc, _normalize_trade_date


def test_normalize_trade_date():
    assert _normalize_trade_date("2024-01-15") == "20240115"
    assert _normalize_trade_date(20240115) == "20240115"


def test_flatten_score_doc():
    row = _flatten_score_doc(
        {
            "symbol": "000001.SZ",
            "score_date": "20240102",
            "growth_score": 0.5,
            "cycle_score": 0.4,
            "value_score": 0.3,
            "fundamental_score": 0.6,
        }
    )
    assert row["ts_code"] == "000001.SZ"
    assert row["trade_date"] == "20240102"
    assert row["quality_score"] == 0.6
