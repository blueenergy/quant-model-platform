import pandas as pd

from quant_model_platform.dataset.labels import forward_open_to_open_return


def test_next_open_forward_return_uses_future_opens():
    prices = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "ts_code": "000001.SZ", "open": 10.0},
            {"trade_date": "2024-01-03", "ts_code": "000001.SZ", "open": 11.0},
            {"trade_date": "2024-01-04", "ts_code": "000001.SZ", "open": 12.0},
            {"trade_date": "2024-01-05", "ts_code": "000001.SZ", "open": 13.0},
        ]
    )
    ret = forward_open_to_open_return(prices, horizon=1)
    # Signal on 2024-01-02 -> enter 11, exit open after 1 day from entry = 12 -> 12/11-1
    assert abs(ret.iloc[0] - (12.0 / 11.0 - 1.0)) < 1e-9
