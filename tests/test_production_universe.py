"""Universe hygiene: ST, new listing, delisted, and suspended rows."""

import pandas as pd

from quant_model_platform.dataset.universe import (
    filter_panel_metadata,
    filter_panel_prices,
)


def _panel():
    return pd.DataFrame(
        {
            "trade_date": ["20180629"] * 5,
            "ts_code": [
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
            ],
            "name": ["平安银行", "*ST XX", "正常", "新股", "退市"],
        }
    )


def _info():
    return pd.DataFrame(
        {
            "ts_code": [
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
            ],
            "list_date": ["19910403", "19910403", "19910403", "20180601", "19910403"],
            "delist_date": [None, None, None, None, "20180601"],
        }
    )


def test_metadata_filter_drops_st_new_and_delisted():
    got = filter_panel_metadata(_panel(), stock_info=_info(), min_list_days=250)
    assert set(got["ts_code"]) == {"000001.SZ", "000003.SZ"}


def test_price_filter_drops_rows_without_a_traded_price():
    keys = pd.DataFrame(
        {
            "trade_date": ["20180629"] * 3,
            "ts_code": ["000001.SZ", "000003.SZ", "000004.SZ"],
        }
    )
    got = filter_panel_prices(_panel(), price_keys=keys)
    assert set(got["ts_code"]) == {"000001.SZ", "000003.SZ", "000004.SZ"}


def test_filters_keep_an_empty_frame_empty():
    empty = pd.DataFrame(columns=["trade_date", "ts_code", "name"])
    assert filter_panel_metadata(empty, stock_info=_info()).empty
    assert filter_panel_prices(empty, price_keys=pd.DataFrame()).empty