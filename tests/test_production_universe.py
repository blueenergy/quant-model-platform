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


def test_winsorize_features_clips_each_date():
    from quant_model_platform.dataset.universe import winsorize_features

    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 5 + ["20240103"] * 5,
            "ts_code": list("abcde") * 2,
            "x": [1.0, 2.0, 3.0, 4.0, 100.0, 10.0, 20.0, 30.0, 40.0, 1000.0],
        }
    )
    got = winsorize_features(frame, ["x"], quantile=0.01)
    assert got.loc[4, "x"] < 100.0
    assert got.loc[9, "x"] < 1000.0
    # The interior values are untouched; with only five points the 1% quantile
    # clips the minimum slightly upward, so only the interior is asserted.
    assert got.loc[1, "x"] == 2.0
    assert got.loc[6, "x"] == 20.0