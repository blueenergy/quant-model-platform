"""Tests for the alternative-channel factors (B6e-2).

Hand-computed cases pin the two semantics that would otherwise fail silently:
the overhang window (``ann_date <= t < float_date <= t + 60d``) and the
analyst revision's PIT direction (a report published later must never move an
earlier value).
"""
import numpy as np
import pandas as pd
import pytest

from quant_model_platform.dataset.alt_factors import (
    load_analyst_revision,
    load_chip_factors,
    load_unlock_overhang,
)
from quant_model_platform.dataset.production import _attach_alt_factors


class FakeCollection:
    """Applies the $in / $gte / $lte parts of the query, so a test can pin the
    loader's date bounds (a fake that ignores them cannot catch a query bug)."""

    def __init__(self, docs):
        self.docs = docs

    def find(self, query, projection=None):
        out = self.docs
        for field, cond in (query or {}).items():
            if not isinstance(cond, dict):
                continue
            if "$in" in cond:
                allowed = set(cond["$in"])
                out = [d for d in out if d.get(field) in allowed]
            else:
                def ok(doc, _f=field, _c=cond):
                    value = doc.get(_f)
                    if value is None:
                        return False
                    if "$gte" in _c and str(value) < str(_c["$gte"]):
                        return False
                    if "$lte" in _c and str(value) > str(_c["$lte"]):
                        return False
                    return True
                out = [d for d in out if ok(d)]
        return list(out)


class FakeDB:
    def __init__(self, colls):
        self.colls = colls

    def __getitem__(self, name):
        return FakeCollection(self.colls.get(name, []))


class FakeClient:
    def __init__(self, colls):
        self.colls = colls

    def __getitem__(self, db):
        return FakeDB(self.colls)


SYMBOL = "000001.SZ"


def unlock_frame(events, start, end):
    docs = [
        {"ts_code": SYMBOL, "ann_date": a, "float_date": f, "float_ratio": r}
        for a, f, r in events
    ]
    client = FakeClient({"share_float": docs})
    return load_unlock_overhang(
        client,
        data_db_name="quant_data",
        symbols=[SYMBOL],
        start_date=start,
        end_date=end,
    )


def value_on(frame, day, col="unlock_overhang_60d"):
    row = frame[frame["trade_date"] == day]
    return float(row[col].iloc[0]) if len(row) else np.nan


def test_overhang_counts_from_announcement_until_unlock():
    frame = unlock_frame([("20240101", "20240301", 5.0)], "20231230", "20240401")
    assert value_on(frame, "20231231") == 0.0          # before announcement
    assert value_on(frame, "20240101") == 5.0          # float <= t + 60d
    assert value_on(frame, "20240229") == 5.0
    assert value_on(frame, "20240301") == 0.0          # unlocked


def test_overhang_waits_until_unlock_enters_the_window():
    frame = unlock_frame([("20240101", "20240401", 3.0)], "20231230", "20240402")
    assert value_on(frame, "20240101") == 0.0          # 0401 > 0101 + 60d
    assert value_on(frame, "20240131") == 0.0          # 0401 > 0131 + 60d
    assert value_on(frame, "20240201") == 3.0          # 0401 <= 0201 + 60d
    assert value_on(frame, "20240331") == 3.0
    assert value_on(frame, "20240401") == 0.0


def test_overhang_ignores_unannounced_events():
    frame = unlock_frame([("20240301", "20240315", 5.0)], "20240101", "20240320")
    assert value_on(frame, "20240115") == 0.0          # inside window but not yet announced
    assert value_on(frame, "20240301") == 5.0
    assert value_on(frame, "20240315") == 0.0


def test_overhang_carries_announced_before_grid_start():
    frame = unlock_frame([("20231201", "20240201", 5.0)], "20240101", "20240202")
    assert value_on(frame, "20240101") == 5.0
    assert value_on(frame, "20240201") == 0.0


def test_overhang_sums_overlapping_events():
    frame = unlock_frame(
        [("20240101", "20240201", 2.0), ("20240110", "20240301", 3.0)],
        "20240101",
        "20240302",
    )
    assert value_on(frame, "20240115") == 5.0
    assert value_on(frame, "20240201") == 3.0          # first event expired
    assert value_on(frame, "20240229") == 3.0
    assert value_on(frame, "20240301") == 0.0


def analyst_frame(eps_events, start, end):
    docs = [
        {"ts_code": SYMBOL, "report_date": d, "eps": e}
        for d, e in eps_events
    ]
    client = FakeClient({"earnings_report_rc": docs})
    return load_analyst_revision(
        client,
        data_db_name="quant_data",
        symbols=[SYMBOL],
        start_date=start,
        end_date=end,
    )


def test_analyst_revision_hand_computed_value():
    frame = analyst_frame([("20240115", 1.0), ("20240310", 2.0)],
                          "20240401", "20240501")
    assert frame["trade_date"].min() == "20240401"     # only the panel window
    # level(0415) = (23*1.0 + 37*2.0) / 60; level(0215) = 1.0
    expected = (97.0 / 60.0 - 1.0) / 1.0
    assert value_on(frame, "20240415", "analyst_rev_60d") == pytest.approx(expected, rel=1e-3)


def test_analyst_revision_seeds_from_reports_before_the_lookback():
    # The only report predates the 130-day lookback of a 2024-01-01 start;
    # the forward-fill must still seed from it instead of opening with NaN.
    frame = analyst_frame([("20230110", 1.0)], "20240101", "20240201")
    assert value_on(frame, "20240115", "analyst_rev_60d") == 0.0


def test_analyst_revision_is_pit_a_report_cannot_move_the_past():
    base = analyst_frame([("20240115", 1.0), ("20240310", 2.0)],
                         "20240401", "20240501")
    shifted = analyst_frame([("20240115", 1.0), ("20240310", 2.0), ("20240420", 9.0)],
                            "20240401", "20240501")
    for day in ("20240401", "20240415", "20240419"):
        assert (value_on(base, day, "analyst_rev_60d")
                == value_on(shifted, day, "analyst_rev_60d")) or (
                np.isnan(value_on(base, day, "analyst_rev_60d"))
                and np.isnan(value_on(shifted, day, "analyst_rev_60d")))


def chip_frame(rows, start, end):
    docs = [
        {"symbol": s, "trade_date": d, **vals} for s, d, vals in rows
    ]
    client = FakeClient({"cyq_perf": docs})
    symbols = sorted({s for s, _, _ in rows}) or [SYMBOL]
    return load_chip_factors(
        client,
        data_db_name="quant_data",
        symbols=symbols,
        start_date=start,
        end_date=end,
    )


def test_chip_factors_hand_computed():
    # 000001.SZ on 20261016 as stored by the backfill: 10.0 / 11.0 / 11.6, avg 11.0
    frame = chip_frame(
        [(SYMBOL, "20261016",
          {"cost_15pct": 10.0, "cost_50pct": 11.0,
           "cost_85pct": 11.6, "weight_avg": 11.0})],
        "20261001", "20261031",
    )
    assert value_on(frame, "20261016", "cyq_conc") == pytest.approx(1.6 / 11.0, rel=1e-4)
    assert value_on(frame, "20261016", "cyq_skew") == pytest.approx(0.6, rel=1e-4)


def test_chip_skew_guards_zero_denominator():
    frame = chip_frame(
        [(SYMBOL, "20261016",
          {"cost_15pct": 11.0, "cost_50pct": 11.0,
           "cost_85pct": 12.0, "weight_avg": 11.0})],
        "20261001", "20261031",
    )
    assert np.isnan(value_on(frame, "20261016", "cyq_skew"))
    assert value_on(frame, "20261016", "cyq_conc") == pytest.approx(1.0 / 11.0, rel=1e-4)


def test_chip_conc_guards_nonpositive_weight_avg():
    frame = chip_frame(
        [(SYMBOL, "20261016",
          {"cost_15pct": 10.0, "cost_50pct": 11.0,
           "cost_85pct": 11.6, "weight_avg": 0.0})],
        "20261001", "20261031",
    )
    assert np.isnan(value_on(frame, "20261016", "cyq_conc"))
    assert value_on(frame, "20261016", "cyq_skew") == pytest.approx(0.6, rel=1e-4)


def test_attach_fills_unlock_zero_and_keeps_analyst_nan():
    panel = pd.DataFrame({
        "trade_date": ["20240101", "20240101"],
        "ts_code": [SYMBOL, "000002.SZ"],
        "analyst_rev_60d": [np.nan, np.nan],
    })
    client = FakeClient({
        "share_float": [{"ts_code": SYMBOL, "ann_date": "20240101",
                         "float_date": "20240201", "float_ratio": 5.0}],
        "earnings_report_rc": [],
    })
    out = _attach_alt_factors(
        client, panel, data_db_name="quant_data",
        symbols=[SYMBOL, "000002.SZ"], start_date="20240101", end_date="20240102",
    )
    assert len(out) == len(panel)                       # no fan-out
    assert out.loc[out["ts_code"] == SYMBOL, "unlock_overhang_60d"].iloc[0] == 5.0
    assert out.loc[out["ts_code"] == "000002.SZ", "unlock_overhang_60d"].iloc[0] == 0.0
    assert out["analyst_rev_60d"].isna().all()
