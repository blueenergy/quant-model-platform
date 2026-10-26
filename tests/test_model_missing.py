"""A missing feature must be visible to the fit, not quietly zero-filled.

``_feature_matrix`` used to ``fillna(0.0)`` before the callers' all-finite
mask, which made that mask unconditionally true -- so every row trained with
"missing" encoded as the number zero. With only the four score columns, which
the panel fills 100%, nothing showed. A raw factor filled at ~98% makes it
bite: ``log_market_cap`` of 0 is roughly thirteen standard deviations below
the mean, which is a leverage point, not a neutral value.
"""

import numpy as np
import pandas as pd
import pytest

from quant_model_platform.training.models import (
    _feature_matrix,
    predict_model,
    predict_sklearn_linear,
    train_model,
    train_ridge,
    train_rule_baseline,
)

SCORES = ["growth_score", "cycle_score", "value_score", "quality_score"]


def _frame(rows):
    """rows: list of (ts_code, score, label, momentum)."""
    return pd.DataFrame(
        {
            "trade_date": ["20240102"] * len(rows),
            "ts_code": [r[0] for r in rows],
            "growth_score": [r[1] for r in rows],
            "cycle_score": [r[1] * 0.5 for r in rows],
            "value_score": [100.0 - r[1] for r in rows],
            "quality_score": [50.0] * len(rows),
            "momentum_20d": [r[3] for r in rows],
            "label_fwd_20d": [r[2] for r in rows],
        }
    )


def _rows(n=12, momentum=0.01):
    return [(f"{i:06d}.SZ", float(i * 5 + 10), float(i) * 0.001, momentum) for i in range(n)]


# --- column selection --------------------------------------------------------


def test_a_column_that_is_entirely_nan_is_dropped_not_zero_filled():
    frame = _frame(_rows())
    frame["log_market_cap"] = np.nan
    frame["pe_ttm"] = np.nan

    x, cols = _feature_matrix(frame, [*SCORES, "log_market_cap", "pe_ttm"])

    assert cols == SCORES
    assert x.shape == (len(frame), len(SCORES))


def test_a_column_that_is_absent_from_the_frame_is_dropped():
    frame = _frame(_rows())

    _, cols = _feature_matrix(frame, [*SCORES, "roe", "revenue_yoy"])

    assert cols == SCORES


def test_the_baseline_uses_the_same_column_rule_as_the_model():
    """Two rules would drift apart; both must ignore the dead columns."""
    frame = _frame(_rows())
    frame["log_market_cap"] = np.nan

    baseline = train_rule_baseline(frame, [*SCORES, "log_market_cap"])

    assert baseline.feature_names == SCORES


# --- fitting really excludes incomplete rows ---------------------------------


def test_a_row_with_a_missing_feature_is_excluded_from_the_fit():
    """The fit with a NaN row must equal the fit with that row deleted."""
    rows = _rows()
    complete = _frame(rows)
    complete["momentum_20d"] = [0.01 * i for i in range(len(rows))]

    holed = complete.copy()
    holed.loc[3, "momentum_20d"] = np.nan

    dropped = complete.drop(index=3).reset_index(drop=True)

    features = [*SCORES, "momentum_20d"]
    with_hole = train_ridge(holed, features, "label_fwd_20d")
    without_row = train_ridge(dropped, features, "label_fwd_20d")

    np.testing.assert_allclose(
        with_hole.artifact["model"].coef_, without_row.artifact["model"].coef_
    )
    np.testing.assert_allclose(
        with_hole.artifact["scaler"].mean_, without_row.artifact["scaler"].mean_
    )


def test_the_extreme_value_of_a_hole_would_have_changed_the_fit():
    """Positive control for the test above: 0.0 is not a harmless stand-in."""
    rows = _rows()
    complete = _frame(rows)
    complete["momentum_20d"] = [0.01 * i for i in range(len(rows))]

    holed = complete.copy()
    holed.loc[3, "momentum_20d"] = np.nan
    zero_filled = complete.copy()
    zero_filled.loc[3, "momentum_20d"] = 0.0

    features = [*SCORES, "momentum_20d"]
    a = train_ridge(holed, features, "label_fwd_20d").artifact["model"].coef_
    b = train_ridge(zero_filled, features, "label_fwd_20d").artifact["model"].coef_

    assert not np.allclose(a, b)


# --- scoring still covers every row ------------------------------------------


def test_a_row_missing_an_input_still_gets_a_finite_score():
    frame = _frame(_rows())
    frame["momentum_20d"] = [0.01 * i for i in range(len(frame))]
    holed = frame.copy()
    holed.loc[2, "momentum_20d"] = np.nan

    model = train_ridge(frame, [*SCORES, "momentum_20d"], "label_fwd_20d")
    pred = predict_sklearn_linear(holed, model)

    assert pred.notna().all()
    assert np.isfinite(pred.to_numpy()).all()


def test_a_missing_input_is_imputed_at_the_training_mean():
    """Imputation must be z=0, i.e. the feature contributes nothing."""
    frame = _frame(_rows())
    frame["momentum_20d"] = [0.01 * i for i in range(len(frame))]
    model = train_ridge(frame, [*SCORES, "momentum_20d"], "label_fwd_20d")
    mean = float(frame["momentum_20d"].mean())

    holed = frame.copy()
    holed.loc[2, "momentum_20d"] = np.nan
    at_mean = frame.copy()
    at_mean.loc[2, "momentum_20d"] = mean

    assert predict_sklearn_linear(holed, model).iloc[2] == pytest.approx(
        predict_sklearn_linear(at_mean, model).iloc[2]
    )


# --- scoring keeps the artifact width when today's panel drifts ---------------


def test_scoring_keeps_a_column_that_is_empty_in_todays_snapshot():
    """An all-NaN feature in today's panel must be imputed, not dropped.

    The live shadow loop crashed on exactly this: a model trained with a
    populated column met a snapshot where the column was empty, the matrix
    shrank from 13 to 11 columns, and scaler.mean_ could not broadcast.
    """
    train = _frame(_rows())
    train["momentum_20d"] = [0.01 * i for i in range(len(train))]
    train["log_market_cap"] = [10.0 + i for i in range(len(train))]
    model = train_ridge(train, [*SCORES, "momentum_20d", "log_market_cap"], "label_fwd_20d")

    today = _frame(_rows())
    today["momentum_20d"] = [0.01 * i for i in range(len(today))]
    today["log_market_cap"] = np.nan
    at_mean = today.copy()
    at_mean["log_market_cap"] = float(train["log_market_cap"].mean())

    pred = predict_sklearn_linear(today, model)
    assert np.isfinite(pred.to_numpy()).all()
    pd.testing.assert_series_equal(pred, predict_sklearn_linear(at_mean, model))


def test_scoring_survives_a_column_absent_from_todays_snapshot():
    """Same as above, but the column is missing entirely, not just all-NaN."""
    train = _frame(_rows())
    train["momentum_20d"] = [0.01 * i for i in range(len(train))]
    train["log_market_cap"] = [10.0 + i for i in range(len(train))]
    model = train_ridge(train, [*SCORES, "momentum_20d", "log_market_cap"], "label_fwd_20d")

    # log_market_cap is absent from today's frame entirely (not just all-NaN).
    today = _frame(_rows())
    today["momentum_20d"] = [0.01 * i for i in range(len(today))]
    at_mean = today.copy()
    at_mean["log_market_cap"] = float(train["log_market_cap"].mean())

    pred = predict_sklearn_linear(today, model)
    assert np.isfinite(pred.to_numpy()).all()
    pd.testing.assert_series_equal(pred, predict_sklearn_linear(at_mean, model))


def test_lightgbm_scoring_survives_an_empty_column_in_todays_snapshot():
    pytest.importorskip("lightgbm")
    from quant_model_platform.training.models import train_lightgbm

    train = _frame(_rows(n=40))
    train["momentum_20d"] = [0.01 * i for i in range(len(train))]
    train["log_market_cap"] = [10.0 + i for i in range(len(train))]
    model = train_lightgbm(
        train,
        [*SCORES, "momentum_20d", "log_market_cap"],
        "label_fwd_20d",
        {"num_boost_round": 5},
    )

    today = _frame(_rows(n=40))
    today["momentum_20d"] = [0.01 * i for i in range(len(today))]
    today["log_market_cap"] = np.nan

    pred = predict_model(today, model)
    assert np.isfinite(pred.to_numpy()).all()


def test_baseline_scoring_survives_an_absent_column():
    frame = _frame(_rows())
    frame["log_market_cap"] = [10.0 + i for i in range(len(frame))]
    wide = train_rule_baseline(frame, [*SCORES, "log_market_cap"])
    narrow = train_rule_baseline(frame, SCORES)
    today = frame.drop(columns=["log_market_cap"])

    pd.testing.assert_series_equal(predict_model(today, wide), predict_model(today, narrow))


# --- the regression pin ------------------------------------------------------


def test_dead_columns_do_not_move_the_fit():
    """The four scores are fully populated, so adding never-filled columns to
    the declared list must leave the model bit-identical. This is what lets the
    control job reproduce the pre-change IC."""
    frame = _frame(_rows())
    frame["momentum_20d"] = [0.01 * i for i in range(len(frame))]
    for dead in ("log_market_cap", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"):
        frame[dead] = np.nan

    declared = [
        *SCORES,
        "log_market_cap",
        "pe_ttm",
        "pb",
        "roe",
        "revenue_yoy",
        "profit_yoy",
        "momentum_20d",
    ]
    narrow = train_ridge(frame, [*SCORES, "momentum_20d"], "label_fwd_20d")
    wide = train_ridge(frame, declared, "label_fwd_20d")

    assert wide.feature_names == narrow.feature_names
    np.testing.assert_allclose(wide.artifact["model"].coef_, narrow.artifact["model"].coef_)
    pd.testing.assert_series_equal(
        predict_sklearn_linear(frame, wide), predict_sklearn_linear(frame, narrow)
    )


def test_the_baseline_score_is_unchanged_by_dead_columns():
    """An all-NaN column z-scores to NaN and is skipped by mean(axis=1), so the
    historical baseline IC was already the four-score average."""
    frame = _frame(_rows())
    for dead in ("log_market_cap", "pe_ttm", "roe"):
        frame[dead] = np.nan

    narrow = predict_model(frame, train_rule_baseline(frame, SCORES))
    wide = predict_model(frame, train_rule_baseline(frame, [*SCORES, "log_market_cap", "pe_ttm", "roe"]))

    pd.testing.assert_series_equal(wide, narrow)


# --- through the dispatcher --------------------------------------------------


def test_the_dispatcher_survives_a_panel_that_is_mostly_dead_columns():
    frame = _frame(_rows())
    frame["momentum_20d"] = [0.01 * i for i in range(len(frame))]
    for dead in ("log_market_cap", "pe_ttm", "pb", "roe", "revenue_yoy", "profit_yoy"):
        frame[dead] = np.nan

    model = train_model("ridge", frame, [*SCORES, "log_market_cap", "pe_ttm"], "label_fwd_20d")
    pred = predict_model(frame, model)

    assert pred.notna().all()
