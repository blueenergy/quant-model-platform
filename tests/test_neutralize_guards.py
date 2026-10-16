"""A present-but-empty size column must announce itself before it activates.

Both neutralization paths test whether ``log_market_cap`` is *in* the frame,
which the production panel makes trivially true -- it pre-creates every registry
name as NaN. So the size branch reads as enabled while behaving as a plain
demean, and the caller has nothing to notice it by.

Nothing passes ``size_neutral`` today, so these tests pin the diagnostic only.
The numbers must not move: the reachable path (the default ``demean``) is
untouched.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from quant_model_platform.dataset.labels import MIN_SIZE_SAMPLES, neutralize_cross_section
from quant_model_platform.evaluation.metrics import industry_size_neutralize_scores


def _frame(caps, labels=None):
    n = len(caps)
    return pd.DataFrame(
        {
            "trade_date": ["20240102"] * n,
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "log_market_cap": caps,
            "label_raw": labels if labels is not None else [0.01 * i for i in range(n)],
        }
    )


def _score_frame(caps):
    n = len(caps)
    return pd.DataFrame(
        {
            "trade_date": ["20240102"] * n,
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "log_market_cap": caps,
            "score": [float(i) for i in range(n)],
        }
    )


def _warnings(caplog):
    return [str(r.message) for r in caplog.records if r.levelno >= logging.WARNING]


def _flat(result):
    """Values regardless of container.

    ``size_neutral`` returns one row per date instead of a flat Series -- a
    pre-existing defect on this unreachable path, found while adding the
    warning and deliberately not fixed here. Values are still comparable.
    """
    return np.asarray(result).ravel()


# --- the label side ----------------------------------------------------------


def test_a_present_but_empty_size_column_warns_before_degrading(caplog):
    with caplog.at_level(logging.WARNING):
        neutralize_cross_section(_frame([np.nan] * 8), "label_raw", mode="size_neutral")

    assert any("size_neutral" in m for m in _warnings(caplog)), "degraded silently"


def test_an_absent_size_column_warns_too(caplog):
    frame = _frame([np.nan] * 8).drop(columns=["log_market_cap"])

    with caplog.at_level(logging.WARNING):
        neutralize_cross_section(frame, "label_raw", mode="size_neutral")

    assert any("absent" in m for m in _warnings(caplog))


def test_an_empty_size_column_leaves_the_label_values_identical(caplog):
    """The reason this change is warnings-only: the output cannot move."""
    frame = _frame([np.nan] * 8)

    with caplog.at_level(logging.WARNING):
        with_empty = neutralize_cross_section(frame, "label_raw", mode="size_neutral")
    demeaned = neutralize_cross_section(frame, "label_raw", mode="demean")

    np.testing.assert_allclose(_flat(with_empty), _flat(demeaned))


def test_a_filled_size_column_residualizes_and_stays_quiet(caplog):
    """Positive control: with real data the branch must do its job."""
    caps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    frame = _frame(caps, labels=[2.0 * c for c in caps])

    with caplog.at_level(logging.WARNING):
        got = neutralize_cross_section(frame, "label_raw", mode="size_neutral")

    assert not _warnings(caplog)
    demeaned = neutralize_cross_section(frame, "label_raw", mode="demean")
    # The label is an exact multiple of size, so residualizing flattens it.
    assert np.abs(_flat(got)).max() < np.abs(_flat(demeaned)).max()


def test_a_thin_date_is_reported_by_how_many_dates_it_affects(caplog):
    """One thin date out of two -- the count is the signal, so it has to be right."""
    frame = _frame([np.nan] * 8)
    frame["trade_date"] = ["20240102"] * 6 + ["20240103"] * 2
    frame.loc[:5, "log_market_cap"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    with caplog.at_level(logging.WARNING):
        neutralize_cross_section(frame, "label_raw", mode="size_neutral")

    assert any("1 of 2 dates" in m for m in _warnings(caplog))
    assert MIN_SIZE_SAMPLES == 3


def test_the_default_mode_is_unaffected_by_any_of_this(caplog):
    """Both production call sites use the default, so this is the live path."""
    frame = _frame([np.nan] * 8)

    with caplog.at_level(logging.WARNING):
        got = neutralize_cross_section(frame, "label_raw", mode="demean")

    assert not _warnings(caplog)
    assert isinstance(got, pd.Series)
    assert got.iloc[0] == pytest.approx(-0.035)


# --- the score side ----------------------------------------------------------


def test_the_score_side_warns_on_a_present_but_empty_size_column(caplog):
    with caplog.at_level(logging.WARNING):
        industry_size_neutralize_scores(_score_frame([np.nan] * 6), "score")

    assert any("size residualization skipped" in m for m in _warnings(caplog))


def test_the_score_side_stays_quiet_when_size_is_usable(caplog):
    with caplog.at_level(logging.WARNING):
        industry_size_neutralize_scores(_score_frame([float(i) for i in range(6)]), "score")

    assert not _warnings(caplog)


def test_the_score_side_still_demeans_without_an_industry_column(caplog):
    """``industry`` is not in the registry, so this is the production shape."""
    with caplog.at_level(logging.WARNING):
        got = industry_size_neutralize_scores(_score_frame([np.nan] * 6), "score")

    assert np.mean(_flat(got)) == pytest.approx(0.0, abs=1e-12)
