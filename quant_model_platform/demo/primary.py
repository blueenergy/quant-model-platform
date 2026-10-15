"""The primary evidence: per-date IC and TOP-K excess over the day's own pool.

The contract (docs/DEMO_HS300_20D_TOP20.md section 5) ranks IC and the
pool-relative excess above the NAV, and the reason is statistical rather than
aesthetic: both use every labelled name that day, while the NAV uses only the
ten held. On the same 129 rebalance dates the 10-name NAV carries standard
errors wide enough to hide any effect this demo could hope to see, so it cannot
be the ruler the platform is judged by. This module is that ruler.

Three properties of the measurement are pinned here because they are easy to get
wrong and silent when wrong.

**Demeaning the label does not move an IC.** Pearson and Spearman are both
invariant under adding a constant, and a pool demean subtracts one constant per
date -- one constant *within* the group each IC is computed over. So a demeaned
IC and a raw IC are the same number, exactly, and "neutralized IC" is not a
second flavour of IC. Demeaning changes only level statistics: the pool mean,
the TOP-K excess, and a NAV measured against the pool. That matters here because
the demo's own price path returns raw ``exit/entry - 1`` while the platform's
``label_fwd_20d`` is demeaned per date; the two are therefore directly
comparable, and this module pins the identity with a test so a later reader does
not "fix" it into a real per-industry residualization, which *would* move the
IC.

**The pool is the day's cross-section, never the window's symbol union.** A
symbol with prices is not automatically a name that was in the index that day.
The panel spans index membership over the window, so the union carries names
that only joined later -- and membership follows a run-up -- which makes it a
basket nobody could have held. On this window that look-ahead is worth about
+6.5pp per period. The day's scored cross-section is the contract's "every
labelled name that day", so that is the benchmark; the union is reported beside
it as ``pool_return_wide`` so the gap is visible instead of assumed.

**The sample size is the number of dates, not the number of observations.**
Twenty-day forward returns overlap, so 2,591 daily ICs carry about as much
independent information as the ~130 rebalance dates they are made of. The
rebalance grid's own points do not overlap, which is why this module measures
there and why a plain t-test is valid without a Newey-West correction. The
observation count is printed next to ``effective_n`` for audit only; reading it
as a sample size would inflate every t by roughly the square root of the names
per date.

Pure module: dicts in, dicts out. It deliberately does not import
``demo.portfolio`` (which imports this one) nor ``demo.sources`` (which would
drag pymongo into the statistics).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from scipy import stats

from quant_model_platform.evaluation.metrics import MIN_IC_NAMES, cross_sectional_ic

#: z_{0.975} + z_{0.80}: 80% power at a two-sided 5%. Contract section 5.5
#: quotes the same convention as ``2.8016 * sd / sqrt(n)``.
MDE_Z_80 = 2.8016

TRADING_DAYS = 252.0

#: Track labels in report order; the strings ``portfolio.simulate_track`` writes.
TRACKS = ("baseline", "model", "control_random")

#: The benchmark concept: the equal-weight basket of the day's labelled names.
POOL_BASIS = "day_cross_section"
#: Which cross-section stood in for it, per date.
POOL_SOURCE_A = "baseline_cross_section"
POOL_SOURCE_B = "model_cross_section"
POOL_SOURCE_NONE = "none"

_COST_NOTE = (
    "IC is a cross-sectional correlation between a score and a forward return; cost acts on the book's "
    "turnover, not on a single name's return, so there is no such thing as a cost-adjusted IC. Costs are "
    "charged in topk_excess_net only."
)
_GRID_NOTE = (
    "One IC per rebalance date, not per trading day: a 20-day label overlaps itself, so daily ICs would "
    "add 20 numbers per independent observation. Consecutive dates are hold_days apart, so their return "
    "intervals touch and share no return, and a plain t-test is valid without a Newey-West correction. "
    "One crack remains: EXIT_LIMIT_DELAY_DAYS=5 can push an individual exit up to 5 sessions late, so a "
    "few names overlap the next period -- at most a quarter of one period, a second-order term."
)
_POWER_NOTE = (
    "effective_n counts dates, because that is what carries independent information. pooled_observations "
    "is the baseline's (name, date) IC inputs and is printed for audit only: treating it as a sample size "
    "would inflate every t by roughly sqrt(names per date), i.e. about 17x on hs300. mde_80 is the "
    "smallest effect a series of this length could have detected 80% of the time."
)
_JOIN_NOTE = (
    "Baseline scored names that found a forward return, over the dates that had a cross-section. Scores "
    "key on stock_scores.symbol and labels on volume_price.symbol; if those vocabularies ever diverge this "
    "rate goes to zero and every IC becomes NaN, which otherwise reads as 'no data' rather than 'harness "
    "broken'. The model's and control's own counts are under tracks[*].dates_without_a_join."
)


def _by_date(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Re-key on the 8-character date; callers pass maps of unknown key width."""
    return {str(key)[:8]: value for key, value in (mapping or {}).items()}


def _symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("ts_code") or row.get("symbol") or "").strip()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean(value: Any) -> Any:
    """Recursively replace non-finite floats with None.

    ``json.dumps`` writes a bare ``NaN`` for ``float("nan")`` -- invalid JSON
    that Python reads straight back, so neither a round-trip test nor
    ``json.loads`` would catch it and only a strict consumer would break.
    """
    if isinstance(value, Mapping):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def pool_symbols_by_date(
    scores_a: Mapping[str, Sequence[Mapping[str, Any]]],
    scores_b: Mapping[str, Sequence[Mapping[str, Any]]],
    signal_dates: Sequence[str],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """The day's own cross-section, per date, with the source that supplied it.

    Track A's cross-section is the validator's own view of the day's membership,
    so it is preferred; a date it refused to rank (below the name floor, or a
    single distinct score) falls back to the model's, which exists on every
    signal date. Both are same-day sets -- the point is only that the fallback is
    never the window's union.
    """
    by_a, by_b = _by_date(scores_a), _by_date(scores_b)
    symbols: Dict[str, List[str]] = {}
    sources: Dict[str, str] = {}
    for day in signal_dates:
        key = str(day)[:8]
        for mapping, label in ((by_a, POOL_SOURCE_A), (by_b, POOL_SOURCE_B)):
            found = list(dict.fromkeys(s for s in (_symbol(row) for row in mapping.get(key) or ()) if s))
            if found:
                symbols[key], sources[key] = found, label
                break
        else:
            symbols[key], sources[key] = [], POOL_SOURCE_NONE
    return symbols, sources


def ic_frame(
    scores_by_date: Mapping[str, Sequence[Mapping[str, Any]]],
    forward_returns: Mapping[str, Mapping[str, float]],
    dates: Sequence[str],
) -> pd.DataFrame:
    """Long ``[trade_date, score, label]`` table for :func:`cross_sectional_ic`.

    Built once per track so the IC comes out of the platform's single IC
    primitive rather than a second correlation implementation that can drift
    from it. Scores that will not parse are coerced to NaN and dropped by the
    same rule the primitive applies to everything else.
    """
    labels_by_date = _by_date(forward_returns)
    scores = _by_date(scores_by_date)
    records: List[Dict[str, Any]] = []
    for day in dates:
        key = str(day)[:8]
        labels = labels_by_date.get(key) or {}
        if not labels:
            continue
        for row in scores.get(key) or ():
            symbol = _symbol(row)
            if not symbol or symbol not in labels:
                continue
            records.append({"trade_date": key, "score": row.get("score"), "label": labels[symbol]})
    frame = pd.DataFrame.from_records(records, columns=["trade_date", "score", "label"])
    if frame.empty:
        return frame
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    return frame


def _usable(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows an IC is actually computed on: both sides finite.

    The same rule as the ``dropna`` inside :func:`cross_sectional_ic`, so the
    name counts reported beside an IC describe the IC that was computed.
    """
    if frame.empty:
        return frame
    return frame.dropna(subset=["score", "label"])


def _joined_by_date(frame: pd.DataFrame) -> Dict[str, int]:
    if frame.empty:
        return {}
    counted = frame.groupby("trade_date", sort=False).size()
    return {str(day): int(count) for day, count in counted.items()}


def _series_by_date(series: pd.Series) -> Dict[str, Optional[float]]:
    return {str(day): _finite(value) for day, value in series.items()}


def daily_metrics(
    scores_by_track: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    forward_returns: Mapping[str, Mapping[str, float]],
    track_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    pool_symbols: Mapping[str, Sequence[str]],
    pool_sources: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """One row per ``(trade_date, track)``: the single source for CSV and JSON.

    The grid comes from ``track_rows`` -- the equity engine's own rebalance rows
    -- rather than from a date list, so a date the engine dropped cannot appear
    here with an IC and no return.

    ``topk_mean`` is copied from the track's own ``gross_return`` rather than
    recomputed from its holdings, so this file cannot disagree with
    ``rebalances.csv`` about what a period earned. A carried date has no scoring
    event, so it has no IC: inventing one from the held names would report a
    ranking that never happened.
    """
    labels_by_date = _by_date(forward_returns)
    rows: List[Dict[str, Any]] = []
    for track in TRACKS:
        track_rows_for = list(track_rows.get(track) or ())
        scored_dates = [str(row["trade_date"]) for row in track_rows_for if row.get("cross_section") != "carried"]
        frame = _usable(ic_frame(scores_by_track.get(track) or {}, forward_returns, scored_dates))
        joined = _joined_by_date(frame)
        rank_ic: Dict[str, Optional[float]] = {}
        pearson_ic: Dict[str, Optional[float]] = {}
        if len(frame):
            rank_ic = _series_by_date(cross_sectional_ic(frame, "score", "label", min_names=MIN_IC_NAMES))
            pearson_ic = _series_by_date(
                cross_sectional_ic(frame, "score", "label", method="pearson", min_names=MIN_IC_NAMES)
            )
        for row in track_rows_for:
            day = str(row.get("trade_date"))[:8]
            labels = labels_by_date.get(day) or {}
            pool = list(pool_symbols.get(day) or ())
            priced_pool = [float(labels[symbol]) for symbol in pool if labels.get(symbol) is not None]
            wide = [float(value) for value in labels.values() if value is not None]
            pool_return = (sum(priced_pool) / len(priced_pool)) if priced_pool else None
            wide_return = (sum(wide) / len(wide)) if wide else None
            topk_mean = _finite(row.get("gross_return"))
            cost = _finite(row.get("cost"))
            excess_gross = None if topk_mean is None or pool_return is None else topk_mean - pool_return
            excess_net = None if excess_gross is None or cost is None else excess_gross - cost
            rows.append(
                _clean(
                    {
                        "trade_date": day,
                        "track": track,
                        "pool_basis": pool_sources.get(day) or POOL_SOURCE_NONE,
                        "n_pool": len(priced_pool),
                        "pool_return": pool_return,
                        "pool_return_wide": wide_return,
                        "n_wide": len(wide),
                        "n_scored": int(row.get("n_scored") or 0),
                        "n_joined": joined.get(day, 0),
                        "n_priced": int(row.get("n_priced") or 0),
                        "pricing": row.get("pricing"),
                        "carried": 1 if row.get("cross_section") == "carried" else 0,
                        "ic_rank": rank_ic.get(day),
                        "ic_pearson": pearson_ic.get(day),
                        "topk_mean": topk_mean,
                        "excess_gross": excess_gross,
                        "cost": cost,
                        "excess_net": excess_net,
                    }
                )
            )
    return rows


def _sd(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    return (sum(known) / len(known)) if known else None


def _describe(values: Sequence[Any], *, hold_days: int, unit: str) -> Dict[str, Any]:
    """Mean, dispersion and the power of the sample that produced them.

    ``ic_ir`` is the platform's ``mean / sd`` (``evaluation.metrics.summarize_ic``)
    and is not annualized. Each family then carries the one extra key that is
    meaningful for it: a ratio scales with the square root of the period count
    and a return with the count itself, and mixing the two conventions is a
    factor of about 3.5 at a 20-day horizon.
    """
    series = [float(value) for value in values]
    count = len(series)
    if not count:
        empty: Dict[str, Any] = {
            "mean": None,
            "sd": None,
            "ic_ir": None,
            "t_stat": None,
            "p_value": None,
            "n_dates": 0,
            "mde_80": None,
        }
        empty["ic_ir_annualized" if unit == "ic" else "annualized_mean"] = None
        return empty
    mean = sum(series) / count
    spread = _sd(series)
    error = (spread / math.sqrt(count)) if spread else None
    t_stat = (mean / error) if error else None
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=count - 1)) if t_stat is not None else None
    block: Dict[str, Any] = {
        "mean": mean,
        "sd": spread,
        "ic_ir": (mean / spread) if spread else None,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_dates": count,
        "mde_80": (MDE_Z_80 * error) if error else None,
    }
    if unit == "ic":
        block["ic_ir_annualized"] = (
            block["ic_ir"] * math.sqrt(TRADING_DAYS / hold_days) if block["ic_ir"] else None
        )
    else:
        block["annualized_mean"] = mean * TRADING_DAYS / hold_days
    return _clean(block)


def _values_by_date(rows: Sequence[Mapping[str, Any]], track: str, key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        if row.get("track") != track:
            continue
        value = _finite(row.get(key))
        if value is not None:
            out[str(row.get("trade_date"))[:8]] = value
    return out


def _paired(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    hold_days: int,
    unit: str,
) -> Dict[str, Any]:
    """Left minus right, paired by date; a date missing on either side is dropped.

    Dropping each side independently would compare two different date sets and
    then call the difference an effect. Pairing buys far less than it looks like
    it should here -- the tracks share their dates almost completely -- so the
    paired spread is reported and left to speak for itself.
    """
    common = sorted(set(left) & set(right))
    differences = [left[day] - right[day] for day in common]
    block = _describe(differences, hold_days=hold_days, unit=unit)
    block["n_paired"] = len(common)
    spread = block.get("sd")
    if spread and block.get("mean") is not None and len(common) > 1:
        low, high = stats.t.interval(
            0.95, df=len(common) - 1, loc=block["mean"], scale=spread / math.sqrt(len(common))
        )
        block["ci_95"] = [float(low), float(high)]
    else:
        block["ci_95"] = None
    return _clean(block)


def _grid(signal_dates: Sequence[str], trading_days: Sequence[str], hold_days: int) -> Dict[str, Any]:
    """Describe the sampling grid, and check it is what the t-test assumes.

    Overlap is measured in sessions off the window calendar, not in calendar
    days: consecutive signals sit ``interval`` sessions apart, so the return
    intervals ``[entry, entry + hold]`` touch exactly and share no return. With
    no calendar to measure against the claim is refused rather than assumed.
    """
    index = {day: position for position, day in enumerate(sorted({str(d)[:8] for d in trading_days}))}
    positions = [index[str(day)[:8]] for day in signal_dates if str(day)[:8] in index]
    gaps = [later - earlier for earlier, later in zip(positions, positions[1:])]
    session_gap = min(gaps) if gaps else None
    return _clean(
        {
            "kind": "rebalance_dates",
            "n_dates": len(signal_dates),
            "horizon_days": hold_days,
            "min_session_gap": session_gap,
            "non_overlapping": bool(session_gap is not None and session_gap >= hold_days),
            "note": _GRID_NOTE,
        }
    )


def _power(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal_dates: Sequence[str],
    hold_days: int,
    grid: Mapping[str, Any],
) -> Dict[str, Any]:
    """What this many dates could have detected, in each family's own units."""
    baseline_ic = _describe(list(_values_by_date(rows, "baseline", "ic_rank").values()), hold_days=hold_days, unit="ic")
    baseline_excess = _describe(
        list(_values_by_date(rows, "baseline", "excess_gross").values()), hold_days=hold_days, unit="return"
    )
    per_year = TRADING_DAYS / hold_days
    excess_mde = baseline_excess.get("mde_80")
    return _clean(
        {
            "grid": grid.get("kind"),
            "n_dates": len(signal_dates),
            "effective_n": len(signal_dates),
            # Baseline only: the three tracks are scored on the same dates and
            # sum to three copies of one cross-section, which would make the
            # number describe the row count rather than the data.
            "pooled_observations": sum(
                int(row.get("n_joined") or 0) for row in rows if row.get("track") == "baseline"
            ),
            "mde_80_multiplier": MDE_Z_80,
            "mde_80_rank_ic_baseline": baseline_ic.get("mde_80"),
            "mde_80_excess_baseline_pp_per_period": (excess_mde * 100.0) if excess_mde is not None else None,
            "mde_80_excess_baseline_pct_per_year": (excess_mde * per_year * 100.0) if excess_mde is not None else None,
            "note": _POWER_NOTE,
        }
    )


def _control(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    t_max: float,
) -> Dict[str, Any]:
    """Track R's IC against the null its own construction implies.

    Independent scores and labels give a single-day rank IC a sampling standard
    deviation of about ``1 / sqrt(m - 1)`` for ``m`` names, so the mean of ``D``
    such days has one of ``sqrt(sum(1 / (m_d - 1))) / D``. That is a second,
    analytic check on the score-to-label join, independent of the t-test and of
    ``control_r`` (which tests the ledger and the cost model, not the join).
    """
    series = _values_by_date(rows, "control_random", "ic_rank")
    joined = {
        str(row.get("trade_date"))[:8]: int(row.get("n_joined") or 0)
        for row in rows
        if row.get("track") == "control_random"
    }
    per_day = [math.sqrt(1.0 / (joined[day] - 1)) for day in series if joined.get(day, 0) > 1]
    null_per_day = (sum(per_day) / len(per_day)) if per_day else None
    null_of_mean = (math.sqrt(sum(value**2 for value in per_day)) / len(per_day)) if per_day else None
    mean_ic = (sum(series.values()) / len(series)) if series else None
    t_stat = (mean_ic / null_of_mean) if mean_ic is not None and null_of_mean else None
    return _clean(
        {
            "seed": seed,
            "analytic_null_sd_per_day": null_per_day,
            "analytic_null_sd_of_mean": null_of_mean,
            "mean_rank_ic": mean_ic,
            "t_stat": t_stat,
            "t_max": t_max,
            "passed": bool(t_stat is not None and abs(t_stat) <= t_max),
            "note": (
                "Track R's scores are a permutation of the day's names, so its rank IC has zero expectation "
                "and a known sampling spread. Its series can be shorter than the baseline's, because R only "
                "exists where the pool does. A deviation means the score-to-label join is wrong, not that "
                "randomness paid."
            ),
        }
    )


def summarize_primary(
    rows: Sequence[Mapping[str, Any]],
    *,
    trading_days: Sequence[str] = (),
    hold_days: int = 20,
    pool_basis: str = POOL_BASIS,
    control_seed: int = 0,
    control_t_max: float = 3.0,
    horizon_declared_by: str = "rebalance_days",
) -> Dict[str, Any]:
    """Everything the report says about the primary evidence, derived from ``rows``.

    Deriving the summary from the same rows the CSV is written from is what makes
    the two impossible to disagree: there is no second computation to drift. The
    ``vs pool`` family needs no separate block -- a TOP-K excess over the pool
    *is* that comparison, and it is tested against zero in place.
    """
    rows = list(rows)
    signal_dates = list(dict.fromkeys(str(row.get("trade_date"))[:8] for row in rows))
    grid = _grid(signal_dates, trading_days, hold_days)
    grid["declared_by"] = horizon_declared_by

    pool_sizes = sorted({int(row.get("n_pool") or 0) for row in rows})
    wide_sizes = sorted({int(row.get("n_wide") or 0) for row in rows})
    narrow = [_finite(row.get("pool_return")) for row in rows]
    wide = [_finite(row.get("pool_return_wide")) for row in rows]
    narrow = [value for value in narrow if value is not None]
    wide = [value for value in wide if value is not None]

    # The top-level audit numbers describe the baseline's cross-section, the same
    # one ``_power`` counts; the other two tracks are scored on the same dates and
    # would only add copies of it. Their own counts live under ``tracks``.
    baseline_rows = [row for row in rows if row.get("track") == "baseline" and not row.get("carried")]
    scored = sum(int(row.get("n_scored") or 0) for row in baseline_rows)
    joined = sum(int(row.get("n_joined") or 0) for row in baseline_rows)

    tracks: Dict[str, Any] = {}
    for track in TRACKS:
        track_rows = [row for row in rows if row.get("track") == track]
        tracks[track] = _clean(
            {
                "rank_ic": _describe(
                    list(_values_by_date(rows, track, "ic_rank").values()), hold_days=hold_days, unit="ic"
                ),
                "pearson_ic": _describe(
                    list(_values_by_date(rows, track, "ic_pearson").values()), hold_days=hold_days, unit="ic"
                ),
                "topk_excess_gross_vs_pool": _describe(
                    list(_values_by_date(rows, track, "excess_gross").values()), hold_days=hold_days, unit="return"
                ),
                "topk_excess_net_vs_pool": _describe(
                    list(_values_by_date(rows, track, "excess_net").values()), hold_days=hold_days, unit="return"
                ),
                "avg_cost_per_period": _mean([_finite(row.get("cost")) for row in track_rows]),
                "dates_without_cross_section": sum(1 for row in track_rows if row.get("carried")),
                "dates_unpriced": sum(1 for row in track_rows if row.get("pricing") == "none"),
                "dates_below_min_names": sum(
                    1
                    for row in track_rows
                    if not row.get("carried") and 0 < int(row.get("n_joined") or 0) < MIN_IC_NAMES
                ),
                "dates_without_a_join": sum(
                    1 for row in track_rows if not row.get("carried") and int(row.get("n_joined") or 0) == 0
                ),
            }
        )

    comparisons = {
        family: {
            f"{left}_minus_{right}": _paired(
                _values_by_date(rows, left, key),
                _values_by_date(rows, right, key),
                hold_days=hold_days,
                unit="ic" if family.endswith("_ic") else "return",
            )
            for left, right in (
                ("model", "baseline"),
                ("model", "control_random"),
                ("baseline", "control_random"),
            )
        }
        for family, key in (
            ("rank_ic", "ic_rank"),
            ("pearson_ic", "ic_pearson"),
            ("topk_excess_gross", "excess_gross"),
        )
    }

    return _clean(
        {
            "grid": grid,
            "pool": {
                "basis": pool_basis,
                "sources": [POOL_SOURCE_A, POOL_SOURCE_B],
                "fallback_used": sum(1 for row in rows if row.get("pool_basis") == POOL_SOURCE_B),
                "n_pool_mean": _mean(pool_sizes),
                "n_pool_min": pool_sizes[0] if pool_sizes else None,
                "mean_pool_return": _mean(narrow),
                "wide_basis_n_mean": _mean(wide_sizes),
                "wide_basis_mean_return": _mean(wide),
                "note": (
                    "The benchmark is the day's own scored cross-section, priced. pool_return_wide is the "
                    "same day's full priced universe -- every symbol the window's price load touched -- and "
                    "mixes in names that were not index members yet, so it is reported beside the benchmark "
                    "instead of being it. Its premium is not constant: it is about +26pp per period in "
                    "2016 and disappears by 2024, the shape of membership look-ahead rather than of a "
                    "fixed bias. A date whose baseline cross-section the validator refused to rank falls "
                    "back to the model's, which exists on every signal date."
                ),
            },
            "join": {
                "track": "baseline",
                "scored_names": scored,
                "joined_names": joined,
                "join_rate": (joined / scored) if scored else None,
                "dates_without_a_join": sum(
                    1 for row in baseline_rows if int(row.get("n_joined") or 0) == 0
                ),
                "note": _JOIN_NOTE,
            },
            "tracks": tracks,
            "comparisons": comparisons,
            "control": _control(rows, seed=control_seed, t_max=control_t_max),
            "power": _power(rows, signal_dates=signal_dates, hold_days=hold_days, grid=grid),
            "cost": {"applied_to": "topk_excess_net only", "note": _COST_NOTE},
            "caveats": [
                "IC is measured on raw forward returns. A per-date demean is one constant within the date "
                "each IC is computed over, and a correlation is invariant under adding a constant, so a "
                "demeaned IC equals a raw IC exactly -- this is not a second, 'neutralized' flavour of IC.",
                "Rank IC (Spearman) and Pearson IC are the same definition with the labels ranked first or "
                "not; they differ only in how much a single extreme name is allowed to matter.",
                "The window's symbol union appears as pool_return_wide for visibility only and is never the "
                "benchmark; the benchmark is the day's own cross-section.",
                "A date with no cross-section has no scoring event and therefore no IC; it is counted in "
                "dates_without_cross_section rather than filled in from the held names.",
                "Costs are a deterministic drag on the pool-relative excess, so topk_excess_net is the "
                "cost-bearing series and topk_excess_gross is the one with a zero expectation under the "
                "control track.",
            ],
        }
    )
