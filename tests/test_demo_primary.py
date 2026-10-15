"""The primary evidence: what the IC and pool-excess block must and must not claim."""

import json
import math
import random
from statistics import pstdev

import pytest

from quant_model_platform.demo.calendar import rebalance_signal_dates
from quant_model_platform.demo.portfolio import simulate_track
from quant_model_platform.demo.primary import (
    MDE_Z_80,
    QUANTILE_BUCKETS,
    daily_metrics,
    pool_symbols_by_date,
    quantile_rows,
    summarize_primary,
    summarize_quantiles,
)
from quant_model_platform.demo.ranking import random_control_scores

CONTROL_SEED = 11
INTERVAL = 20


def _days(count):
    return [f"2025{index:04d}" for index in range(1, count + 1)]


def _names(count):
    return [f"S{index:03d}" for index in range(count)]


def _monotone(days, names):
    """A ranks the day's names ascending, B descending, labels ascend too.

    Monotone on purpose: the IC of each track is then exactly +1 or -1, so a
    sign error or a shuffled join cannot hide behind noise.
    """
    centre = len(names) / 2.0
    scores_a, scores_b, forwards = {}, {}, {}
    for day in days:
        forwards[day] = {name: (index - centre) / 100.0 for index, name in enumerate(names)}
        scores_a[day] = [{"ts_code": name, "score": float(index)} for index, name in enumerate(names)]
        scores_b[day] = [{"ts_code": name, "score": float(len(names) - 1 - index)} for index, name in enumerate(names)]
    return scores_a, scores_b, forwards


def _varying(days, names):
    """A monotone ranking against labels that shift day to day.

    The wobble is deliberately larger than the label's own step, so it moves
    *ranks* and not just levels -- a rank IC would not see a perturbation that
    only nudged values. The IC series then has a real spread, which is what the
    dispersion, t and minimum-detectable-effect checks need, and the pool mean
    is off zero so the demean test can watch it move.
    """
    centre = len(names) / 2.0
    scores_a, scores_b, forwards = {}, {}, {}
    for index, day in enumerate(days):
        forwards[day] = {
            name: (position - centre) * 0.01 + (((position * 7 + index * 3) % 11) - 5) * 0.012
            for position, name in enumerate(names)
        }
        scores_a[day] = [{"ts_code": name, "score": float(position)} for position, name in enumerate(names)]
        scores_b[day] = [
            {"ts_code": name, "score": float(position) + ((position * 5 + index) % 7)}
            for position, name in enumerate(names)
        ]
    return scores_a, scores_b, forwards


def _rows(scores_a, scores_b, forwards, days, *, top_n=2, control=None):
    """Daily metric rows on the rebalance grid the demo actually samples."""
    signals = rebalance_signal_dates(days, interval=INTERVAL)
    control = random_control_scores(scores_a, seed=CONTROL_SEED) if control is None else control
    scores = {"baseline": scores_a, "model": scores_b, "control_random": control}
    track_rows = {track: simulate_track(signals, scores[track], forwards, top_n=top_n, track=track) for track in scores}
    symbols, sources = pool_symbols_by_date(scores_a, scores_b, signals)
    rows = daily_metrics(scores, forwards, track_rows, pool_symbols=symbols, pool_sources=sources)
    return rows, track_rows


def _summary(days, names, *, source=None, top_n=2, **kwargs):
    scores_a, scores_b, forwards = (source or _varying)(days, names)
    rows, track_rows = _rows(scores_a, scores_b, forwards, days, top_n=top_n)
    return summarize_primary(rows, trading_days=days, hold_days=INTERVAL, **kwargs), rows, track_rows


def test_ic_is_exactly_one_when_the_ranking_matches_the_label():
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _monotone(days, names)

    rows, _ = _rows(scores_a, scores_b, forwards, days)

    assert len(rows) == 3 * len(rebalance_signal_dates(days, interval=INTERVAL))
    for row in rows:
        if row["track"] == "baseline":
            assert row["ic_rank"] == pytest.approx(1.0)
            assert row["ic_pearson"] == pytest.approx(1.0)
        elif row["track"] == "model":
            assert row["ic_rank"] == pytest.approx(-1.0)
            assert row["ic_pearson"] == pytest.approx(-1.0)


def test_demeaning_the_label_cannot_move_an_ic():
    """The whole point of the identity in the module docstring.

    A pool demean subtracts one constant per date, inside the group each IC is
    computed over, so a correlation -- which is invariant under adding a
    constant -- cannot see it. If someone later replaces this with a real
    per-industry residualization, this test is what goes red.
    """
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)
    demeaned = {
        day: {name: value - sum(labels.values()) / len(labels) for name, value in labels.items()}
        for day, labels in forwards.items()
    }

    raw, _ = _rows(scores_a, scores_b, forwards, days)
    shifted, _ = _rows(scores_a, scores_b, demeaned, days)
    by_key = {(row["track"], row["trade_date"]): row for row in shifted}

    assert raw[0]["pool_return"] != 0.0  # or the test could pass for the wrong reason
    for row in raw:
        moved = by_key[(row["track"], row["trade_date"])]
        assert moved["ic_rank"] == pytest.approx(row["ic_rank"])
        assert moved["ic_pearson"] == pytest.approx(row["ic_pearson"])
        # The level statistics are what a demean does move, and the excess is a
        # difference of two levels, so it lands back where it started.
        assert moved["pool_return"] == pytest.approx(0.0)
        assert row["pool_return"] != pytest.approx(0.0)
        assert moved["excess_gross"] == pytest.approx(row["excess_gross"])


def test_the_pool_is_the_days_cross_section_not_the_windows_symbol_union():
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _monotone(days, names)
    # A symbol the price loader happened to see but that no track ever ranked: a
    # name that was not in the index that day. It must not reach the benchmark.
    for day in days:
        forwards[day]["GHOST"] = 5.0

    rows, _ = _rows(scores_a, scores_b, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)

    for row in rows:
        assert row["n_pool"] == len(names)
        assert row["n_wide"] == len(names) + 1
        assert abs(row["pool_return"]) < 0.01  # the labels are near-symmetric
        assert row["pool_return_wide"] > 0.1  # GHOST drags the union basket up
    assert summary["pool"]["wide_basis_n_mean"] > summary["pool"]["n_pool_mean"]
    assert abs(summary["pool"]["mean_pool_return"]) < 0.01
    assert summary["pool"]["wide_basis_mean_return"] > 0.1
    assert {row["pool_basis"] for row in rows} == {"baseline_cross_section"}


def test_the_pool_falls_back_to_the_model_cross_section_when_a_has_none():
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)
    scores_a.pop(days[0])

    rows, _ = _rows(scores_a, scores_b, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)

    first = [row for row in rows if row["trade_date"] == days[0]]
    assert len(first) == 3
    assert {row["pool_basis"] for row in first} == {"model_cross_section"}
    assert first[0]["n_pool"] == len(names)
    assert summary["pool"]["fallback_used"] == len(first)


def test_the_pool_size_is_a_mean_over_dates_not_over_distinct_values():
    """Twenty rebalance dates, two different pool sizes, in unequal numbers.

    Fifteen dates rank thirty names and five rank twenty, so the honest mean is
    27.5. Averaging the two *distinct* sizes instead gives 25 -- a number no date
    ever had. This is what a set comprehension over ``n_pool`` produces, and it is
    invisible whenever the sizes happen to be equally frequent.
    """
    days = _days(400)
    names = _names(30)
    scores_a, scores_b, forwards = {}, {}, {}
    for index, day in enumerate(days):
        keep = names[: 30 if index < 300 else 20]
        forwards[day] = {name: (position - 10) / 100.0 for position, name in enumerate(names)}
        scores_a[day] = [{"ts_code": name, "score": float(position)} for position, name in enumerate(keep)]
        scores_b[day] = [{"ts_code": name, "score": float(position)} for position, name in enumerate(keep)]

    rows, _ = _rows(scores_a, scores_b, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)

    sizes = [row["n_pool"] for row in rows if row["track"] == "baseline"]
    assert len(sizes) == 20
    assert sizes.count(30) == 15 and sizes.count(20) == 5  # or the test measures nothing
    assert summary["pool"]["n_pool_mean"] == pytest.approx(27.5)
    assert summary["pool"]["n_pool_min"] == 20
    assert summary["pool"]["wide_basis_n_mean"] == pytest.approx(30.0)  # labels never shrank


def test_an_unpriced_date_has_no_pool_size_and_no_vote_in_the_mean():
    """The last rebalance date has no forward window, so it has no pool.

    Counting it as a pool of size zero would drag the mean down; the sizes must
    describe the same dates ``mean_pool_return`` averages.
    """
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)
    signals = rebalance_signal_dates(days, interval=INTERVAL)
    del forwards[signals[-1]]

    rows, _ = _rows(scores_a, scores_b, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)

    last = [row for row in rows if row["track"] == "baseline" and row["trade_date"] == signals[-1]]
    assert [row["pricing"] for row in last] == ["none"]
    assert last[0]["n_pool"] == 0
    assert summary["pool"]["n_pool_mean"] == pytest.approx(len(names))
    assert summary["pool"]["n_pool_min"] == len(names)


def test_a_thin_cross_section_reports_a_row_but_no_ic():
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)
    thin = days[0]
    for mapping in (scores_a, scores_b):
        mapping[thin] = mapping[thin][:4]

    rows, _ = _rows(scores_a, scores_b, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)

    thin_rows = [row for row in rows if row["trade_date"] == thin]
    assert len(thin_rows) == 3
    for row in thin_rows:
        assert row["n_joined"] == 4
        assert row["ic_rank"] is None
        assert row["pool_return"] is not None  # the pool still exists at four names
    # The row is still there and still reports an excess; it just cannot be
    # correlated, so the IC series is one date shorter than the grid.
    assert summary["tracks"]["baseline"]["rank_ic"]["n_dates"] == summary["power"]["n_dates"] - 1
    assert summary["tracks"]["baseline"]["dates_below_min_names"] == 1
    assert summary["power"]["n_dates"] == 20


def test_a_date_with_no_prices_has_no_ic_and_is_not_a_flat_zero():
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)
    forwards.pop(days[0])

    rows, _ = _rows(scores_a, scores_b, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)

    for row in [row for row in rows if row["trade_date"] == days[0]]:
        assert row["pricing"] == "none"
        assert row["ic_rank"] is None
        assert row["pool_return"] is None
        assert row["excess_gross"] is None
        assert row["excess_net"] is None
    assert summary["tracks"]["baseline"]["dates_unpriced"] == 1


def test_a_partial_forward_window_still_correlates_on_the_priced_subset():
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _monotone(days, names)
    forwards[days[0]] = {name: value for name, value in forwards[days[0]].items() if name != names[-1]}

    rows, _ = _rows(scores_a, scores_b, forwards, days)

    first = [row for row in rows if row["trade_date"] == days[0] and row["track"] == "baseline"][0]
    assert first["pricing"] == "partial"
    assert first["n_joined"] == len(names) - 1
    assert first["ic_rank"] == pytest.approx(1.0)


def test_a_carried_date_has_no_scoring_event_so_it_has_no_ic():
    """The trap: the held names are not a ranking that happened that day."""
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)
    blank = days[INTERVAL]  # the second signal date: there is a book to carry into it
    scores_a.pop(blank)

    rows, track_rows = _rows(scores_a, scores_b, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)

    baseline = [row for row in rows if row["trade_date"] == blank and row["track"] == "baseline"][0]
    assert track_rows["baseline"][1]["cross_section"] == "carried"
    assert baseline["carried"] == 1
    assert baseline["n_joined"] == 0
    assert baseline["ic_rank"] is None
    # The pool still exists -- the model scored that day -- and the held book
    # still earns against it. Only the correlation is unavailable.
    assert baseline["pool_basis"] == "model_cross_section"
    assert baseline["excess_gross"] is not None
    assert summary["tracks"]["baseline"]["dates_without_cross_section"] == 1
    assert summary["tracks"]["model"]["dates_without_cross_section"] == 0


def test_excess_and_cost_compose_exactly():
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)

    rows, track_rows = _rows(scores_a, scores_b, forwards, days)
    gross = {row["trade_date"]: row["gross_return"] for row in track_rows["baseline"]}

    for row in rows:
        if row["track"] != "baseline":
            continue
        # topk_mean is the equity engine's own number, not a recomputation.
        assert row["topk_mean"] == gross[row["trade_date"]]
        if row["excess_gross"] is None:
            assert row["pool_return"] is None
            continue
        assert row["excess_gross"] == pytest.approx(row["topk_mean"] - row["pool_return"])
        assert row["excess_net"] == pytest.approx(row["excess_gross"] - row["cost"])


def test_the_random_control_ic_matches_its_analytic_null():
    """A second check on the join, independent of the t-test it corroborates."""
    days, names = _days(400), _names(60)
    summary, _, _ = _summary(days, names)
    control = summary["control"]

    expected = (1.0 / math.sqrt(len(names) - 1)) / math.sqrt(summary["power"]["n_dates"])
    assert control["analytic_null_sd_per_day"] == pytest.approx(1.0 / math.sqrt(len(names) - 1))
    assert abs(control["mean_rank_ic"]) <= 3.0 * expected
    assert control["passed"] is True
    assert control["seed"] == 0  # summarize_primary's own default, not the demo's


def test_power_counts_dates_and_only_prints_the_observation_count():
    days, names = _days(400), _names(60)
    summary, rows, _ = _summary(days, names)
    power = summary["power"]

    assert power["n_dates"] == 20  # 400 trading days on a 20-day grid
    assert power["effective_n"] == power["n_dates"]
    # The baseline's cross-section, not the three tracks' rows summed: the
    # tracks share their dates, so the sum would be three copies of one series.
    assert power["pooled_observations"] == sum(
        row["n_joined"] for row in rows if row["track"] == "baseline"
    )
    every_track = sum(row["n_joined"] for row in rows)
    assert every_track > power["pooled_observations"] * 2
    assert power["pooled_observations"] > power["effective_n"] * len(names) // 2


def test_the_audit_counts_describe_one_cross_section_not_three():
    """The top-level counts are the baseline's; the tracks share these dates."""
    days, names = _days(400), _names(60)
    summary, rows, _ = _summary(days, names)
    join = summary["join"]

    baseline = [row for row in rows if row["track"] == "baseline" and not row["carried"]]
    assert join["track"] == "baseline"
    assert join["scored_names"] == sum(row["n_scored"] for row in baseline)
    assert join["joined_names"] == sum(row["n_joined"] for row in baseline)
    assert join["joined_names"] == summary["power"]["pooled_observations"]
    assert join["dates_without_a_join"] == sum(1 for row in baseline if row["n_joined"] == 0)
    assert join["dates_without_a_join"] == 0


def test_every_summarised_family_reads_a_column_that_exists():
    """A renamed row key must fail here, not silently become a null block.

    ``pearson_ic`` once summarised the column ``pearson_ic`` while the rows
    called it ``ic_pearson``: every block came out ``None`` and nothing went red.
    """
    days, names = _days(400), _names(60)
    summary, rows, _ = _summary(days, names)
    columns = ("ic_rank", "ic_pearson", "excess_gross", "excess_net")

    for track in ("baseline", "model", "control_random"):
        block = summary["tracks"][track]
        for family, column in (
            ("rank_ic", "ic_rank"),
            ("pearson_ic", "ic_pearson"),
            ("topk_excess_gross_vs_pool", "excess_gross"),
            ("topk_excess_net_vs_pool", "excess_net"),
        ):
            assert column in rows[0], column
            values = [row[column] for row in rows if row["track"] == track and row[column] is not None]
            assert values, f"{track}.{family} summarised nothing"
            assert block[family]["n_dates"] == len(values)
            assert block[family]["mean"] == pytest.approx(sum(values) / len(values))


def test_mde_follows_the_contract_convention():
    days, names = _days(400), _names(60)
    summary, _, _ = _summary(days, names)
    block = summary["tracks"]["baseline"]["rank_ic"]

    assert block["sd"] > 0
    assert block["mde_80"] == pytest.approx(MDE_Z_80 * block["sd"] / math.sqrt(block["n_dates"]))
    assert block["t_stat"] == pytest.approx(block["mean"] / (block["sd"] / math.sqrt(block["n_dates"])))
    assert block["ic_ir"] == pytest.approx(block["mean"] / block["sd"])
    assert block["ic_ir_annualized"] == pytest.approx(block["ic_ir"] * math.sqrt(252.0 / INTERVAL))


def test_the_grid_only_claims_non_overlap_when_it_can_measure_it():
    days, names = _days(400), _names(60)
    summary, rows, _ = _summary(days, names)

    spaced = summary["grid"]
    assert spaced["min_session_gap"] == INTERVAL
    assert spaced["non_overlapping"] is True

    # A 25-day label over a 20-session grid overlaps, and must say so.
    assert summarize_primary(rows, trading_days=days, hold_days=25)["grid"]["non_overlapping"] is False
    # With no calendar there is nothing to measure, so nothing is claimed.
    blind = summarize_primary(rows, hold_days=INTERVAL)["grid"]
    assert blind["min_session_gap"] is None
    assert blind["non_overlapping"] is False


def test_the_summary_is_a_pure_function_of_the_rows():
    """CSV and JSON cannot disagree, because there is no second computation."""
    days, names = _days(400), _names(60)
    summary, rows, _ = _summary(days, names)

    mutated = [dict(row) for row in rows]
    for row in mutated:
        if row["track"] == "baseline" and row["excess_gross"] is not None:
            row["excess_gross"] += 0.01
    after = summarize_primary(mutated, trading_days=days, hold_days=INTERVAL)

    before_mean = summary["tracks"]["baseline"]["topk_excess_gross_vs_pool"]["mean"]
    assert after["tracks"]["baseline"]["topk_excess_gross_vs_pool"]["mean"] == pytest.approx(before_mean + 0.01)
    assert after["tracks"]["model"]["topk_excess_gross_vs_pool"]["mean"] == pytest.approx(
        summary["tracks"]["model"]["topk_excess_gross_vs_pool"]["mean"]
    )


def test_pairing_drops_the_dates_either_side_lacks():
    days, names = _days(400), _names(60)
    summary, rows, _ = _summary(days, names)
    baseline_dates = {row["trade_date"] for row in rows if row["track"] == "baseline" and row["ic_rank"] is not None}
    model_dates = {row["trade_date"] for row in rows if row["track"] == "model" and row["ic_rank"] is not None}

    block = summary["comparisons"]["rank_ic"]["model_minus_baseline"]

    assert block["n_paired"] == len(baseline_dates & model_dates) == 20
    assert block["ci_95"] is not None
    assert block["ci_95"][0] <= block["mean"] <= block["ci_95"][1]


def test_a_degenerate_difference_reports_nothing_rather_than_dividing_by_zero():
    """Two identical tracks: the paired difference is zero everywhere."""
    days, names = _days(400), _names(30)
    scores_a, _, forwards = _monotone(days, names)

    rows, _ = _rows(scores_a, scores_a, forwards, days)
    summary = summarize_primary(rows, trading_days=days, hold_days=INTERVAL)
    block = summary["comparisons"]["rank_ic"]["model_minus_baseline"]

    assert block["mean"] == pytest.approx(0.0)
    assert block["sd"] == pytest.approx(0.0)
    assert block["t_stat"] is None
    assert block["ci_95"] is None


def test_no_summary_ever_serializes_a_bare_nan():
    """``json.dumps(nan)`` emits invalid JSON that Python reads straight back."""
    days, names = _days(400), _names(30)
    summary, _, _ = _summary(days, names)
    empty = summarize_primary([], trading_days=days, hold_days=INTERVAL)

    for block in (summary, empty):
        # allow_nan=False is the check: it raises on any non-finite float, where
        # a substring search would also hit the prose in the notes.
        text = json.dumps(block, allow_nan=False)
        assert json.loads(text) == block


def test_an_empty_grid_reports_absent_numbers_not_zeros():
    summary = summarize_primary([], trading_days=(), hold_days=INTERVAL)

    assert summary["power"]["n_dates"] == 0
    assert summary["power"]["effective_n"] == 0
    assert summary["power"]["pooled_observations"] == 0
    assert summary["power"]["mde_80_rank_ic_baseline"] is None
    assert summary["pool"]["n_pool_mean"] is None
    assert summary["join"]["join_rate"] is None
    assert summary["control"]["passed"] is False  # nothing to pass with
    for track in ("baseline", "model", "control_random"):
        assert summary["tracks"][track]["rank_ic"]["n_dates"] == 0
        assert summary["tracks"][track]["rank_ic"]["mean"] is None


def _ladder(days, names, *, source=None, top_n=2, buckets=QUANTILE_BUCKETS, seed=CONTROL_SEED):
    """The quantile ladder and the metric rows it must agree with, one run."""
    scores_a, scores_b, forwards = (source or _varying)(days, names)
    signals = rebalance_signal_dates(days, interval=INTERVAL)
    control = random_control_scores(scores_a, seed=seed)
    scores = {"baseline": scores_a, "model": scores_b, "control_random": control}
    track_rows = {track: simulate_track(signals, scores[track], forwards, top_n=top_n, track=track) for track in scores}
    symbols, sources = pool_symbols_by_date(scores_a, scores_b, signals)
    quantiles = quantile_rows(scores, forwards, track_rows, pool_symbols=symbols, buckets=buckets)
    metrics = daily_metrics(scores, forwards, track_rows, pool_symbols=symbols, pool_sources=sources)
    return quantiles, metrics


def _by_day_track(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["trade_date"], row["track"]), []).append(row)
    return grouped


def _shuffled(seed):
    """Scores drawn at random: there is no ranking to find, so no shape to report."""

    def source(days, names):
        generator = random.Random(seed)
        scores, forwards = {}, {}
        for day in days:
            order = list(names)
            generator.shuffle(order)
            scores[day] = [{"ts_code": name, "score": float(position)} for position, name in enumerate(order)]
            forwards[day] = {name: generator.uniform(-0.1, 0.1) for name in names}
        return scores, scores, forwards

    return source


def test_the_rungs_reconstruct_the_pool_return():
    """Ten buckets are a partition of the pool, so their n-weighted mean is it.

    This is the invariant that makes a ladder readable at all: if the rungs do
    not add back up to the benchmark, a rung's excess is measured against a
    number the rung is not a part of, and the shape means nothing.
    """
    days, names = _days(400), _names(30)
    quantiles, _ = _ladder(days, names)

    grouped = _by_day_track(quantiles)
    assert grouped
    for (day, track), rows in grouped.items():
        assert len(rows) == QUANTILE_BUCKETS, (day, track)
        assert len({row["pool_return"] for row in rows}) == 1  # one benchmark a day
        total = sum(row["n"] for row in rows)
        weighted = sum(row["n"] * row["mean_return"] for row in rows)
        assert weighted / total == pytest.approx(rows[0]["pool_return"], abs=1e-12)


def test_the_rungs_partition_exactly_the_names_the_ic_joined():
    days, names = _days(400), _names(30)
    quantiles, metrics = _ladder(days, names)

    joined = {(row["trade_date"], row["track"]): row["n_joined"] for row in metrics}
    totals = {}
    for row in quantiles:
        key = (row["trade_date"], row["track"])
        totals[key] = totals.get(key, 0) + row["n"]

    assert totals
    for key, total in totals.items():
        assert total == joined[key] > 0


def test_a_monotone_ranking_has_a_perfectly_monotone_ladder():
    days, names = _days(400), _names(30)
    quantiles, _ = _ladder(days, names, source=_monotone)
    summary = summarize_quantiles(quantiles, hold_days=INTERVAL)

    # A ranks with the label, B against it: +1 and -1, not merely "positive".
    assert summary["tracks"]["baseline"]["monotonicity"] == pytest.approx(1.0, abs=1e-9)
    assert summary["tracks"]["model"]["monotonicity"] == pytest.approx(-1.0, abs=1e-9)
    baseline = summary["tracks"]["baseline"]["buckets"]
    assert baseline[0]["excess_mean"] > baseline[-1]["excess_mean"]
    assert summary["tracks"]["baseline"]["spread_top_bottom"]["mean"] > 0.0
    assert summary["tracks"]["model"]["spread_top_bottom"]["mean"] < 0.0


def test_a_random_ranking_has_no_shape():
    """The analytic null this note leans on, measured rather than assumed.

    Ten rungs give a rank correlation a null spread of ``1/sqrt(k-1)``, which
    is 0.33 -- wide. So the test is not "is this one ladder flat" (it cannot be)
    but "does the statistic behave like the null it is quoted as": averaged over
    many draws the shape must sit within 3 standard errors of zero, and its
    spread must not be wildly wider than the null's.
    """
    days, names = _days(400), _names(30)
    shapes = []
    for seed in range(30):
        quantiles, _ = _ladder(days, names, source=_shuffled(seed))
        shapes.append(summarize_quantiles(quantiles, hold_days=INTERVAL)["tracks"]["baseline"]["monotonicity"])

    null = 1.0 / math.sqrt(QUANTILE_BUCKETS - 1)
    assert all(value is not None for value in shapes)
    assert abs(sum(shapes) / len(shapes)) <= 3.0 * null / math.sqrt(len(shapes))
    assert 0.5 * null <= pstdev(shapes) <= 1.6 * null


def test_a_cross_section_thinner_than_the_ladder_keeps_its_empty_rungs():
    """Four names cannot fill ten rungs. The gaps stay visible as gaps."""
    days, names = _days(400), _names(4)
    quantiles, _ = _ladder(days, names, top_n=2, buckets=10)

    grouped = _by_day_track(quantiles)
    assert grouped
    for (day, track), rows in grouped.items():
        assert len(rows) == 10, (day, track)
        filled = [row for row in rows if row["n"]]
        assert len(filled) == 4
        assert sum(row["n"] for row in rows) == 4
        for row in rows:
            if not row["n"]:
                assert row["mean_return"] is None
                assert row["excess_return"] is None


def test_a_carried_day_contributes_no_rungs():
    """No cross-section means no ranking happened, so no ladder may be invented.

    The trap this guards: using the held names as a stand-in cross-section,
    which would report a ranking that was never produced.
    """
    days, names = _days(400), _names(30)
    scores_a, scores_b, forwards = _varying(days, names)
    signals = rebalance_signal_dates(days, interval=INTERVAL)
    assert len(signals) >= 3
    stale_a = {signals[0]: scores_a[signals[0]]}
    stale_b = {signals[0]: scores_b[signals[0]]}
    control = random_control_scores(stale_a, seed=CONTROL_SEED)
    scores = {"baseline": stale_a, "model": stale_b, "control_random": control}
    track_rows = {
        track: simulate_track(signals, scores[track], forwards, top_n=2, track=track) for track in scores
    }
    symbols, sources = pool_symbols_by_date(stale_a, stale_b, signals)
    quantiles = quantile_rows(scores, forwards, track_rows, pool_symbols=symbols)
    metrics = daily_metrics(scores, forwards, track_rows, pool_symbols=symbols, pool_sources=sources)

    carried = [row for row in metrics if row["carried"]]
    assert carried and all(row["topk_mean"] is not None for row in carried)
    assert {row["trade_date"] for row in quantiles} == {signals[0]}


def test_paired_comparisons_only_use_dates_both_tracks_have():
    days, names = _days(400), _names(30)
    quantiles, _ = _ladder(days, names)
    summary = summarize_quantiles(quantiles, hold_days=INTERVAL)

    present = {}
    for row in quantiles:
        if row["excess_return"] is not None:
            present.setdefault((row["track"], row["bucket"]), set()).add(row["trade_date"])
    for bucket in range(1, QUANTILE_BUCKETS + 1):
        key = str(bucket)
        for track in ("model", "control_random"):
            block = summary["comparisons"][f"{track}_minus_baseline"][key]
            expected = present.get((track, bucket), set()) & present.get(("baseline", bucket), set())
            assert block["n_paired"] == len(expected)


def test_an_empty_ladder_reports_absent_numbers_not_zeros():
    summary = summarize_quantiles([], hold_days=INTERVAL)

    assert summary["buckets"] == QUANTILE_BUCKETS
    assert len(summary["tracks"]["baseline"]["buckets"]) == QUANTILE_BUCKETS
    for track in ("baseline", "model", "control_random"):
        block = summary["tracks"][track]
        assert block["monotonicity"] is None
        assert block["spread_top_bottom"]["mean"] is None
        assert block["spread_top_bottom"]["n_paired"] == 0
        for rung in block["buckets"]:
            assert rung["n_mean"] is None
            assert rung["excess_mean"] is None
            assert rung["excess_t_stat"] is None
            assert rung["n_dates"] == 0


def test_no_ladder_ever_serializes_a_bare_nan():
    days, names = _days(400), _names(30)
    quantiles, _ = _ladder(days, names)

    for block in (summarize_quantiles(quantiles, hold_days=INTERVAL), summarize_quantiles([], hold_days=INTERVAL)):
        text = json.dumps(block, allow_nan=False)
        assert json.loads(text) == block
