"""Daily mark-to-market NAV, so a drawdown is measured on the days that happened.

The rebalance grid gives ten years about 130 points, thirteen a year, so a
drawdown taken there steps over every trough inside a twenty-session holding
period. That is why the contract calls the NAV communication rather than
evidence, and it is also why "which track draws down less" could not be asked.
This module keeps the period returns exactly as they are and adds the days in
between.

The construction is deliberately downstream of the entry/exit rules in
``sources``: each priced symbol gets its own path -- bought at the entry open,
marked at each close, sold at its own exit open, then frozen as cash -- and the
book is the equal-weight mean of those paths. Cash before a symbol's own entry
(its first bar after the signal date may be days late) counts at 1.0. The
product of a period's daily factors therefore telescopes to ``exit / entry``,
so **the chained value of a period equals ``1 + gross_return`` exactly**, with
no dependence on per-symbol exit delays, on suspensions, or on which symbols
the rules refused. That equality is the point: it is the only thing that says
this series describes the same strategy the period rows do.

Costs enter the daily series multiplicatively -- the entry day's factor is
scaled by ``1 - cost`` -- while the period rows keep their additive
``gross - cost``. The two differ by ``gross * cost``, which is second order and
is reported rather than assumed away (``reconcile.net_gap_bound``).

Two cracks worth naming rather than smoothing over:

* A period whose exit was delayed by a limit-down still holds its book on days
  the next period has already bought. The chain multiplies both, which is what
  a book-per-period model says; a real account would have split the capital.
  ``coverage.overlap_days`` counts those days so the distortion stays visible.
* A ``carried`` period holds the previous book, but its path is still built
  from its own signal date, matching how the period engine prices it.
"""

from __future__ import annotations

import math
from statistics import pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence

from quant_model_platform.demo.sources import holding_path

#: The chained daily value of a period must equal its row's ``1 + gross_return``
#: to within this. It is the only check that the daily path describes the same
#: strategy the period rows do, so it is deliberately far above float noise.
RECONCILE_TOLERANCE = 1e-9


def drawdown_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Drawdown and downside shape of one NAV series.

    ``peak_to_trough_days`` and ``time_underwater_share`` come along because
    comparing two maximum drawdowns over ten years is a comparison of a handful
    of independent episodes; duration says whether the depth was a spike or a
    year of bleeding.
    """
    if not values:
        return {}
    peak = values[0]
    peak_index = 0
    worst = 0.0
    worst_peak = 0
    worst_trough = 0
    underwater = 0
    for index, value in enumerate(values):
        if value > peak:
            peak = value
            peak_index = index
        if value < peak:
            underwater += 1
        drawdown = value / peak - 1.0
        if drawdown < worst:
            worst = drawdown
            worst_peak = peak_index
            worst_trough = index
    returns = [values[i] / values[i - 1] - 1.0 for i in range(1, len(values)) if values[i - 1] > 0]
    return {
        "total_return": values[-1] - 1.0,
        "max_drawdown": worst,
        "peak_to_trough_days": worst_trough - worst_peak,
        "time_underwater_share": underwater / len(values),
        "ann_vol": (pstdev(returns) * math.sqrt(252.0)) if len(returns) > 1 else None,
        "downside_deviation": _downside_deviation(returns),
        "worst_day": min(returns) if returns else None,
        "n_days": len(values),
    }


def _downside_deviation(returns: Sequence[float]) -> Optional[float]:
    if not returns:
        return None
    squares = [value * value for value in returns if value < 0]
    if not squares:
        return 0.0
    return math.sqrt(sum(squares) / len(squares)) * math.sqrt(252.0)


def _priced(holdings: Sequence[str], labels: Mapping[str, float]) -> List[str]:
    """The book the period engine actually holds: the names that got a price.

    Same rule as ``simulate_track``'s ``priced`` and ``primary.pool_return``'s
    ``priced`` -- written the same way on purpose, because the reconciliation
    check below is what proves the three still agree.
    """
    return [symbol for symbol in holdings if labels.get(symbol) is not None]


def _ratio_path(
    history: Sequence[Dict[str, Any]],
    path: Mapping[str, Any],
) -> Dict[str, float]:
    """``{day: value / entry_price}`` from the entry bar through the exit bar."""
    entry_price = float(path["entry_price"])
    exit_index = int(path["exit_index"])
    series: Dict[str, float] = {}
    for index in range(int(path["entry_index"]), exit_index + 1):
        bar = history[index]
        price = float(path["exit_price"]) if index == exit_index else float(bar["close"])
        # The exit bar is sold at the open, so its ratio is the exit price; the
        # entry bar's own return runs from the entry open to that day's close.
        if price > 0:
            series[str(bar["trade_date"])] = price / entry_price
    return series


def _book_periods(
    rows: Sequence[Mapping[str, Any]],
    prices: Mapping[str, Sequence[Dict[str, Any]]],
    stamps: Mapping[str, Sequence[str]],
    forward_returns: Mapping[str, Mapping[str, float]],
    *,
    horizon: int,
) -> List[Dict[str, Any]]:
    """One entry per period: the days its book spans, and the day-to-day factors."""
    periods: List[Dict[str, Any]] = []
    for row in rows:
        day = str(row.get("trade_date"))[:8]
        labels = forward_returns.get(day) or {}
        paths: Dict[str, Dict[str, float]] = {}
        for symbol in _priced(list(row.get("holdings") or []), labels):
            history = prices.get(symbol)
            if not history:
                continue
            path = holding_path(history, day, horizon, dates=stamps.get(symbol))
            if path is None:
                continue
            series = _ratio_path(history, path)
            if series:
                paths[symbol] = series
        period: Dict[str, Any] = {
            "rebalance_date": day,
            "days": [],
            "factors": {},
            "cost": float(row.get("cost") or 0.0),
            "carried": row.get("cross_section") == "carried",
            "gross_gap": None,
            "net_gap": None,
            "net_gap_bound": None,
        }
        if paths:
            period["days"] = sorted({stamp for series in paths.values() for stamp in series})
            period["factors"] = _period_factors(paths, period["days"])
            gross = row.get("gross_return")
            if gross is not None:
                chained = _chain(period["factors"].values())
                period["gross_gap"] = chained - (1.0 + float(gross))
                net = row.get("net_return")
                if net is not None:
                    period["net_gap"] = chained * (1.0 - period["cost"]) - (1.0 + float(net))
                    period["net_gap_bound"] = abs(float(gross)) * period["cost"]
        periods.append(period)
    return periods


def _period_factors(paths: Mapping[str, Mapping[str, float]], days: Sequence[str]) -> Dict[str, float]:
    """Day-to-day value ratios of an equal-weight book, cash before each entry.

    A symbol the signal date could not reach on its next session sits at 1.0
    until its own entry bar, and holds its exit ratio afterwards -- sold means
    cash, and cash earns nothing here.
    """
    held = {symbol: 1.0 for symbol in paths}
    previous = 1.0
    factors: Dict[str, float] = {}
    for day in days:
        for symbol, series in paths.items():
            value = series.get(day)
            if value is not None:
                held[symbol] = value
        value = sum(held.values()) / len(held)
        factors[day] = value / previous
        previous = value
    return factors


def _chain(factors: Sequence[float]) -> float:
    value = 1.0
    for factor in factors:
        value *= factor
    return value


def _overlap_days(periods: Sequence[Mapping[str, Any]]) -> int:
    """Days a book still runs after the next period has already bought.

    The scheduled handover day is not an overlap: the old book sells at that
    open and the new one buys at the same open, so both factors belong to that
    day. A **delayed** exit is the real thing -- two books held at once, which
    this model charges to neither.
    """
    counted = 0
    for previous, current in zip(periods, periods[1:]):
        if not previous["days"] or not current["days"]:
            continue
        started = current["days"][0]
        shared = set(previous["days"]) & set(current["days"])
        counted += sum(1 for day in shared if day > started)
    return counted


def daily_block(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    prices: Mapping[str, Sequence[Dict[str, Any]]],
    forward_returns: Mapping[str, Mapping[str, float]],
    *,
    horizon: int,
    trading_days: Sequence[str] = (),
    relative_to: str = "",
) -> Optional[Dict[str, Any]]:
    """Daily NAV per book plus the drawdown summary; None when there is nothing.

    ``books`` maps a label (``baseline``, ``model``, ``pool``, ...) to that
    book's period rows -- the same rows ``simulate_track`` produces, so the
    benchmark is a book like any other rather than a second code path. The grid
    is the window calendar plus every day a path touches; a day nothing trades
    on leaves the NAV flat and is simply absent.
    """
    if not books or not prices:
        return None
    stamps = {symbol: [str(bar["trade_date"]) for bar in history] for symbol, history in prices.items()}
    built = {
        label: _book_periods(rows, prices, stamps, forward_returns, horizon=horizon)
        for label, rows in books.items()
    }
    if not any(period["days"] for periods in built.values() for period in periods):
        return None
    touched = {day for periods in built.values() for period in periods for day in period["days"]}
    grid = sorted(set(str(day)[:8] for day in trading_days) | touched)
    index_of = {day: index for index, day in enumerate(grid)}
    series = {label: _nav_series(periods, index_of, len(grid)) for label, periods in built.items()}
    net = {label: _nav_series(periods, index_of, len(grid), net=True) for label, periods in built.items()}
    owner = _owners(built, index_of, len(grid))
    rows = _rows(grid, series, net, owner, relative_to=relative_to)
    return {
        "rows": rows,
        "summary": _summarize(built, series, net, grid, relative_to=relative_to, trading_days=trading_days),
    }


def _nav_series(
    periods: Sequence[Mapping[str, Any]],
    index_of: Mapping[str, int],
    size: int,
    *,
    net: bool = False,
) -> List[float]:
    """Chained NAV over the whole grid; periods stack where their days overlap."""
    factors = [1.0] * size
    for period in periods:
        opening = True
        for day, factor in period["factors"].items():
            index = index_of.get(day)
            if index is None:
                continue
            if opening and net:
                factor *= 1.0 - period["cost"]
            opening = False
            factors[index] *= factor
    values: List[float] = []
    value = 1.0
    for factor in factors:
        value *= factor
        values.append(value)
    return values


def _owners(
    periods_by_book: Mapping[str, Sequence[Mapping[str, Any]]],
    index_of: Mapping[str, int],
    size: int,
) -> List[str]:
    owner = [""] * size
    for periods in periods_by_book.values():
        for period in periods:
            for day in period["days"]:
                index = index_of.get(day)
                if index is not None:
                    owner[index] = period["rebalance_date"]
    return owner


def _rows(
    grid: Sequence[str],
    series: Mapping[str, Sequence[float]],
    net: Mapping[str, Sequence[float]],
    owner: Sequence[str],
    *,
    relative_to: str,
) -> List[Dict[str, Any]]:
    base = series.get(relative_to)
    rows: List[Dict[str, Any]] = []
    for index, day in enumerate(grid):
        row: Dict[str, Any] = {"trade_date": day, "rebalance_date": owner[index] or ""}
        for label, values in series.items():
            row[f"{label}_gross"] = values[index]
            row[f"{label}_net"] = net[label][index]
            if base is not None and label != relative_to and base[index]:
                row[f"{label}_vs_{relative_to}"] = values[index] / base[index]
        rows.append(row)
    return rows


def _summarize(
    built: Mapping[str, Sequence[Mapping[str, Any]]],
    series: Mapping[str, Sequence[float]],
    net: Mapping[str, Sequence[float]],
    grid: Sequence[str],
    *,
    relative_to: str,
    trading_days: Sequence[str],
) -> Dict[str, Any]:
    tracks: Dict[str, Any] = {}
    for label, values in series.items():
        stats = drawdown_stats(values)
        stats["max_drawdown_net"] = drawdown_stats(net[label]).get("max_drawdown")
        stats["total_return_net"] = net[label][-1] - 1.0 if net[label] else None
        tracks[label] = stats
    relative: Dict[str, Any] = {}
    base = series.get(relative_to)
    if base:
        for label, values in series.items():
            if label == relative_to:
                continue
            ratio = [value / base[i] for i, value in enumerate(values) if base[i]]
            relative[f"{label}_vs_{relative_to}"] = drawdown_stats(ratio)
    gaps = {label: _max_gap(periods, "gross_gap") for label, periods in built.items()}
    net_gaps = {label: _max_gap(periods, "net_gap") for label, periods in built.items()}
    bounds = {label: _max_gap(periods, "net_gap_bound") for label, periods in built.items()}
    beyond = [day for day in grid if trading_days and str(day) > str(trading_days[-1])[:8]]
    return {
        "grid": {
            "kind": "trading_days plus every day a holding path touches",
            "first": grid[0] if grid else None,
            "last": grid[-1] if grid else None,
            "n_days": len(grid),
            "days_beyond_window": len(beyond),
            "declared_by": "window calendar",
        },
        "note": (
            "Daily marks of the same books the period rows describe. Costs are multiplicative "
            "here (the entry day's factor times 1 - cost) while the period rows stay additive, "
            "so the two differ by gross * cost -- bounded and reported below, not assumed away. "
            "The pool book is gross: it is a benchmark, and it pays no cost."
        ),
        "tracks": tracks,
        "relative": relative,
        "coverage": {
            "n_days": len(grid),
            "periods": {label: len(periods) for label, periods in built.items()},
            "carried_periods": {
                label: sum(1 for period in periods if period["carried"]) for label, periods in built.items()
            },
            "periods_without_a_book": {
                label: sum(1 for period in periods if not period["days"]) for label, periods in built.items()
            },
            "overlap_days": {label: _overlap_days(periods) for label, periods in built.items()},
        },
        "reconcile": {
            "tolerance": RECONCILE_TOLERANCE,
            "gross_max_abs_gap": gaps,
            "gross_passed": all(value is None or value <= RECONCILE_TOLERANCE for value in gaps.values()),
            "net_gap_vs_period_row": net_gaps,
            "net_gap_bound": bounds,
            "note": (
                "gross_max_abs_gap is the chained daily value of a period minus its row's "
                "1 + gross_return, and must be ~0: it is the only check that the daily path "
                "describes the same strategy. net_gap_vs_period_row is expected to be small "
                "and no larger than net_gap_bound, because the period row charges cost "
                "additively and this series charges it multiplicatively."
            ),
        },
    }


def _max_gap(periods: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [abs(float(period[key])) for period in periods if period.get(key) is not None]
    return max(values) if values else None
