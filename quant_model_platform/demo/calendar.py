"""Rebalance calendar for the hs300 20-day TOP20 demo."""

from __future__ import annotations

from typing import List, Sequence


def trading_days_per_year(years: int) -> int:
    if years not in (1, 2):
        raise ValueError("years must be 1 or 2")
    return 252 if years == 1 else 504


def window_trading_days(all_days: Sequence[str], *, years: int = 1) -> List[str]:
    """Last N unique sorted YYYYMMDD strings (N = 252 or 504).

    Raises when the history is shorter than the window instead of quietly
    returning what it has: a silent short window makes the report claim two
    years over one year of data (docs/DEMO_HS300_20D_TOP20.md section 5, B1).
    """
    days = sorted({str(d)[:8] for d in all_days if str(d).strip()})
    n = trading_days_per_year(years)
    if len(days) < n:
        span = f"{days[0]}..{days[-1]}" if days else "no dates at all"
        raise ValueError(
            f"requested {n} trading days ({years}y) but only {len(days)} available ({span}); "
            "pass a longer history or a smaller --years"
        )
    return days[-n:]


def rebalance_signal_dates(trading_days: Sequence[str], *, interval: int = 20) -> List[str]:
    """Signal dates at index 0, interval, 2*interval, ... within the window.

    Holding period for date i runs until the next signal (exclusive of next signal day
    for membership; execution is T+1 open — handled by the backtester).
    """
    if interval < 1:
        raise ValueError("interval must be >= 1")
    days = sorted({str(d)[:8] for d in trading_days if str(d).strip()})
    if not days:
        return []
    return [days[i] for i in range(0, len(days), interval)]
