"""hs300 20-day TOP20 demo.

``--dates-file`` writes the rebalance calendar only.
``--bundle`` runs the rules track (A), the model track (B) and the random
control track (R) through the equal-weight next-open equity engine and writes
summary.json (see docs/DEMO_HS300_20D_TOP20.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant_model_platform.demo.calendar import rebalance_signal_dates, window_trading_days
from quant_model_platform.demo.portfolio import (
    CONTROL_SEED,
    COST_PER_SIDE,
    ROUND_TRIP_COST_RATE,
    SELF_COMPUTED_SCORE,
    compare_tracks,
)


def _read_dates(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip()[:8] for line in lines if line.strip() and not line.startswith("#")]


def _write_equity_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "trade_date",
        "baseline",
        "model",
        "random",
        "baseline_pricing",
        "model_pricing",
        "random_pricing",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rebalances_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "trade_date",
        "track",
        "holdings",
        "cross_section",
        "n_scored",
        "n_holdings",
        "n_priced",
        "pricing",
        "gross_return",
        "one_way_turnover",
        "cost",
        "net_return",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["holdings"] = " ".join(row.get("holdings") or [])
            writer.writerow(out)


def _resolve_window(days: Any, *, years: int) -> Optional[List[str]]:
    """Window or None; a short history is a hard error, never a quiet shrink."""
    try:
        return window_trading_days(days, years=years)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _cost_block() -> Dict[str, Any]:
    return {
        "per_side_bps": round(COST_PER_SIDE * 10_000, 1),
        "round_trip_bps": round(ROUND_TRIP_COST_RATE * 10_000, 1),
        "source": "alpha_factor_backtest.CostAssumptions",
        "note": "fees 7 bps round trip + slippage 10 bps a side; a full swap pays both sides",
    }


def _caveats(*, top_n: int, rebalance_days: int) -> List[str]:
    return [
        "gross return is the equal-weight mean of forward_return over the hold window (next open to next open)",
        f"net return pays {ROUND_TRIP_COST_RATE * 10_000:.1f} bps per 100% turnover; the opening book pays one side, later swaps both",
        "a signal day with no usable cross-section holds the previous book at zero turnover, it does not go to cash",
        "a period whose forward window has not happened yet is excluded from the NAV and counted in dropped_periods",
        "drawdown and the equity curve are on the rebalance grid, not daily",
        f"one year holds about 252/{rebalance_days} ~= {252 // rebalance_days} independent periods: the NAV cannot support a 'better' claim, only the per-day metrics can",
        f"top_n={top_n} and rebalance_days={rebalance_days} are the run's parameters; the frozen contract's primary/ sensitivity pair is still open (section 9)",
    ]


def _run_bundle(args: argparse.Namespace) -> int:
    payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    window = _resolve_window(payload.get("trading_days") or [], years=args.years)
    if window is None:
        return 2
    signals = rebalance_signal_dates(window, interval=args.rebalance_days)
    report = compare_tracks(
        signals,
        payload.get("scores_a") or {},
        payload.get("scores_b") or {},
        payload.get("forward_returns") or {},
        top_n=args.top_n,
        hold_days=args.rebalance_days,
        score_column_a=args.score_column,
        control_seed=args.control_seed,
    )
    summary = {
        "demo_id": f"demo-hs300-20d-top{args.top_n}-{args.years}y",
        "status": "equity",
        "years": args.years,
        "top_n": args.top_n,
        "rebalance_days": args.rebalance_days,
        "score_column": report["track_a"]["score_column"],
        "model_id": args.model_id,
        "source_job_id": args.source_job_id,
        "cost": _cost_block(),
        "window_start": window[0] if window else None,
        "window_end": window[-1] if window else None,
        "window_trading_days": len(window),
        "rebalance_count": len(signals),
        "rebalance_dates": signals,
        "track_a": report["track_a"],
        "track_b": report["track_b"],
        "track_r": report["track_r"],
        "control_r": report["control_r"],
        "comparison": report["comparison"],
        "caveats": _caveats(top_n=args.top_n, rebalance_days=args.rebalance_days),
    }
    text = json.dumps(summary, indent=2)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
        _write_equity_csv(args.out_dir / "equity_curve.csv", report["equity_curve"])
        _write_rebalances_csv(args.out_dir / "rebalances.csv", report["rebalances"])
        print(f"wrote {args.out_dir / 'summary.json'}")
    else:
        print(text)
    return 0


def _run_calendar(args: argparse.Namespace) -> int:
    window = _resolve_window(_read_dates(args.dates_file), years=args.years)
    if window is None:
        return 2
    signals = rebalance_signal_dates(window, interval=args.rebalance_days)
    payload = {
        "demo_id": f"demo-hs300-20d-top{args.top_n}-{args.years}y",
        "status": "calendar_only",
        "years": args.years,
        "top_n": args.top_n,
        "rebalance_days": args.rebalance_days,
        "score_column": args.score_column or SELF_COMPUTED_SCORE,
        "model_id": args.model_id,
        "source_job_id": args.source_job_id,
        "window_start": window[0] if window else None,
        "window_end": window[-1] if window else None,
        "window_trading_days": len(window),
        "rebalance_count": len(signals),
        "rebalance_dates": signals,
    }
    text = json.dumps(payload, indent=2)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "calendar.json").write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out_dir / 'calendar.json'}")
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="hs300 20-day TOP20 demo")
    parser.add_argument("--years", type=int, default=1, choices=(1, 2))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--rebalance-days", type=int, default=20)
    parser.add_argument(
        "--score-column",
        default="",
        help=f"stock_scores column to score track A by; empty = self-computed blend ({SELF_COMPUTED_SCORE})",
    )
    parser.add_argument("--model-id", default="promo-hs300-2bc005de")
    parser.add_argument("--source-job-id", default="2bc005de-1ca1-4146-bd03-1e19fc31abfd")
    parser.add_argument("--control-seed", type=int, default=CONTROL_SEED, help="track R seed; rerun with another to check R is not a fluke")
    parser.add_argument("--dates-file", type=Path, help="YYYYMMDD per line (calendar only)")
    parser.add_argument("--bundle", type=Path, help="JSON with trading_days, scores_a, scores_b, forward_returns")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    if args.bundle:
        return _run_bundle(args)
    if not args.dates_file:
        print("pass --dates-file for a calendar, or --bundle for A/B equity")
        return 0
    return _run_calendar(args)


if __name__ == "__main__":
    raise SystemExit(main())
