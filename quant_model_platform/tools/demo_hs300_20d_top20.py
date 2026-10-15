"""hs300 20-day TOP20 demo.

``--dates-file`` writes the rebalance calendar only.
``--bundle`` runs the rules track (A), the model track (B) and the random
control track (R) through the equal-weight next-open equity engine and writes
summary.json (see docs/DEMO_HS300_20D_TOP20.md).
``--from-mongo`` builds the same payload from this machine's own Mongo
(``stock_scores`` membership as of each day, hfq opens, forward returns), so
the demo needs no hand-assembled fixture and no remote artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.demo import sources
from quant_model_platform.demo.calendar import rebalance_signal_dates, window_trading_days
from quant_model_platform.demo.portfolio import (
    CONTROL_SEED,
    COST_PER_SIDE,
    ROUND_TRIP_COST_RATE,
    SELF_COMPUTED_SCORE,
    compare_tracks,
)
from quant_model_platform.demo.track_b import resolve_model_scores


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


def _demo_id(args: argparse.Namespace, index_code: str) -> str:
    """Name the run by what it measured, not by the contract's headline title."""
    return f"demo-{index_code}-{args.rebalance_days}d-top{args.top_n}-{args.years}y"


def _source_block(
    args: argparse.Namespace, *, mongo_db: Optional[str] = None, data_db: Optional[str] = None
) -> Dict[str, Any]:
    """Where the run actually read from: the resolved databases, not the flags."""
    if args.from_mongo:
        return {
            "kind": "mongo",
            "mongo_db": mongo_db or args.mongo_db,
            "data_db": data_db or args.data_db,
            "index_code": args.index_code,
            "coverage_start": args.start_date,
            "coverage_end": args.end_date,
            "model_snapshot": str(args.model_snapshot) if args.model_snapshot else None,
            "model_kind": args.model_kind if args.model_snapshot else None,
            "shadow_live_from": args.shadow_live_from or None,
        }
    return {"kind": "bundle", "bundle": str(args.bundle) if args.bundle else None}


def _emit_report(
    args: argparse.Namespace,
    payload: Dict[str, Any],
    *,
    index_code: str,
    source: Optional[Dict[str, Any]] = None,
) -> int:
    window = _resolve_window(payload.get("trading_days") or [], years=args.years)
    if window is None:
        return 2
    signals = rebalance_signal_dates(window, interval=args.rebalance_days)
    scores_b = payload.get("scores_b") or {}
    if not scores_b:
        print("warning: track B has no scores; only A and the control R carry a book", file=sys.stderr)
    report = compare_tracks(
        signals,
        payload.get("scores_a") or {},
        scores_b,
        payload.get("forward_returns") or {},
        top_n=args.top_n,
        hold_days=args.rebalance_days,
        score_column_a=args.score_column,
        control_seed=args.control_seed,
    )
    summary = {
        "demo_id": _demo_id(args, index_code),
        "status": "equity",
        "years": args.years,
        "top_n": args.top_n,
        "rebalance_days": args.rebalance_days,
        "score_column": report["track_a"]["score_column"],
        "model_id": args.model_id,
        "source_job_id": args.source_job_id,
        "source": source or _source_block(args),
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


def _run_bundle(args: argparse.Namespace) -> int:
    payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    return _emit_report(args, payload, index_code=args.index_code)


def _model_scores(args: argparse.Namespace, signals: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Track B from a local snapshot: shadow rows when they exist, PIT refit otherwise."""
    if not args.model_snapshot:
        return {}
    panel = pd.read_parquet(args.model_snapshot)
    params = {"model_kind": args.model_kind}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for day in signals:
        out[str(day)[:8]] = resolve_model_scores(
            [],
            panel,
            params,
            day,
            shadow_live_from=args.shadow_live_from,
        )
    return out


def _run_mongo(args: argparse.Namespace) -> int:
    cfg = PlatformConfig.from_env()
    mongo_db = args.mongo_db or cfg.mongo_db
    data_db = args.data_db or cfg.data_db_name
    client = MongoClient(args.mongo_uri or cfg.mongo_uri, tz_aware=True)
    try:
        docs = sources.load_member_score_docs(
            client,
            mongo_db=mongo_db,
            index_code=args.index_code,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if not docs:
            print(
                f"error: no {args.index_code} rows in {mongo_db}.stock_scores between "
                f"{args.start_date} and {args.end_date}",
                file=sys.stderr,
            )
            return 2
        days = sorted({str(doc["score_date"])[:8] for doc in docs if doc.get("score_date")})
        window = _resolve_window(days, years=args.years)
        if window is None:
            return 2
        signals = rebalance_signal_dates(window, interval=args.rebalance_days)
        payload = sources.build_bundle(
            client,
            docs=docs,
            signal_dates=signals,
            data_db_name=data_db,
            start_date=window[0],
            end_date=window[-1],
            horizon=args.rebalance_days,
            score_column=args.score_column,
        )
        payload["scores_b"] = _model_scores(args, signals)
        payload["trading_days"] = days
    finally:
        client.close()
    return _emit_report(
        args,
        payload,
        index_code=args.index_code,
        source=_source_block(args, mongo_db=mongo_db, data_db=data_db),
    )


def _run_calendar(args: argparse.Namespace) -> int:
    window = _resolve_window(_read_dates(args.dates_file), years=args.years)
    if window is None:
        return 2
    signals = rebalance_signal_dates(window, interval=args.rebalance_days)
    payload = {
        "demo_id": _demo_id(args, args.index_code),
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
    parser.add_argument("--model-id", default="", help="label for track B's model; empty = unspecified")
    parser.add_argument("--source-job-id", default="", help="training job behind track B; empty = unspecified")
    parser.add_argument("--control-seed", type=int, default=CONTROL_SEED, help="track R seed; rerun with another to check R is not a fluke")
    parser.add_argument("--dates-file", type=Path, help="YYYYMMDD per line (calendar only)")
    parser.add_argument("--bundle", type=Path, help="JSON with trading_days, scores_a, scores_b, forward_returns")
    parser.add_argument("--index-code", default="hs300")
    parser.add_argument("--from-mongo", action="store_true", help="read scores, hfq prices and returns from local Mongo")
    parser.add_argument("--mongo-uri", help="override MONGO_URI")
    parser.add_argument("--mongo-db", help="score DB; default MONGO_DB")
    parser.add_argument("--data-db", help="price DB; default DATA_DB_NAME")
    parser.add_argument("--start-date", help="coverage start YYYYMMDD (--from-mongo)")
    parser.add_argument("--end-date", help="coverage end YYYYMMDD (--from-mongo)")
    parser.add_argument("--model-snapshot", type=Path, help="local panel parquet that track B refits from")
    parser.add_argument("--model-kind", default="ridge", help="model fitted by the track B backfill")
    parser.add_argument("--shadow-live-from", default="", help="stored shadow scores replace the backfill from this date")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    if args.from_mongo:
        if not args.start_date or not args.end_date:
            print("error: --from-mongo needs --start-date and --end-date", file=sys.stderr)
            return 2
        return _run_mongo(args)
    if args.bundle:
        return _run_bundle(args)
    if not args.dates_file:
        print("pass --dates-file for a calendar, --bundle for fixture A/B equity, or --from-mongo for local data")
        return 0
    return _run_calendar(args)


if __name__ == "__main__":
    raise SystemExit(main())
