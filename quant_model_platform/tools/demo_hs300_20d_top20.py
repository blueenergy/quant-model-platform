"""hs300 20-day TOP20 demo.

``--dates-file`` writes the rebalance calendar only.
``--bundle`` runs the rules track (A), the model track (B) and the random
control track (R) through the equal-weight next-open equity engine and writes
summary.json (see docs/DEMO_HS300_20D_TOP20.md).
``--from-mongo`` builds the same payload from this machine's own Mongo
(``stock_scores`` membership as of each day, hfq opens, forward returns), so
the demo needs no hand-assembled fixture and no remote artifact.

Track B comes from the same local ``quant_ml`` the shadow loop writes to:
``--source-job-id`` pulls that job's recipe (model kind, features, params) and
its canonical Parquet snapshot, and stored ``model_predictions`` are used as-is
on/after ``--shadow-live-from``. ``--check-overlap`` measures whether the PIT
backfill ranks a day the way the live recipe does.
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
from quant_model_platform.db import get_ml_collections
from quant_model_platform.demo import sources
from quant_model_platform.demo.backfill_pit import predict_cross_section_as_of
from quant_model_platform.demo.calendar import rebalance_signal_dates, window_trading_days
from quant_model_platform.demo.portfolio import (
    CONTROL_SEED,
    COST_PER_SIDE,
    ROUND_TRIP_COST_RATE,
    SELF_COMPUTED_SCORE,
    compare_tracks,
)
from quant_model_platform.demo.provenance import build_provenance, write_provenance
from quant_model_platform.demo.track_b import (
    cross_section_rank_correlation,
    load_shadow_predictions,
    resolve_model_scores,
    scores_by_day,
    training_evidence,
)
from quant_model_platform.protocol import MAX_PURGE_DAYS
from quant_model_platform.shadow.resolve import merge_job_into_shadow_params
from quant_model_platform.training.panel_loader import load_training_panel


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


def _write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Daily primary evidence, one row per (date, track).

    The pool columns repeat on every row rather than being stated once in the
    header: a benchmark that changes halfway down the file would then be
    impossible to miss, and the file stays readable on its own. An absent number
    -- no IC on a carried date, no excess on an unpriced one -- is written as an
    empty cell, not as the string ``None``, which would read back as a category.
    """
    fieldnames = [
        "trade_date",
        "track",
        "pool_basis",
        "n_pool",
        "pool_return",
        "pool_return_wide",
        "n_wide",
        "n_scored",
        "n_joined",
        "n_priced",
        "pricing",
        "carried",
        "ic_rank",
        "ic_pearson",
        "topk_mean",
        "excess_gross",
        "cost",
        "excess_net",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames})


def _write_quantiles_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """The ranking's shape, one row per (date, track, bucket), best bucket first.

    Same discipline as ``metrics_by_date.csv``: the pool return repeats on every
    row so a benchmark that moved halfway down the file could not go unnoticed,
    and an absent number is an empty cell rather than the string ``None``.
    """
    fieldnames = [
        "trade_date",
        "track",
        "bucket",
        "n",
        "mean_return",
        "pool_return",
        "excess_return",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames})


def _write_daily_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Session-by-session NAV per book, one row per trading day.

    ``rebalance_date`` is the signal date whose holding window owns the day, so
    the file joins back to ``rebalances.csv`` without guessing. ``_net`` and
    ``_gross`` differ only by the cost paid at each rebalance; ``_vs_pool`` is
    the book over the benchmark, which is the series a drawdown claim should be
    read off -- three fully invested books mostly draw down with the market.
    """
    fieldnames = [
        "trade_date",
        "rebalance_date",
        "baseline_gross",
        "baseline_net",
        "model_gross",
        "model_net",
        "control_random_gross",
        "control_random_net",
        "pool_gross",
        "pool_net",
        "baseline_vs_pool",
        "model_vs_pool",
        "control_random_vs_pool",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames})


def _resolve_window(days: Any, *, years: int) -> Optional[List[str]]:
    """Window or None; a short history is a hard error, never a quiet shrink.

    ``years <= 0`` means the whole history handed in: ``--start-date/--end-date``
    already say how far back the data goes, and the point of the longer window is
    to stop re-choosing it. A positive value still truncates to the last N years.
    """
    if years <= 0:
        whole = sorted({str(d)[:8] for d in days if str(d).strip()})
        if not whole:
            print("error: no trading days in the window", file=sys.stderr)
            return None
        return whole
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


def _caveats(*, top_n: int, rebalance_days: int, backfill_purge: int, rebalance_count: int = 0) -> List[str]:
    return [
        "gross return is the equal-weight mean of forward_return over the hold window (next open to next open)",
        f"net return pays {ROUND_TRIP_COST_RATE * 10_000:.1f} bps per 100% turnover; the opening book pays one side, later swaps both",
        "a signal day with no usable cross-section holds the previous book at zero turnover, it does not go to cash",
        "a period whose forward window has not happened yet is excluded from the NAV and counted in dropped_periods",
        "drawdown appears twice: the track blocks take it on the rebalance grid (about 13 points a year), daily.tracks takes it every session; the daily one is the measurement, the grid one is kept for continuity",
        "daily.relative is the book over the pool; absolute drawdowns of three fully invested books are mostly the market's",
        f"the window holds about {rebalance_count} independent {rebalance_days}-day periods; the NAV cannot support a 'better' claim, only the per-period metrics can, and they need the count to be large",
        f"top_n={top_n} matches the contract's primary (10) unless overridden; rebalance_days={rebalance_days} follows the live preset",
        f"track B's backfill trains with a {backfill_purge}-day embargo; the live scorer uses none, and the ranking moves with that choice (see model_overlap)",
    ]


def _demo_id(args: argparse.Namespace, index_code: str, window: Sequence[str] = ()) -> str:
    """Name the run by what it measured, not by the contract's headline title.

    The window goes in as its real bounds rather than as ``{years}y``: a label is
    a claim about the data, and once ``--years 0`` means "everything you handed
    in" the label can disagree with the window it names.
    """
    if window:
        span = f"{str(window[0])[:8]}-{str(window[-1])[:8]}"
    else:
        span = f"{args.years}y"
    return f"demo-{index_code}-{args.rebalance_days}d-top{args.top_n}-{span}"


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
        trading_days=window,
        # A bundle carries forward returns built by whoever wrote it, at a
        # horizon only the flag can name; --from-mongo builds them from that
        # same flag, so there the label is a fact rather than an assumption.
        horizon_declared_by="rebalance_days" if args.from_mongo else "rebalance_days (assumed; a bundle declares none)",
        prices=payload.get("prices"),
    )
    primary = report["primary"]
    daily = report["daily"]
    if daily is None:
        print(
            "note: no daily NAV -- the payload carried no price history, so drawdown stays on the "
            "rebalance grid (a hand-written bundle has no prices; --from-mongo does)",
            file=sys.stderr,
        )
    elif not daily["reconcile"]["gross_passed"]:
        print(
            "warning: the daily NAV does not reconcile with the period rows; the drawdown below "
            "describes a different book than the returns do",
            file=sys.stderr,
        )
    if not primary["power"]["n_dates"] or not primary["join"]["joined_names"]:
        print(
            "warning: the primary evidence is empty -- no dated cross-section joined to a forward "
            "return -- so only the NAV-level metrics below mean anything",
            file=sys.stderr,
        )
    summary = {
        "demo_id": _demo_id(args, index_code, window),
        "status": "equity",
        "years": args.years,
        "top_n": args.top_n,
        "rebalance_days": args.rebalance_days,
        "backfill_purge_days": args.backfill_purge,
        "score_column": report["track_a"]["score_column"],
        "model_id": args.model_id,
        "source_job_id": args.source_job_id,
        "source": source or _source_block(args),
        "model_source": payload.get("model_source"),
        "model_overlap": payload.get("model_overlap"),
        "model_training": payload.get("model_training"),
        "primary": primary,
        "daily": daily,
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
        "caveats": _caveats(
            top_n=args.top_n,
            rebalance_days=args.rebalance_days,
            backfill_purge=args.backfill_purge,
            rebalance_count=len(signals),
        ),
    }
    text = json.dumps(summary, indent=2)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
        _write_equity_csv(args.out_dir / "equity_curve.csv", report["equity_curve"])
        _write_rebalances_csv(args.out_dir / "rebalances.csv", report["rebalances"])
        _write_metrics_csv(args.out_dir / "metrics_by_date.csv", report["metrics_by_date"])
        _write_quantiles_csv(args.out_dir / "quantiles_by_date.csv", report["quantiles_by_date"])
        if report["daily_nav"]:
            _write_daily_csv(args.out_dir / "daily_nav.csv", report["daily_nav"])
        # Last, so its digest list covers everything else in the directory.
        metadata = build_provenance(
            args=args,
            index_code=index_code,
            window=window,
            signals=signals,
            source=source or _source_block(args),
            model_source=payload.get("model_source"),
            payload=payload,
            out_dir=args.out_dir,
        )
        write_provenance(args.out_dir, metadata)
        print(f"wrote {args.out_dir / 'summary.json'}")
    else:
        print(text)
    return 0


def _run_bundle(args: argparse.Namespace) -> int:
    payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    return _emit_report(args, payload, index_code=args.index_code)


def _model_context(args: argparse.Namespace, cfg: PlatformConfig, *, index_code: str) -> Optional[Dict[str, Any]]:
    """Track B's panel, recipe and stored predictions, all from the local ``quant_ml``.

    With ``--source-job-id`` the recipe and the canonical snapshot come out of the
    same resolver the shadow loop uses, so the demo cannot drift from the live
    scorer through hand-copied parameters (docs section 1, section 4).
    """
    if not args.source_job_id and not args.model_snapshot:
        return None
    collections = get_ml_collections(config=cfg)
    provenance: Dict[str, Any] = {"kind": "job" if args.source_job_id else "snapshot"}
    if args.source_job_id:
        merged = merge_job_into_shadow_params({"source_job_id": args.source_job_id}, collections)
        job_index = str(merged.get("index_code") or "")
        if job_index and job_index != index_code:
            raise ValueError(f"job index_code={job_index} but --index-code={index_code}")
        panel, snapshot_id, panel_path = load_training_panel(
            merged, parquet_root=Path(cfg.parquet_root), config=cfg
        )
        params = merged
        provenance.update(
            {"source_job_id": args.source_job_id, "snapshot_id": snapshot_id, "parquet_path": panel_path}
        )
    else:
        panel = pd.read_parquet(args.model_snapshot)
        params = {"model_kind": args.model_kind}
        provenance.update({"snapshot_id": None, "parquet_path": str(args.model_snapshot)})
    params.setdefault("model_kind", args.model_kind)
    model_id = args.model_id or str(params.get("model_id") or "")
    live = load_shadow_predictions(
        collections,
        model_id=model_id,
        index_code=index_code,
        start=args.start_date or "",
        end=args.end_date or "",
    )
    provenance.update(
        {
            "model_id": model_id,
            "model_kind": params.get("model_kind"),
            "live_rows": len(live),
            "live_dates": sorted(scores_by_day(live)),
            "shadow_live_from": args.shadow_live_from or None,
        }
    )
    # What the source job concluded about its own model travels with the demo
    # output: the evidence into the summary, its two identifiers into the
    # provenance record. Neither is recomputed here.
    training = training_evidence(collections, job_id=args.source_job_id)
    if training:
        provenance.update({"result_id": training["result_id"], "artifact_hash": training["artifact_hash"]})
    return {"panel": panel, "params": params, "live_rows": live, "provenance": provenance, "training": training}


def _model_scores(
    args: argparse.Namespace, signals: Sequence[str], context: Optional[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Track B per signal day: live rows on/after the boundary, PIT refit before."""
    if not context:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for day in signals:
        out[str(day)[:8]] = resolve_model_scores(
            context["live_rows"],
            context["panel"],
            context["params"],
            day,
            shadow_live_from=args.shadow_live_from,
            purge=args.backfill_purge,
        )
    return out


def _backfill_rows(
    day: str, context: Dict[str, Any], *, purge: int, already: Sequence[Dict[str, Any]] = ()
) -> List[Dict[str, Any]]:
    """Protocol backfill for one day, reusing already-scored rows when possible."""
    rows = [row for row in already if str(row.get("score_source") or "") == "backfill_pit"]
    if rows:
        return rows
    frame = predict_cross_section_as_of(context["panel"], context["params"], day, purge=purge)
    return frame.to_dict(orient="records")


def _overlap_report(
    context: Dict[str, Any],
    signals: Sequence[str],
    scores_b: Dict[str, List[Dict[str, Any]]],
    *,
    purge: int = MAX_PURGE_DAYS,
) -> Dict[str, Any]:
    """Does the backfill rank the way the live recipe does?

    Two comparisons, both rank-based because the training windows differ and so
    the score levels do:

    * ``live_vs_backfill`` -- stored shadow rows against the protocol backfill on
      the days both cover. This is the contract's acceptance check (section 4).
    * ``purge_effect`` -- the protocol backfill against the same refit with the
      embargo removed, on every signal day. The live scorer refits on all labelled
      rows, so this isolates how much of the gap is the embargo itself.
    """
    report: Dict[str, Any] = {
        "purge_days": purge,
        "live_vs_backfill": None,
        "purge_effect": None,
        "note": "rank correlation, per shared day; threshold in demo/track_b.py",
    }
    live = context["live_rows"]
    live_days = sorted(scores_by_day(live))
    if live_days:
        backfill = []
        for day in live_days:
            backfill.extend(_backfill_rows(day, context, purge=purge, already=scores_b.get(day, ())))
        report["live_vs_backfill"] = cross_section_rank_correlation(live, backfill)
    unpurged, purged = [], []
    for day in signals:
        key = str(day)[:8]
        unpurged.extend(
            predict_cross_section_as_of(context["panel"], context["params"], key, purge=0).to_dict("records")
        )
        purged.extend(_backfill_rows(key, context, purge=purge, already=scores_b.get(key, ())))
    report["purge_effect"] = cross_section_rank_correlation(unpurged, purged)
    return report


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
        try:
            context = _model_context(args, cfg, index_code=args.index_code)
        except (ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
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
        payload["scores_b"] = _model_scores(args, signals, context)
        payload["trading_days"] = days
        if context:
            payload["model_source"] = context["provenance"]
            payload["model_training"] = context["training"]
            if args.check_overlap:
                payload["model_overlap"] = _overlap_report(
                    context, signals, payload["scores_b"], purge=args.backfill_purge
                )
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
        "demo_id": _demo_id(args, args.index_code, window),
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
    parser.add_argument(
        "--years",
        type=int,
        default=0,
        help="truncate the window to the last N years; 0 (default) uses every date in the coverage",
    )
    parser.add_argument("--top-n", type=int, default=10, help="names held per period; the contract's primary is 10")
    parser.add_argument("--rebalance-days", type=int, default=20)
    parser.add_argument(
        "--score-column",
        default="",
        help=f"stock_scores column to score track A by; empty = self-computed blend ({SELF_COMPUTED_SCORE})",
    )
    parser.add_argument("--model-id", default="", help="label for track B's model; empty = unspecified")
    parser.add_argument(
        "--source-job-id",
        default="",
        help="quant_ml training job behind track B: supplies its recipe and canonical snapshot",
    )
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
    parser.add_argument("--model-snapshot", type=Path, help="panel parquet to refit from when no --source-job-id")
    parser.add_argument("--model-kind", default="ridge", help="model kind used with --model-snapshot")
    parser.add_argument("--shadow-live-from", default="", help="stored shadow scores replace the backfill from this date")
    parser.add_argument(
        "--check-overlap",
        action="store_true",
        help="report how well the backfill reproduces the live recipe (extra refits)",
    )
    parser.add_argument(
        "--backfill-purge",
        type=int,
        default=MAX_PURGE_DAYS,
        help=f"embargo days between the backfill's training window and the signal day; "
        f"the contract freezes {MAX_PURGE_DAYS}, lower it only to measure sensitivity",
    )
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
