"""hs300 20-day TOP20 demo.

``--dates-file`` writes the rebalance calendar only.
``--bundle`` runs both tracks through the equal-weight next-open equity
engine and writes summary.json (see docs/DEMO_HS300_20D_TOP20.md).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from quant_model_platform.demo.calendar import rebalance_signal_dates, window_trading_days
from quant_model_platform.demo.portfolio import compare_tracks


def _read_dates(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip()[:8] for line in lines if line.strip() and not line.startswith("#")]


def _write_equity_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trade_date", "baseline", "model"])
        writer.writeheader()
        writer.writerows(rows)


def _write_rebalances_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["trade_date", "track", "holdings", "n_priced", "gross_return", "one_way_turnover", "cost", "net_return"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["holdings"] = " ".join(row.get("holdings") or [])
            writer.writerow(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="hs300 20-day TOP20 demo")
    parser.add_argument("--years", type=int, default=1, choices=(1, 2))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--rebalance-days", type=int, default=20)
    parser.add_argument("--score-column", default="composite_growth_cycle_score")
    parser.add_argument("--model-id", default="promo-hs300-2bc005de")
    parser.add_argument("--source-job-id", default="2bc005de-1ca1-4146-bd03-1e19fc31abfd")
    parser.add_argument("--dates-file", type=Path, help="YYYYMMDD per line (calendar only)")
    parser.add_argument("--bundle", type=Path, help="JSON with trading_days, scores_a, scores_b, forward_returns")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    if args.bundle:
        payload = json.loads(args.bundle.read_text(encoding="utf-8"))
        window = window_trading_days(payload.get("trading_days") or [], years=args.years)
        signals = rebalance_signal_dates(window, interval=args.rebalance_days)
        report = compare_tracks(
            signals,
            payload.get("scores_a") or {},
            payload.get("scores_b") or {},
            payload.get("forward_returns") or {},
            top_n=args.top_n,
            hold_days=args.rebalance_days,
        )
        summary = {
            "demo_id": "demo-hs300-20d-top20-1y" if args.years == 1 else "demo-hs300-20d-top20-2y",
            "status": "equity",
            "years": args.years,
            "top_n": args.top_n,
            "rebalance_days": args.rebalance_days,
            "score_column": args.score_column,
            "model_id": args.model_id,
            "source_job_id": args.source_job_id,
            "window_start": window[0] if window else None,
            "window_end": window[-1] if window else None,
            "window_trading_days": len(window),
            "rebalance_count": len(signals),
            "rebalance_dates": signals,
            "track_a": report["track_a"],
            "track_b": report["track_b"],
            "comparison": report["comparison"],
            "caveats": [
                "gross return is the equal-weight mean of forward_return over the hold window (next open to next open)",
                "net return subtracts 10 bps per side; the opening book pays one side, later swaps pay both",
                "ScoreBacktester.calculate_topk_summary is the gross building block; it does not emit this NAV",
            ],
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

    if not args.dates_file:
        print("pass --dates-file for a calendar, or --bundle for A/B equity")
        return 0

    window = window_trading_days(_read_dates(args.dates_file), years=args.years)
    signals = rebalance_signal_dates(window, interval=args.rebalance_days)
    payload = {
        "demo_id": "demo-hs300-20d-top20-1y" if args.years == 1 else "demo-hs300-20d-top20-2y",
        "status": "calendar_only",
        "years": args.years,
        "top_n": args.top_n,
        "rebalance_days": args.rebalance_days,
        "score_column": args.score_column,
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


if __name__ == "__main__":
    raise SystemExit(main())
