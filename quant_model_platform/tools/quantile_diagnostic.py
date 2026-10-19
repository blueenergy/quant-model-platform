"""Quantile decomposition of model rankings on the production rebalance grid.

Answers a narrower question than the NAV: where does the IC come from? Bucket 1
is the top-scored tenth, bucket 10 the bottom. If bucket 1's excess is flat
while bucket 10 is deeply negative, the signal is a short-side effect and no
long-only top-N will harvest it.

Run with the trainer image and the repo mounted over ``/app``:

    python -m quant_model_platform.tools.quantile_diagnostic \
        --predictions-dir /tmp/factor_validation/production_predictions/work \
        --index-codes csi1000 \
        --out /tmp/factor_validation/quantile_csi1000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.demo.primary import quantile_rows, summarize_quantiles
from quant_model_platform.demo.sources import (
    forward_returns_by_signal_date,
    load_price_histories,
)

HOLD_DAYS = 20


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.2%}"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quantile decomposition diagnostic")
    parser.add_argument(
        "--predictions-dir",
        default="/tmp/factor_validation/production_predictions/work",
    )
    parser.add_argument("--index-codes", default="csi1000")
    parser.add_argument("--windows", default="rolling_60,rolling_500,expanding")
    parser.add_argument("--start", default="20150101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--out", default="/tmp/factor_validation/quantile_csi1000")
    args = parser.parse_args(argv)

    windows = [value.strip() for value in args.windows.split(",") if value.strip()]
    index_codes = [value.strip() for value in args.index_codes.split(",") if value.strip()]
    config = PlatformConfig.from_env()
    client = MongoClient(config.mongo_uri, tz_aware=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    try:
        for pool in index_codes:
            predictions = pd.read_parquet(
                Path(args.predictions_dir) / f"predictions_{pool}.parquet"
            )
            predictions = predictions[
                (predictions["index_code"] == pool)
                & (predictions["set_name"] == "baseline")
                & (predictions["window"].isin(windows))
            ].copy()
            if predictions.empty:
                raise RuntimeError(f"no prediction rows for {pool}")
            predictions["trade_date"] = predictions["trade_date"].astype(str)
            predictions["ts_code"] = predictions["ts_code"].astype(str)
            predictions = predictions.drop_duplicates(
                subset=["index_code", "window", "trade_date", "ts_code"],
                keep="last",
            )

            common_dates: Optional[set] = None
            for window in windows:
                dates = set(
                    predictions.loc[predictions["window"] == window, "trade_date"]
                )
                common_dates = dates if common_dates is None else common_dates & dates
            signal_dates = sorted(common_dates or set())
            if not signal_dates:
                raise RuntimeError(f"no common signal dates for {pool}")

            symbols = sorted(predictions["ts_code"].unique())
            print(f"[{pool}] loading prices for {len(symbols)} symbols...", flush=True)
            prices = load_price_histories(
                client,
                data_db_name=config.data_db_name,
                symbols=symbols,
                start_date=args.start,
                end_date=args.end,
            )
            forward_returns = forward_returns_by_signal_date(
                prices, signal_dates, horizon=HOLD_DAYS
            )

            for window in windows:
                frame = predictions[predictions["window"] == window]
                scores_by_date: Dict[str, List[Dict[str, Any]]] = {}
                for day, group in frame.groupby("trade_date", sort=False):
                    scores_by_date[str(day)] = [
                        {"ts_code": str(row.ts_code), "score": float(row.pred)}
                        for row in group.itertuples(index=False)
                    ]
                track_rows = {
                    "model": [
                        {"trade_date": day, "cross_section": "selected"}
                        for day in signal_dates
                    ]
                }
                pool_symbols = {
                    day: [item["ts_code"] for item in scores_by_date.get(day) or []]
                    for day in signal_dates
                }
                rows = quantile_rows(
                    {"model": scores_by_date},
                    forward_returns,
                    track_rows,
                    pool_symbols=pool_symbols,
                    buckets=args.buckets,
                )
                result = summarize_quantiles(rows, hold_days=HOLD_DAYS)
                blocks = result["tracks"]["model"]["buckets"]
                spread = result["tracks"]["model"]["spread_top_bottom"]
                top = blocks[0]
                bottom = blocks[-1]
                weighted = [
                    (float(block["n_mean"]), float(block["mean"]))
                    for block in blocks
                    if block.get("n_mean") and block.get("mean") is not None
                ]
                total_n = sum(size for size, _ in weighted)
                pool_mean = (
                    sum(size * value for size, value in weighted) / total_n
                    if total_n
                    else float("nan")
                )
                top_excess = top.get("excess_mean")
                bottom_excess = bottom.get("excess_mean")
                denominator = abs(top_excess or 0.0) + abs(bottom_excess or 0.0)
                short_share = (
                    abs(bottom_excess) / denominator
                    if denominator and bottom_excess is not None
                    else float("nan")
                )
                summary_rows.append(
                    {
                        "index_code": pool,
                        "window": window,
                        "dates": len(signal_dates),
                        "pool_mean": pool_mean,
                        "top_bucket": 1,
                        "top_mean": top.get("mean"),
                        "top_excess": top_excess,
                        "top_excess_t": top.get("excess_t_stat"),
                        "top_excess_mde": top.get("excess_mde_80"),
                        "top_n_mean": top.get("n_mean"),
                        "bottom_bucket": args.buckets,
                        "bottom_mean": bottom.get("mean"),
                        "bottom_excess": bottom_excess,
                        "bottom_excess_t": bottom.get("excess_t_stat"),
                        "spread": spread.get("mean"),
                        "spread_t": spread.get("t_stat"),
                        "spread_mde": spread.get("mde_80"),
                        "monotonicity": result["tracks"]["model"].get("monotonicity"),
                        "short_side_share": short_share,
                    }
                )
                for block in blocks:
                    bucket_rows.append(
                        {
                            "index_code": pool,
                            "window": window,
                            "bucket": block.get("bucket"),
                            "n_mean": block.get("n_mean"),
                            "mean": block.get("mean"),
                            "excess_mean": block.get("excess_mean"),
                            "excess_t": block.get("excess_t_stat"),
                        }
                    )
                print(f"[{pool}] {window} quantiles done", flush=True)
    finally:
        client.close()

    summary = pd.DataFrame(summary_rows)
    buckets = pd.DataFrame(bucket_rows)

    lines = [
        "# 分位分解诊断",
        "",
        f"- 池：{', '.join(index_codes)}；bucket 1 = 最高分，bucket {args.buckets} = 最低分",
        f"- 每 20 日一个截面；收益为 next-open 到 next-open；超额 = 该 bucket 减去当日池均值",
        "",
        "## 摘要",
        "",
        "| 池 | 窗口 | Q1 超额 | t | Q10 超额 | t | Q1−Q10 | t | 单调性 | 空头贡献占比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["index_code", "window"]).itertuples(index=False):
        lines.append(
            f"| {row.index_code} | `{row.window}` | {_pct(row.top_excess)} | "
            f"{_fmt(row.top_excess_t)} | {_pct(row.bottom_excess)} | "
            f"{_fmt(row.bottom_excess_t)} | {_pct(row.spread)} | {_fmt(row.spread_t)} | "
            f"{_fmt(row.monotonicity)} | {_pct(row.short_side_share)} |"
        )

    lines.extend(
        [
            "",
            "## 判断口径",
            "",
            "- Q1 超额 ≤ 0 而 Q10 超额显著为负：IC 由空头端驱动，long-only 拿不到。",
            "- Q1 超额显著为正：多头端有信号；此时 top-N 的问题才可能靠选股/流动性解决。",
            "- 空头贡献占比 = |Q10 超额| / (|Q1 超额| + |Q10 超额|)；> 0.5 表示更多来自底部。",
            "",
            "## 逐 bucket",
            "",
            "| 池 | 窗口 | bucket | 平均持仓数 | 原始收益 | 超额 | t |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in buckets.sort_values(["index_code", "window", "bucket"]).itertuples(index=False):
        lines.append(
            f"| {row.index_code} | `{row.window}` | {row.bucket} | {_fmt(row.n_mean, 0)} | "
            f"{_pct(row.mean)} | {_pct(row.excess_mean)} | {_fmt(row.excess_t)} |"
        )

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(
            {"summary": summary.to_dict(orient="records"), "buckets": buckets.to_dict(orient="records")},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"report -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
