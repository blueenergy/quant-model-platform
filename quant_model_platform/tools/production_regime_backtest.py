"""Production-style backtest of model windows with the live regime overlay.

Uses the demo's own execution rules (next open, one-price limit-up/down,
13.5 bps per side) and the frozen production regime schedule. The only inputs
from the ML experiment are the full cross-section predictions on each
rebalance date; everything else is the production-adjacent demo engine.

Run with the trainer image and the repo mounted over ``/app``:

    python -m quant_model_platform.tools.production_regime_backtest \
        --predictions-dir /tmp/factor_validation/production_predictions/work \
        --regime /tmp/factor_validation/regime_schedule.parquet \
        --out /tmp/factor_validation/production_backtest
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.demo.portfolio import equity_stats, simulate_track
from quant_model_platform.demo.sources import (
    forward_returns_by_signal_date,
    load_price_histories,
)
from quant_model_platform.evaluation.metrics import cross_sectional_ic, summarize_ic

HOLD_DAYS = 20
TOP_N = 20


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.2%}"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _yearly_returns(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    by_year: Dict[str, List[float]] = {}
    for row in rows:
        net = row.get("net_return")
        if net is None:
            continue
        by_year.setdefault(str(row["trade_date"])[:4], []).append(float(net))
    return {
        year: float(np.prod([1.0 + value for value in values]) - 1.0)
        for year, values in by_year.items()
    }


def _annualized(total_return: float, periods: int) -> float:
    if periods <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (252.0 / (HOLD_DAYS * periods)) - 1.0)


def _cross_sectional_ic(
    scores_by_date: Dict[str, List[Dict[str, Any]]],
    forward_returns: Dict[str, Dict[str, float]],
    signal_dates: Sequence[str],
) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    for day in signal_dates:
        fwd = forward_returns.get(day) or {}
        for item in scores_by_date.get(day) or []:
            symbol = str(item.get("ts_code") or "")
            value = fwd.get(symbol)
            if symbol and value is not None:
                rows.append(
                    {"trade_date": day, "pred": float(item["score"]), "fwd": float(value)}
                )
    if not rows:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}
    frame = pd.DataFrame(rows)
    return summarize_ic(cross_sectional_ic(frame, "pred", "fwd", date_col="trade_date"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Production-style window backtest with regime overlay")
    parser.add_argument(
        "--predictions-dir",
        default="/tmp/factor_validation/production_predictions/work",
    )
    parser.add_argument(
        "--regime",
        default="/tmp/factor_validation/regime_schedule.parquet",
    )
    parser.add_argument("--index-codes", default="hs300,csi500")
    parser.add_argument("--windows", default="rolling_60,rolling_500")
    parser.add_argument("--start", default="20150101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--out", default="/tmp/factor_validation/production_backtest")
    args = parser.parse_args(argv)

    windows = [value.strip() for value in args.windows.split(",") if value.strip()]
    index_codes = [value.strip() for value in args.index_codes.split(",") if value.strip()]
    regime = pd.read_parquet(args.regime)
    regime["trade_date"] = regime["trade_date"].astype(str)

    config = PlatformConfig.from_env()
    client = MongoClient(config.mongo_uri, tz_aware=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    yearly_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    try:
        for pool in index_codes:
            predictions_path = Path(args.predictions_dir) / f"predictions_{pool}.parquet"
            if not predictions_path.exists():
                raise FileNotFoundError(
                    f"missing predictions for {pool}: {predictions_path}. "
                    "Run factor_incremental_experiment with --window-sweep first."
                )
            predictions = pd.read_parquet(predictions_path)
            predictions = predictions[
                (predictions["index_code"] == pool)
                & (predictions["set_name"] == "baseline")
                & (predictions["window"].isin(windows))
            ].copy()
            if predictions.empty:
                raise RuntimeError(f"no prediction rows for {pool} / {windows}")
            predictions["trade_date"] = predictions["trade_date"].astype(str)
            predictions["ts_code"] = predictions["ts_code"].astype(str)
            # A fold can be revisited when a run is resumed; keep the last score
            # so one (date, symbol) never enters a cross-section twice.
            predictions = predictions.drop_duplicates(
                subset=["index_code", "window", "trade_date", "ts_code"],
                keep="last",
            )

            common_dates: Optional[set] = None
            for window in windows:
                dates = set(predictions.loc[predictions["window"] == window, "trade_date"])
                common_dates = dates if common_dates is None else common_dates & dates
            signal_dates = sorted(common_dates or set())
            if not signal_dates:
                raise RuntimeError(f"no common rebalance dates for {pool}")

            symbols = sorted(predictions["ts_code"].astype(str).unique())
            print(f"[{pool}] loading price histories for {len(symbols)} symbols...", flush=True)
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
            pool_regime = regime[regime["index_code"] == pool]
            cash_dates = set(
                pool_regime.loc[pool_regime["action"] == "cash", "trade_date"].astype(str)
            )
            cash_share = float(len([d for d in signal_dates if d in cash_dates]) / len(signal_dates))

            rows_by_key: Dict[tuple, List[Dict[str, Any]]] = {}
            for window in windows:
                frame = predictions[predictions["window"] == window]
                scores_by_date: Dict[str, List[Dict[str, Any]]] = {}
                for day, group in frame.groupby("trade_date", sort=False):
                    scores_by_date[str(day)] = [
                        {"ts_code": str(row.ts_code), "score": float(row.pred)}
                        for row in group.itertuples(index=False)
                    ]
                ic_summary = _cross_sectional_ic(scores_by_date, forward_returns, signal_dates)
                for overlay in (False, True):
                    rows = simulate_track(
                        signal_dates,
                        scores_by_date,
                        forward_returns,
                        top_n=args.top_n,
                        track=f"{window}_{'overlay' if overlay else 'raw'}",
                        cash_dates=cash_dates if overlay else None,
                        rank_start=args.rank_start,
                    )
                    rows_by_key[(window, overlay)] = rows
                    for row in rows:
                        trade_rows.append(
                            {
                                "index_code": pool,
                                "window": window,
                                "overlay": overlay,
                                **row,
                                "holdings": ";".join(
                                    str(symbol) for symbol in (row.get("holdings") or [])
                                ),
                            }
                        )
                    stats = equity_stats(rows, hold_days=HOLD_DAYS)
                    periods = sum(1 for row in rows if row.get("net_return") is not None)
                    yearly = _yearly_returns(rows)
                    for year, value in yearly.items():
                        yearly_rows.append(
                            {
                                "index_code": pool,
                                "window": window,
                                "overlay": overlay,
                                "year": year,
                                "net_return": value,
                            }
                        )
                    summary_rows.append(
                        {
                            "index_code": pool,
                            "window": window,
                            "overlay": overlay,
                            "periods": periods,
                            "ic_mean": ic_summary.get("mean_ic"),
                            "ic_ir": ic_summary.get("ic_ir"),
                            "ann_return_net": _annualized(stats["total_return"], periods),
                            "total_return_net": stats["total_return"],
                            "max_drawdown": stats["max_drawdown"],
                            "worst_year": min(yearly.values()) if yearly else float("nan"),
                            "avg_turnover": float(
                                np.mean([row["one_way_turnover"] for row in rows])
                            ),
                            "total_cost": float(np.sum([row["cost"] for row in rows])),
                            "cash_share": cash_share if overlay else 0.0,
                            "t_stat": stats.get("t_stat"),
                        }
                    )
                print(f"[{pool}] {window} done", flush=True)

            # Paired per-period difference against the first window. The
            # absolute-mean t in the summary treats each window separately; this
            # is the statistic that answers "is rolling_500 better than
            # rolling_60?" because the two rows share the same date, pool and
            # regime state.
            reference_window = windows[0]
            for overlay in (False, True):
                ref_rows = rows_by_key.get((reference_window, overlay)) or []
                ref_map = {
                    str(row["trade_date"]): float(row["net_return"])
                    for row in ref_rows
                    if row.get("net_return") is not None
                }
                for window in windows[1:]:
                    cur_rows = rows_by_key.get((window, overlay)) or []
                    cur_map = {
                        str(row["trade_date"]): float(row["net_return"])
                        for row in cur_rows
                        if row.get("net_return") is not None
                    }
                    common = sorted(set(ref_map) & set(cur_map))
                    diffs = [cur_map[day] - ref_map[day] for day in common]
                    if not diffs:
                        continue
                    mean_diff = float(np.mean(diffs))
                    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
                    se = (
                        std_diff / math.sqrt(len(diffs))
                        if std_diff and not math.isnan(std_diff)
                        else float("nan")
                    )
                    paired_rows.append(
                        {
                            "index_code": pool,
                            "window": window,
                            "reference": reference_window,
                            "overlay": overlay,
                            "periods": len(diffs),
                            "mean_diff": mean_diff,
                            "t_stat": (
                                mean_diff / se
                                if se and not math.isnan(se)
                                else float("nan")
                            ),
                            "mde80": (
                                2.802 * se if se and not math.isnan(se) else float("nan")
                            ),
                            "win_rate": float((np.asarray(diffs) > 0).mean()),
                            "diffs": [
                                {"trade_date": day, "diff": cur_map[day] - ref_map[day]}
                                for day in common
                            ],
                        }
                    )
    finally:
        client.close()

    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    paired = pd.DataFrame(paired_rows)
    trades = pd.DataFrame(trade_rows)

    lines = [
        "# 生产口径回测：四窗口 × regime overlay",
        "",
        f"- 引擎：`demo.sources.holding_path`（next-open，一字涨跌停拒单）+ `demo.portfolio.simulate_track`",
        f"- 书：rank_start={args.rank_start}, top_n={args.top_n}（0 为 top-N）；持有 {HOLD_DAYS} 日；成本 13.5 bp/边（27 bp 全换手）",
        f"- regime：`k_slow_anchor`，bull 持有，非 bull 空仓",
        "",
        "## 汇总",
        "",
        "| 池 | 窗口 | overlay | 期数 | IC | 净年化 | 最大回撤 | 最差年份 | 换手 | 空仓占比 | t |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["index_code", "window", "overlay"]).itertuples(index=False):
        lines.append(
            f"| {row.index_code} | `{row.window}` | {'on' if row.overlay else 'off'} | "
            f"{row.periods} | {_fmt(row.ic_mean, 4)} | {_pct(row.ann_return_net)} | "
            f"{_pct(row.max_drawdown)} | {_pct(row.worst_year)} | "
            f"{_pct(row.avg_turnover)} | {_pct(row.cash_share)} | {_fmt(row.t_stat)} |"
        )

    if not paired.empty:
        lines.extend(
            [
                "",
                "## 配对差（rolling_500 − rolling_60，逐期）",
                "",
                "| 池 | 窗口 | overlay | 期数 | 平均差/期 | t | MDE80 | 胜率 |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in paired.sort_values(
            ["index_code", "window", "overlay"]
        ).itertuples(index=False):
            lines.append(
                f"| {row.index_code} | `{row.window}` | {'on' if row.overlay else 'off'} | "
                f"{row.periods} | {_pct(row.mean_diff)} | {_fmt(row.t_stat)} | "
                f"{_pct(row.mde80)} | {_pct(row.win_rate)} |"
            )

    if not yearly.empty:
        lines.extend(["", "## 分年净收益（overlay on）", ""])
        on = yearly[yearly["overlay"]]
        for pool in sorted(on["index_code"].unique()):
            subset = on[on["index_code"] == pool]
            table = subset.pivot(index="year", columns="window", values="net_return")
            lines.append(f"### {pool}")
            lines.append("")
            lines.append("| 年份 | " + " | ".join(f"`{name}`" for name in table.columns) + " |")
            lines.append("|---|" + "---:|" * len(table.columns))
            for year, row in table.iterrows():
                lines.append(
                    f"| {year} | " + " | ".join(_pct(row[name]) for name in table.columns) + " |"
                )
            lines.append("")

    lines.extend(
        [
            "## 解读注意",
            "",
            "- 这是 ML demo 引擎的生产近似：next-open、一字涨跌停拒单、27bp 全换手成本；不是完整组合引擎。",
            "- 没有组合约束、资金曲线风控、行业/个股上限；绝对收益不是实盘预期。",
            "- 关键问题：rolling_500 相对 rolling_60 的优势在 overlay 下是否还在。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "summary": summary.to_dict(orient="records"),
                "paired": paired.to_dict(orient="records"),
                "yearly": yearly.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if args.save_trades and not trades.empty:
        trades_path = out_dir / "trades.csv"
        trades.to_csv(trades_path, index=False, encoding="utf-8")
        print(f"trades -> {trades_path} rows={len(trades)}")
    print(summary.to_string(index=False))
    print(f"report -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
