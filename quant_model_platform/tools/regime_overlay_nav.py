"""Recompute the training-window NAV with and without the production regime overlay.

Consumes the per-fold records from ``factor_incremental_experiment.py`` and the
frozen regime schedule from ``stock-scoring-system/analysis/research/export_regime_schedule.py``.
No model is retrained: the overlay only changes which 20-day blocks are held in
cash and how turnover is charged at the cash transitions.

Run with the trainer image and the repo mounted over ``/app``:

    python -m quant_model_platform.tools.regime_overlay_nav \
        --report /tmp/factor_validation/window_sweep_nav/report.json \
        --regime /tmp/factor_validation/regime_schedule.parquet \
        --out /tmp/factor_validation/regime_overlay
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quant_model_platform.tools.factor_incremental_experiment import _nav_summary


def _load_regime(path: Path, pool: str) -> pd.Series:
    frame = pd.read_parquet(path)
    frame["trade_date"] = frame["trade_date"].astype(str)
    if "index_code" in frame.columns:
        frame = frame[frame["index_code"] == pool]
    frame = frame.drop_duplicates(subset=["trade_date"], keep="last")
    return frame.set_index("trade_date")["action"]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.2%}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regime-overlay NAV comparison")
    parser.add_argument(
        "--report",
        default="/tmp/factor_validation/window_sweep_nav/report.json",
    )
    parser.add_argument(
        "--regime",
        default="/tmp/factor_validation/regime_schedule.parquet",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=27.0)
    parser.add_argument("--reference", default="rolling_60")
    parser.add_argument("--out", default="/tmp/factor_validation/regime_overlay")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    per_fold = pd.DataFrame(report["per_fold"])
    per_fold = per_fold[per_fold.get("set", "baseline") == "baseline"].copy()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nav_frames: List[pd.DataFrame] = []
    yearly_frames: List[pd.DataFrame] = []
    regime_stats: Dict[str, Dict[str, Any]] = {}
    for pool in sorted(per_fold["index_code"].unique()):
        actions = _load_regime(Path(args.regime), pool)
        switches = int((actions != actions.shift()).sum())
        regime_stats[pool] = {
            "dates": int(len(actions)),
            "cash_share": float((actions == "cash").mean()),
            "switches": switches,
            "avg_hold_days": float(len(actions) / max(switches, 1)),
        }
        group = per_fold[per_fold["index_code"] == pool]
        for overlay in (False, True):
            nav, yearly = _nav_summary(
                group,
                args.reference,
                top_k=args.top_k,
                cost_bps=args.cost_bps,
                actions=actions,
                overlay=overlay,
            )
            if nav.empty:
                continue
            nav["overlay"] = overlay
            nav["index_code"] = pool
            nav_frames.append(nav)
            if not yearly.empty:
                yearly["overlay"] = overlay
                yearly["index_code"] = pool
                yearly_frames.append(yearly)

    nav_all = pd.concat(nav_frames, ignore_index=True)
    yearly_all = (
        pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    )

    # Side-by-side off/on table.
    off = nav_all[~nav_all["overlay"]].set_index(["index_code", "window"])
    on = nav_all[nav_all["overlay"]].set_index(["index_code", "window"])
    comparison = on.join(
        off[
            [
                "ann_return_net",
                "max_drawdown_net",
                "worst_year_net",
                "avg_turnover",
                "cash_share",
            ]
        ],
        lsuffix="_on",
        rsuffix="_off",
    ).reset_index()
    comparison["delta_ann_return"] = (
        comparison["ann_return_net_on"] - comparison["ann_return_net_off"]
    )
    comparison["delta_max_drawdown"] = (
        comparison["max_drawdown_net_on"] - comparison["max_drawdown_net_off"]
    )
    comparison["delta_worst_year"] = (
        comparison["worst_year_net_on"] - comparison["worst_year_net_off"]
    )

    lines = [
        "# 四窗口 × 生产 regime overlay",
        "",
        f"- regime 规则：`k_slow_anchor`（K-fast/K-slow EWMA，5/2 日确认）",
        f"- 基准：hs300 → 000300.SH，csi500 → 000905.SH",
        f"- Top-K：{args.top_k}；每 20 日再平衡；成本 {args.cost_bps:.0f} bp / 100% 换手",
        "- overlay：bull 持有 top-K，非 bull 空仓；现金期收益 0，切换按全额换手计成本",
        "",
        "## regime 状态",
        "",
        "| 池 | 交易日 | 空仓占比 | 切换次数 | 平均持有天数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for pool, stats in regime_stats.items():
        lines.append(
            f"| {pool} | {stats['dates']} | {_pct(stats['cash_share'])} | "
            f"{stats['switches']} | {stats['avg_hold_days']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Overlay off vs on",
            "",
            "| 池 | 窗口 | 净年化(off) | 净年化(on) | Δ净年化 | 最大回撤(off) | 最大回撤(on) | Δ回撤 | 最差年份(on) | 换手(on) | 空仓占比 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.sort_values(["index_code", "window"]).itertuples(index=False):
        lines.append(
            f"| {row.index_code} | `{row.window}` | {_pct(row.ann_return_net_off)} | "
            f"{_pct(row.ann_return_net_on)} | {_pct(row.delta_ann_return)} | "
            f"{_pct(row.max_drawdown_net_off)} | {_pct(row.max_drawdown_net_on)} | "
            f"{_pct(row.delta_max_drawdown)} | {_pct(row.worst_year_net_on)} | "
            f"{_pct(row.avg_turnover_on)} | {_pct(row.cash_share_on)} |"
        )

    lines.extend(["", "## Ranking（overlay on，按净年化）", ""])
    ranking = nav_all[nav_all["overlay"]].sort_values(
        ["index_code", "ann_return_net"], ascending=[True, False]
    )
    lines.append("| 池 | 排名 | 窗口 | 净年化 | 最大回撤 | 最差年份 | Sharpe | 换手 | 空仓占比 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for pool, group in ranking.groupby("index_code", sort=True):
        for rank, row in enumerate(group.itertuples(index=False), start=1):
            lines.append(
                f"| {pool} | {rank} | `{row.window}` | {_pct(row.ann_return_net)} | "
                f"{_pct(row.max_drawdown_net)} | {_pct(row.worst_year_net)} | "
                f"{_fmt(row.sharpe_net)} | {_pct(row.avg_turnover)} | {_pct(row.cash_share)} |"
            )

    if not yearly_all.empty:
        lines.extend(["", "## 分年净收益（overlay on）", ""])
        on_yearly = yearly_all[yearly_all["overlay"]]
        for pool in sorted(on_yearly["index_code"].unique()):
            subset = on_yearly[on_yearly["index_code"] == pool]
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
            "- regime 标签来自生产默认规则，按日期 as-of 计算；空仓期收益按 0 处理。",
            "- 切换成本按全额换手计（卖出或买入各一次），比连续持仓的换手成本更保守。",
            "- 这是研究 NAV，不含涨跌停/停牌顺延、组合约束和资金曲线风控；绝对收益不是实盘预期。",
            "- 关键问题不是「overlay 好不好」，而是 **rolling_500 相对 rolling_60 的优势在 overlay 下是否还在**。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "regime_stats": regime_stats,
                "comparison": comparison.to_dict(orient="records"),
                "nav": nav_all.to_dict(orient="records"),
                "yearly": yearly_all.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(f"report -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
