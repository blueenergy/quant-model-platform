"""Controlled incremental test: do candidate factors add value to the model?

The single-factor health check screened each factor in isolation. This script
answers the model-level question with the same algorithm on both sides:

    baseline  = LightGBM on the existing score + raw-feature set
    treatment = LightGBM on baseline + one candidate feature group

Everything else is held fixed: same panel, same folds, same label, same
hyper-parameters, same cross-sectional rank transform. A treatment that looks
better only because it changed the model or the row set is not evidence.

The factors come from a frozen Parquet snapshot produced by
``stock-scoring-system/analysis/research/export_factor_snapshot.py``. That
keeps the compute code in its own repository and makes the run reproducible.

Run with the trainer image and the repo mounted over ``/app``:

    python -m quant_model_platform.tools.factor_incremental_experiment \
        --factor-parquet /tmp/factor_validation/snapshot/factors_2018_2024.parquet \
        --index-codes hs300,csi500 --start 20180101 --end 20241231 \
        --out /tmp/factor_validation/incremental
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.labels import forward_open_to_open_return
from quant_model_platform.dataset.production import build_production_panel, load_price_history
from quant_model_platform.evaluation.metrics import spearman_ic, summarize_ic
from quant_model_platform.evaluation.walk_forward import WalkForwardFold, purged_walk_forward_folds
from quant_model_platform.training.models import predict_model, train_lightgbm

LABEL_COL = "label_fwd_20d"

BASE_FEATURES = [
    "growth_score",
    "cycle_score",
    "value_score",
    "quality_score",
    "momentum_20d",
    "momentum_60d",
    "volatility_20d",
    "turnover_20d",
    "log_market_cap",
]

#: ``avg_turnover_21d`` duplicates ``turnover_20d`` (cross-sectional rank
#: correlation 0.9996) and ``return_21d`` duplicates ``momentum_20d`` (0.9675).
#: They stay in ``all`` as a redundancy control, not as candidates.
NOVEL_FACTORS = [
    "bias_turn_21d_252d",
    "beta_60d_000300",
    "gpm_ttm",
    "yoy_roe",
    "alpha101_12",
]
ALL_FACTORS = [
    "alpha101_1",
    "alpha101_101",
    "alpha101_12",
    "alpha101_33",
    "avg_turnover_21d",
    "bias_turn_21d_252d",
    "beta_60d_000300",
    "gpm_ttm",
    "return_21d",
    "yoy_roe",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "baseline": BASE_FEATURES,
    "liquidity_pair": BASE_FEATURES + ["avg_turnover_21d", "bias_turn_21d_252d"],
    "liquidity_bias": BASE_FEATURES + ["bias_turn_21d_252d"],
    "novel": BASE_FEATURES + NOVEL_FACTORS,
    "all": BASE_FEATURES + ALL_FACTORS,
}


def _date(value: Any) -> str:
    text = str(value or "").replace("-", "").strip()
    return text[:8] if len(text) >= 8 and text[:8].isdigit() else ""


def _shift_calendar(date_text: str, days: int) -> str:
    return (datetime.strptime(date_text, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


def _rank_features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per-date cross-sectional percentile rank.

    Raw factor levels differ by orders of magnitude across columns; ranking
    each column within a date removes that scale and makes the LightGBM split
    thresholds comparable. NaNs stay NaN so LightGBM can still route them.
    """
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out.groupby("trade_date", sort=False)[column].rank(pct=True)
    return out


def _attach_raw_returns(
    client: MongoClient,
    panel: pd.DataFrame,
    *,
    data_db_name: str,
    start_date: str,
    end_date: str,
    horizon: int,
) -> pd.DataFrame:
    """Raw next-open forward return, for long-only top-K evaluation."""
    prices = load_price_history(
        client,
        data_db_name=data_db_name,
        symbols=panel["ts_code"].unique().tolist(),
        start_date=_shift_calendar(start_date, -200),
        end_date=end_date,
    )
    if prices.empty:
        out = panel.copy()
        out["raw_fwd"] = np.nan
        return out
    tmp = prices[["trade_date", "ts_code"]].copy()
    tmp["raw_fwd"] = forward_open_to_open_return(prices, horizon=horizon)
    return panel.merge(tmp, on=["trade_date", "ts_code"], how="left")


def _load_factor_snapshot(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    factors = pd.read_parquet(path)
    if "symbol" in factors.columns and "ts_code" not in factors.columns:
        factors = factors.rename(columns={"symbol": "ts_code"})
    factors["trade_date"] = factors["trade_date"].map(_date)
    factors["ts_code"] = factors["ts_code"].astype(str)
    factors = factors[
        (factors["trade_date"] >= start_date) & (factors["trade_date"] <= end_date)
    ]
    return factors.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")


def _attach_factors(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    overlap = [
        column
        for column in factors.columns
        if column in panel.columns and column not in ("trade_date", "ts_code")
    ]
    out = panel.drop(columns=overlap) if overlap else panel
    before = len(out)
    out = out.merge(factors, on=["trade_date", "ts_code"], how="left")
    if len(out) != before:
        raise RuntimeError(f"factor join fanned the panel out: {before} -> {len(out)}")
    return out


def _expanding_folds(
    dates: Sequence[str],
    *,
    min_train_days: int,
    test_days: int,
    purge_days: int,
) -> List[WalkForwardFold]:
    """Expanding-window folds: every test block trains on all prior history.

    The rolling 60-day window in the main experiment is deliberately short so
    the A/B fits the existing demo protocol. A higher-power follow-up can use
    every prior session instead; the first test block then starts after
    ``min_train_days`` of history plus the purge gap.
    """
    ordered = sorted({str(value) for value in dates})
    folds: List[WalkForwardFold] = []
    test_start = int(min_train_days) + int(purge_days)
    fold_id = 0
    while test_start + int(test_days) <= len(ordered):
        train_end = test_start - int(purge_days)
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_dates=ordered[:train_end],
                test_dates=ordered[test_start : test_start + int(test_days)],
                purge_dates=ordered[train_end:test_start],
            )
        )
        fold_id += 1
        test_start += int(test_days)
    return folds


def _topk_returns(test: pd.DataFrame, k: int) -> Tuple[float, float]:
    """Mean raw return of top-K by prediction, and its excess over the pool."""
    rows: List[Tuple[float, float]] = []
    for _, group in test.groupby("trade_date", sort=False):
        valid = group.dropna(subset=["pred", "raw_fwd"])
        if len(valid) < k:
            continue
        top = valid.nlargest(k, "pred")
        rows.append((float(top["raw_fwd"].mean()), float(valid["raw_fwd"].mean())))
    if not rows:
        return float("nan"), float("nan")
    top_mean = float(np.mean([row[0] for row in rows]))
    excess = float(np.mean([row[0] - row[1] for row in rows]))
    return top_mean, excess


def _run_pool(
    index_code: str,
    *,
    factor_parquet: Path,
    start_date: str,
    end_date: str,
    config: PlatformConfig,
    client: MongoClient,
    work_dir: Path,
    feature_sets: Dict[str, List[str]],
    train_days: int,
    test_days: int,
    purge_days: int,
    expanding: bool,
    min_train_days: int,
    top_ks: Sequence[int],
    label_col: str,
    lgb_params: Dict[str, Any],
    min_train_rows: int,
    min_test_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    panel, manifest = build_production_panel(
        index_code=index_code,
        start_date=start_date,
        end_date=end_date,
        parquet_dir=work_dir / "panel" / index_code,
        config=config,
        client=client,
    )
    panel["trade_date"] = panel["trade_date"].map(_date)
    panel["ts_code"] = panel["ts_code"].astype(str)

    factors = _load_factor_snapshot(factor_parquet, start_date, end_date)
    panel = _attach_factors(panel, factors)
    panel = _attach_raw_returns(
        client,
        panel,
        data_db_name=config.data_db_name,
        start_date=start_date,
        end_date=end_date,
        horizon=int(label_col.rsplit("_", 1)[-1].replace("d", ""))
        if label_col.endswith("d")
        else 20,
    )

    all_features = sorted({name for names in feature_sets.values() for name in names})
    present = [name for name in all_features if name in panel.columns and panel[name].notna().any()]
    panel = _rank_features(panel, present)

    dates = sorted(panel["trade_date"].unique())
    if expanding:
        folds = _expanding_folds(
            dates,
            min_train_days=min_train_days,
            test_days=test_days,
            purge_days=purge_days,
        )
    else:
        folds = purged_walk_forward_folds(
            dates, train_days=train_days, test_days=test_days, purge_days=purge_days
        )

    records: List[Dict[str, Any]] = []
    for fold in folds:
        train = panel[panel["trade_date"].isin(fold.train_dates)]
        test = panel[panel["trade_date"].isin(fold.test_dates)]
        train_usable = int(train[label_col].notna().sum())
        test_usable = int(test[label_col].notna().sum())
        if train_usable < min_train_rows or test_usable < min_test_rows:
            continue
        for set_name, names in feature_sets.items():
            usable_names = [name for name in names if name in train.columns and train[name].notna().any()]
            model = train_lightgbm(train, usable_names, label_col, params=lgb_params)
            scored = test.copy()
            scored["pred"] = predict_model(scored, model)
            ic = summarize_ic(spearman_ic(scored, "pred", label_col))
            record: Dict[str, Any] = {
                "index_code": index_code,
                "fold_id": fold.fold_id,
                "set": set_name,
                "test_start": fold.test_dates[0] if fold.test_dates else None,
                "test_end": fold.test_dates[-1] if fold.test_dates else None,
                "n_train_rows": train_usable,
                "n_test_rows": test_usable,
                "n_features": len(model.feature_names),
                "feature_names": model.feature_names,
                "ic_mean": ic.get("mean_ic"),
                "ic_ir": ic.get("ic_ir"),
                "n_ic_days": ic.get("n_days"),
            }
            for k in top_ks:
                top_mean, excess = _topk_returns(scored, int(k))
                record[f"top{k}_mean"] = top_mean
                record[f"top{k}_excess"] = excess
            records.append(record)
        if (fold.fold_id + 1) % 20 == 0:
            print(f"[{index_code}] fold {fold.fold_id + 1}/{len(folds)}", flush=True)

    meta = {
        "index_code": index_code,
        "panel_rows": int(len(panel)),
        "panel_dates": int(panel["trade_date"].nunique()),
        "panel_symbols": int(panel["ts_code"].nunique()),
        "folds": len(folds),
        "expanding": expanding,
        "min_train_days": min_train_days if expanding else None,
        "snapshot_id": getattr(manifest, "snapshot_id", None),
        "feature_fill": {
            name: float(panel[name].notna().mean()) for name in present
        },
    }
    return records, meta


def _paired_summary(records: pd.DataFrame, baseline: str) -> pd.DataFrame:
    base = records[records["set"] == baseline].set_index("fold_id")
    rows: List[Dict[str, Any]] = []
    for set_name, group in records.groupby("set", sort=True):
        if set_name == baseline:
            continue
        treated = group.set_index("fold_id")
        paired = treated.join(base, how="inner", lsuffix="_t", rsuffix="_b")
        if paired.empty:
            continue
        record: Dict[str, Any] = {
            "set_name": set_name,
            "folds": int(len(paired)),
            "ic_treatment": float(paired["ic_mean_t"].mean()),
            "ic_baseline": float(paired["ic_mean_b"].mean()),
        }
        for metric in ("ic_mean",) + tuple(
            column for column in records.columns if column.endswith("_excess")
        ):
            diff = paired[f"{metric}_t"] - paired[f"{metric}_b"]
            diff = diff.dropna()
            if diff.empty:
                continue
            mean_diff = float(diff.mean())
            std_diff = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
            se = std_diff / math.sqrt(len(diff)) if std_diff and not math.isnan(std_diff) else float("nan")
            record[f"delta_{metric}"] = mean_diff
            record[f"t_{metric}"] = mean_diff / se if se and not math.isnan(se) else float("nan")
            record[f"mde80_{metric}"] = 2.802 * se if se and not math.isnan(se) else float("nan")
            record[f"win_{metric}"] = float((diff > 0).mean())
        rows.append(record)
    return pd.DataFrame(rows)


def _verdict(row: Dict[str, Any]) -> str:
    delta_ic = row.get("delta_ic_mean")
    delta_top = row.get("delta_top20_excess", row.get("delta_top50_excess"))
    t_ic = row.get("t_ic_mean")
    if delta_ic is None or delta_top is None:
        return "NO_DATA"
    if math.isnan(delta_ic) or math.isnan(delta_top):
        return "NO_DATA"
    if delta_ic <= 0 or delta_top <= 0:
        return "NOT_CONFIRMED"
    # An effect smaller than the 80%-power threshold cannot be called promising
    # even when its t-stat happens to clear 1.5; that is how a low-power result
    # gets promoted into a false lead.
    mde = row.get("mde80_ic_mean")
    if mde is not None and not math.isnan(mde) and abs(delta_ic) < mde:
        return "INCONCLUSIVE"
    if t_ic is not None and not math.isnan(t_ic) and t_ic >= 1.5:
        return "PROMISING"
    return "NOT_CONFIRMED"


def _render_report(
    per_fold: pd.DataFrame,
    aggregate: pd.DataFrame,
    metas: Dict[str, Any],
    *,
    out_dir: Path,
) -> None:
    top_metric = next(
        (
            column
            for column in aggregate.columns
            if column.startswith("delta_top") and column.endswith("_excess")
        ),
        None,
    )
    top_t_metric = f"t_{top_metric[len('delta_'):]}" if top_metric else None
    top_label = top_metric.replace("delta_top", "Top").replace("_excess", "") if top_metric else "Top"
    lines = [
        "# 因子模型增量实验",
        "",
        f"- 池：{', '.join(metas)}",
        f"- 标签：{LABEL_COL}（next-open，持有 20 日）",
        "- 模型：LightGBM，所有对照组共用同一超参与同一 walk-forward",
        "- 特征：按日截面百分位 rank；NaN 保留给 LightGBM 原生处理",
        "",
        "## 配对结果（相对 baseline）",
        "",
        f"| 池 | 特征组 | 折数 | IC 组 | IC 基线 | ΔIC | t(ΔIC) | MDE80(ΔIC) | Δ{top_label} 超额 | t(Δ{top_label}) | 胜率 | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.index_code} | `{row.set_name}` | {row.folds} | "
            f"{_fmt(row.ic_treatment)} | {_fmt(row.ic_baseline)} | "
            f"{_fmt(row.delta_ic_mean)} | {_fmt(row.t_ic_mean)} | {_fmt(row.mde80_ic_mean)} | "
            f"{_fmt(getattr(row, top_metric, None))} | {_fmt(getattr(row, top_t_metric, None))} | "
            f"{_pct(row.win_ic_mean)} | **{row.verdict}** |"
        )

    lines.extend(["", "## 各池面板", "", "| 池 | 行数 | 日期数 | 股票数 | 折数 |", "|---|---:|---:|---:|---:|"])
    for index_code, meta in metas.items():
        lines.append(
            f"| {index_code} | {meta['panel_rows']} | {meta['panel_dates']} | "
            f"{meta['panel_symbols']} | {meta['folds']} |"
        )

    lines.extend(
        [
            "",
            "## 解读注意",
            "",
            "- baseline 也是 LightGBM，不是规则分；两组只差特征，模型/折/行集完全一致。",
            "- IC 在 demean 后的标签上计算；Top-K 超额用原始 next-open 收益减当日池均值。",
            "- 配对 t 以「折」为单位；测试块不重叠，但 20 日标签仍有序列相关，t 偏乐观。",
            "- MDE80 是 80% 功效下的最小可检测效应；|Δ| < MDE 时写 INCONCLUSIVE，不写「无效」。",
            "- 因子快照来自 `export_factor_snapshot.py`；baseline 已含 `turnover_20d`，"
            "`avg_turnover_21d` 与它相关 0.9996，`return_21d` 与 `momentum_20d` 相关 0.9675。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.0%}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Model-level factor incremental experiment")
    parser.add_argument("--factor-parquet", required=True)
    parser.add_argument("--index-codes", default="hs300,csi500")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--sets", default="baseline,liquidity_pair,liquidity_bias,novel,all")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--purge-days", type=int, default=60)
    parser.add_argument("--expanding", action="store_true", help="train on all prior history")
    parser.add_argument("--min-train-days", type=int, default=250)
    parser.add_argument("--top-k", default="20,50")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument("--num-boost-round", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="/tmp/factor_validation/incremental")
    args = parser.parse_args(argv)

    selected_sets = [name.strip() for name in args.sets.split(",") if name.strip()]
    feature_sets = {name: FEATURE_SETS[name] for name in selected_sets if name in FEATURE_SETS}
    if "baseline" not in feature_sets:
        parser.error("baseline must be one of the sets")
    index_codes = [value.strip() for value in args.index_codes.split(",") if value.strip()]
    top_ks = [int(value) for value in args.top_k.split(",")]

    config = PlatformConfig.from_env()
    client = MongoClient(config.mongo_uri, tz_aware=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    lgb_params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "deterministic": True,
        "num_threads": 4,
        "seed": args.seed,
    }

    all_records: List[Dict[str, Any]] = []
    metas: Dict[str, Any] = {}
    try:
        for index_code in index_codes:
            print(f"[{index_code}] building panel and running folds...", flush=True)
            records, meta = _run_pool(
                index_code,
                factor_parquet=Path(args.factor_parquet),
                start_date=_date(args.start),
                end_date=_date(args.end),
                config=config,
                client=client,
                work_dir=work_dir,
                feature_sets=feature_sets,
                train_days=args.train_days,
                test_days=args.test_days,
                purge_days=args.purge_days,
                expanding=args.expanding,
                min_train_days=args.min_train_days,
                top_ks=top_ks,
                label_col=LABEL_COL,
                lgb_params={**lgb_params, "num_boost_round": args.num_boost_round},
                min_train_rows=args.min_train_rows,
                min_test_rows=args.min_test_rows,
            )
            all_records.extend(records)
            metas[index_code] = meta
            print(f"[{index_code}] {len(records)} fold-set rows", flush=True)
    finally:
        client.close()

    per_fold = pd.DataFrame(all_records)
    if per_fold.empty:
        raise RuntimeError("no folds produced any results")

    aggregates: List[Dict[str, Any]] = []
    for index_code, group in per_fold.groupby("index_code", sort=True):
        paired = _paired_summary(group, "baseline")
        for row in paired.to_dict(orient="records"):
            row["index_code"] = index_code
            row["verdict"] = _verdict(row)
            aggregates.append(row)
    aggregate = pd.DataFrame(aggregates)

    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "meta": {
                    "factor_parquet": args.factor_parquet,
                    "start": args.start,
                    "end": args.end,
                    "sets": selected_sets,
                    "train_days": args.train_days,
                    "test_days": args.test_days,
                    "purge_days": args.purge_days,
                    "expanding": args.expanding,
                    "min_train_days": args.min_train_days,
                    "top_ks": top_ks,
                    "lgb_params": lgb_params,
                    "pools": metas,
                },
                "per_fold": per_fold.to_dict(orient="records"),
                "aggregate": aggregate.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _render_report(per_fold, aggregate, metas, out_dir=out_dir)
    print(aggregate.to_string(index=False))
    print(f"report -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
