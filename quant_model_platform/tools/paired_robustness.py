"""Robustness checks for the paired rolling_500 - rolling_60 differences.

The production backtest reports a mean per-period difference and a t-stat. Both
assume a roughly symmetric distribution; the overlay-on cells have a positive
mean with a sub-50% win rate, which is the signature of a few large periods
carrying the average. This script reports the statistics that do not assume
symmetry: median, sign test, bootstrap CI, and the mean with 2020 removed.

Run with the trainer image and the repo mounted over ``/app``:

    python -m quant_model_platform.tools.paired_robustness \
        --report /tmp/factor_validation/production_backtest_3pools/report.json \
        --out /tmp/factor_validation/paired_robustness
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.2%}"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _verdict(mean: float, lower: float, win_nonzero: float, mean_ex_2020: float) -> str:
    """Judge on non-zero periods.

    Cash dates give both windows the same return, so their difference is
    exactly zero. Counting those ties as "not a win" would turn a regime
    overlay into a low win rate mechanically; the economic win rate is the one
    over periods where the two books actually differed.
    """
    if mean <= 0:
        return "NEGATIVE"
    if (not math.isnan(win_nonzero) and win_nonzero < 0.5) or (
        not math.isnan(mean_ex_2020) and mean_ex_2020 <= 0
    ):
        return "OUTLIER_DRIVEN"
    if lower > 0 and win_nonzero > 0.5:
        return "ROBUST"
    return "WEAK"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Robustness checks for paired window differences")
    parser.add_argument(
        "--report",
        default="/tmp/factor_validation/production_backtest_3pools/report.json",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="/tmp/factor_validation/paired_robustness")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    paired = report.get("paired", [])
    if not paired:
        raise RuntimeError(
            "report has no paired rows with per-period diffs; rerun "
            "production_regime_backtest with the current code"
        )

    rows: List[Dict[str, Any]] = []
    for item in paired:
        diffs = item.get("diffs") or []
        if not diffs:
            continue
        dates = [str(entry["trade_date"]) for entry in diffs]
        values = np.asarray([float(entry["diff"]) for entry in diffs], dtype=float)
        n = len(values)
        mean = float(values.mean())
        median = float(np.median(values))
        win = float((values > 0).mean())
        nonzero = values[values != 0]
        n_nonzero = int(len(nonzero))
        positive = int((nonzero > 0).sum())
        win_nonzero = float((nonzero > 0).mean()) if n_nonzero else float("nan")
        median_nonzero = float(np.median(nonzero)) if n_nonzero else float("nan")
        sign_p = (
            float(binomtest(positive, n_nonzero, 0.5).pvalue)
            if n_nonzero
            else float("nan")
        )
        lower, upper = _bootstrap_ci(values, n_boot=args.bootstrap, seed=args.seed)
        ex_2020 = np.asarray(
            [value for value, date in zip(values, dates) if not date.startswith("2020")],
            dtype=float,
        )
        mean_ex_2020 = float(ex_2020.mean()) if len(ex_2020) else float("nan")
        total = float(values.sum())
        y2020 = float(
            sum(value for value, date in zip(values, dates) if date.startswith("2020"))
        )
        share_2020 = y2020 / total if total else float("nan")
        rows.append(
            {
                "index_code": item.get("index_code"),
                "window": item.get("window"),
                "overlay": bool(item.get("overlay")),
                "periods": n,
                "nonzero_periods": n_nonzero,
                "mean": mean,
                "median": median,
                "median_nonzero": median_nonzero,
                "win_rate": win,
                "win_rate_nonzero": win_nonzero,
                "sign_test_p": sign_p,
                "bootstrap_lower": lower,
                "bootstrap_upper": upper,
                "mean_ex_2020": mean_ex_2020,
                "share_2020": share_2020,
                "verdict": _verdict(mean, lower, win_nonzero, mean_ex_2020),
            }
        )

    summary = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 配对差稳健性",
        "",
        f"- 源报告：`{args.report}`",
        f"- Bootstrap：{args.bootstrap} 次，95% 百分位区间",
        "- 判定：均值>0；胜率<50% 或剔除 2020 后均值≤0 → OUTLIER_DRIVEN；否则 bootstrap 下界>0 且胜率>50% → ROBUST。",
        "",
        "| 池 | overlay | 期数 | 非零期 | 均值 | 中位(全) | 中位(非零) | 胜率(全) | 胜率(非零) | 符号p(非零) | Bootstrap 95% | 剔除2020 | 2020占比 | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in summary.sort_values(["index_code", "overlay"]).itertuples(index=False):
        lines.append(
            f"| {row.index_code} | {'on' if row.overlay else 'off'} | {row.periods} | "
            f"{row.nonzero_periods} | {_pct(row.mean)} | {_pct(row.median)} | "
            f"{_pct(row.median_nonzero)} | {_pct(row.win_rate)} | {_pct(row.win_rate_nonzero)} | "
            f"{_fmt(row.sign_test_p, 3)} | [{_pct(row.bootstrap_lower)}, {_pct(row.bootstrap_upper)}] | "
            f"{_pct(row.mean_ex_2020)} | {_pct(row.share_2020)} | **{row.verdict}** |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- 现金期两个窗口收益完全相同，diff=0；胜率/符号检验只用非零期，均值仍包含这些 0。",
            "- 符号检验 p 小且非零胜率<50% 时，方向与正均值相反：均值由少数大期拉动。",
            "- 剔除 2020 后的均值若≤0，说明 2020 的 COVID 行情是主要支柱。",
            "- Bootstrap 区间只看均值；中位数和非零胜率才说明典型期。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps({"summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"report -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
