"""IC, FDR, neutralization helpers, and turnover cost."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from quant_model_platform.protocol import DEFAULT_SLIPPAGE_BPS


def spearman_ic(frame: pd.DataFrame, score_col: str, label_col: str, date_col: str = "trade_date") -> pd.Series:
    def _ic(group: pd.DataFrame) -> float:
        sub = group[[score_col, label_col]].dropna()
        if len(sub) < 5:
            return np.nan
        return float(sub[score_col].corr(sub[label_col], method="spearman"))

    return frame.groupby(date_col, sort=False).apply(_ic)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool), p
    max_idx = np.where(passed)[0].max()
    cutoff = ranked[max_idx]
    return p <= cutoff, p


def permutation_test_ic(
    frame: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> float:
    """One-sided test: observed mean IC > permuted mean IC."""
    obs = spearman_ic(frame, score_col, label_col).mean()
    if np.isnan(obs):
        return 1.0
    rng = np.random.default_rng(seed)
    perm_means = []
    for _ in range(n_perm):
        shuffled = frame.copy()
        shuffled[score_col] = rng.permutation(shuffled[score_col].values)
        perm_means.append(spearman_ic(shuffled, score_col, label_col).mean())
    return float((np.sum(np.array(perm_means) >= obs) + 1) / (n_perm + 1))


def industry_size_neutralize_scores(
    frame: pd.DataFrame,
    score_col: str,
    *,
    date_col: str = "trade_date",
    industry_col: str = "industry",
    size_col: str = "log_market_cap",
) -> pd.Series:
    """Demean within industry; optional size residualization per date."""

    def _neutral(group: pd.DataFrame) -> pd.Series:
        s = group[score_col].astype(float)
        if industry_col in group.columns:
            s = s - group.groupby(industry_col, sort=False)[score_col].transform("mean")
        if size_col in group.columns:
            x = group[size_col].astype(float)
            mask = s.notna() & x.notna()
            if mask.sum() >= 3:
                coef = np.polyfit(x[mask], s[mask], 1)
                s = s - (coef[0] * x + coef[1])
        return s - s.mean()

    return frame.groupby(date_col, sort=False, group_keys=False).apply(_neutral)


def top_k_turnover_cost(
    frame: pd.DataFrame,
    score_col: str,
    *,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    top_k: int = 50,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> Dict[str, float]:
    """Rough turnover from daily Top-K membership changes * slippage."""
    dates = sorted(frame[date_col].unique())
    prev: Optional[set] = None
    turnovers: List[float] = []
    for d in dates:
        day = frame.loc[frame[date_col] == d].nlargest(top_k, score_col)
        current = set(day[code_col].tolist())
        if prev is not None and prev:
            changed = len(current - prev) + len(prev - current)
            turnovers.append(changed / (2 * top_k))
        prev = current
    avg_turn = float(np.mean(turnovers)) if turnovers else 0.0
    cost = avg_turn * (slippage_bps / 10000.0) * 2
    return {"avg_turnover": avg_turn, "slippage_cost": cost}


def summarize_ic(ic_series: pd.Series) -> Dict[str, float]:
    clean = ic_series.dropna()
    if clean.empty:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "n_days": 0}
    mean_ic = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    ir = mean_ic / std if std and not np.isnan(std) else float("nan")
    return {"mean_ic": mean_ic, "ic_ir": ir, "n_days": int(len(clean))}
