"""IC, FDR, neutralization helpers, and turnover cost."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from quant_model_platform.protocol import DEFAULT_SLIPPAGE_BPS

log = logging.getLogger(__name__)


#: Fewest usable names a date needs before its IC is reported. Dates below it
#: yield NaN rather than a correlation off a handful of names.
MIN_IC_NAMES = 5

#: Fewest usable pairs a date needs before its score is residualized on size.
MIN_SIZE_SAMPLES = 3


def cross_sectional_ic(
    frame: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    date_col: str = "trade_date",
    method: str = "spearman",
    min_names: int = MIN_IC_NAMES,
) -> pd.Series:
    """Per-date cross-sectional correlation between a score and a label.

    The single IC definition in this package. ``method`` goes straight to
    :meth:`pandas.Series.corr`: ``"spearman"`` ranks first, ``"pearson"`` does
    not, so the two differ on how much an extreme name is allowed to matter --
    and in nothing else.

    Neither flavour moves under a per-date shift of the label. Both are
    invariant under adding a constant, and a "pool demean" subtracts one
    constant per date, so a demeaned IC and a raw IC are the same number. Only
    *level* statistics (a pool mean, a top-K excess, a NAV measured against the
    pool) change. ``demo/primary.py`` relies on that and pins it with a test.
    """

    def _ic(group: pd.DataFrame) -> float:
        sub = group[[score_col, label_col]].dropna()
        if len(sub) < min_names:
            return np.nan
        return float(sub[score_col].corr(sub[label_col], method=method))

    return frame.groupby(date_col, sort=False).apply(_ic)


def spearman_ic(frame: pd.DataFrame, score_col: str, label_col: str, date_col: str = "trade_date") -> pd.Series:
    """Rank IC per date; see :func:`cross_sectional_ic`."""
    return cross_sectional_ic(frame, score_col, label_col, date_col=date_col)


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
    # Same trap as the label side: a present-but-empty size column makes this
    # collapse silently to an industry demean. The size branch reads as enabled
    # because the production panel declares ``log_market_cap`` long before
    # anything fills it.
    if size_col in frame.columns:
        usable = (frame[score_col].notna() & frame[size_col].notna()).groupby(
            frame[date_col], sort=False
        ).sum()
        thin = int((usable < MIN_SIZE_SAMPLES).sum())
        if thin:
            log.warning(
                "size residualization skipped on %d of %d dates: fewer than %d usable %s samples",
                thin,
                int(usable.size),
                MIN_SIZE_SAMPLES,
                size_col,
            )

    def _neutral(group: pd.DataFrame) -> pd.Series:
        s = group[score_col].astype(float)
        if industry_col in group.columns:
            s = s - group.groupby(industry_col, sort=False)[score_col].transform("mean")
        if size_col in group.columns:
            x = group[size_col].astype(float)
            mask = s.notna() & x.notna()
            if mask.sum() >= MIN_SIZE_SAMPLES:
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
