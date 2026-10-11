"""Promotion gates vs rule baseline."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from quant_model_platform.evaluation.metrics import spearman_ic, summarize_ic, top_k_turnover_cost
from quant_model_platform.protocol import (
    PROMOTION_FDR_ALPHA,
    PROMOTION_MIN_UNIVERSES,
    PROMOTION_MIN_WALK_FORWARD_FOLDS,
    PROMOTION_PERMUTATION_P,
)


def evaluate_fold(
    frame: pd.DataFrame,
    score_col: str,
    label_col: str = "label_fwd_20d",
) -> Dict[str, Any]:
    ic = spearman_ic(frame, score_col, label_col)
    summary = summarize_ic(ic)
    cost = top_k_turnover_cost(frame, score_col)
    return {**summary, **cost}


def passes_promotion_gates(
    fold_results: List[Dict[str, Any]],
    *,
    baseline_mean_ic: float,
    universe_count: int = 1,
    permutation_p: float = 1.0,
) -> Dict[str, Any]:
    if len(fold_results) < PROMOTION_MIN_WALK_FORWARD_FOLDS:
        return {"passed": False, "reason": "insufficient_walk_forward_folds"}
    mean_ics = [r.get("mean_ic") for r in fold_results if r.get("mean_ic") is not None]
    if not mean_ics:
        return {"passed": False, "reason": "no_ic"}
    model_mean = float(pd.Series(mean_ics).mean())
    if model_mean <= 0 or model_mean < baseline_mean_ic:
        return {"passed": False, "reason": "ic_below_baseline", "model_mean_ic": model_mean, "baseline": baseline_mean_ic}
    if universe_count < PROMOTION_MIN_UNIVERSES:
        return {"passed": False, "reason": "insufficient_universes"}
    if permutation_p > PROMOTION_PERMUTATION_P:
        return {"passed": False, "reason": "permutation_failed", "p_value": permutation_p}
    return {
        "passed": True,
        "model_mean_ic": model_mean,
        "baseline_mean_ic": baseline_mean_ic,
        "fdr_alpha": PROMOTION_FDR_ALPHA,
    }
