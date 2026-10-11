"""Locked research protocol constants (see docs/RESEARCH_PROTOCOL.md)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

LABEL_HORIZON_DAYS = 20
MAX_PURGE_DAYS = 60
DEFAULT_EXECUTION_PRICE = "next_open"
DEFAULT_SLIPPAGE_BPS = 10.0
DEFAULT_TURNOVER_ASSUMPTION = 1.0

UniverseCode = Literal["hs300", "a500", "csi1000", "csi2000"]
LabelNeutralize = Literal["none", "demean", "size_neutral"]

PROMOTION_MIN_WALK_FORWARD_FOLDS = 3
PROMOTION_MIN_UNIVERSES = 2
PROMOTION_FDR_ALPHA = 0.05
PROMOTION_PERMUTATION_P = 0.05


@dataclass(frozen=True)
class ResearchProtocol:
    label_horizon_days: int = LABEL_HORIZON_DAYS
    max_purge_days: int = MAX_PURGE_DAYS
    execution_price: str = DEFAULT_EXECUTION_PRICE
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    label_neutralize: LabelNeutralize = "demean"
    universes_for_promotion: Tuple[UniverseCode, ...] = ("csi1000", "hs300")


DEFAULT_PROTOCOL = ResearchProtocol()
