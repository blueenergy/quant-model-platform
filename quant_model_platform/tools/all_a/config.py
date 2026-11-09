"""Configuration for the all-A production pipeline."""
from __future__ import annotations

from pathlib import Path

START = "20150101"
WARMUP = "20141001"
FEATURES = ["momentum_20d", "momentum_60d", "volatility_20d", "turnover_20d", "log_market_cap"]
LABEL = "label_fwd_20d"
N = 20
CAPITAL_PER_SLEEVE = 250_000.0
LIQ_MIN = 20_000  # 20-day average amount, 千元 -> 20M CNY
HORIZON = 20
PANEL_DIR = Path("/tmp/all_a_prod/panel")
NAMECHANGE_CSV = Path("/tmp/namechange.csv")

BUY_RATE = 0.0001
SELL_RATE = 0.0001
STAMP_RATE = 0.0005
MIN_COMMISSION = 5.0

PAPER_SLEEVES = "paper_sn_all_a_sleeves"
PAPER_ORDERS = "paper_sn_all_a_orders"
PAPER_POSITIONS = "paper_sn_all_a_positions"
PAPER_EQUITY = "paper_sn_all_a_equity"

# Initial stagger. ``signal_done`` sleeves already have their first targets in
# the ledger (A was generated manually on 2026-11-09); their next signal is
# 20 trading days later. B/C/D still need their first signal on the given date.
INITIAL_SCHEDULE = {
    "A": {"first_signal": "20261109", "signal_done": True},
    "B": {"first_signal": "20261116", "signal_done": False},
    "C": {"first_signal": "20261123", "signal_done": False},
    "D": {"first_signal": "20261130", "signal_done": False},
}
