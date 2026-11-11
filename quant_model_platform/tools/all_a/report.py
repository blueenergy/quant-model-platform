"""Verbose paper report: sleeves, open positions, equity curve, order stats."""
from __future__ import annotations

import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.tools.all_a.config import (
    PAPER_EQUITY,
    PAPER_ORDERS,
    PAPER_POSITIONS,
    PAPER_SLEEVES,
)


def report() -> None:
    cfg = PlatformConfig.from_env()
    client = MongoClient(cfg.mongo_uri, tz_aware=True)
    data_db = cfg.data_db_name
    db = client["quant_trading"]

    latest = client[data_db]["index_prices"].find_one(
        {"ts_code": "000852.SH"}, {"_id": 0, "trade_date": 1}, sort=[("trade_date", -1)]
    )
    latest_date = str(latest["trade_date"]) if latest else None
    closes = {}
    if latest_date:
        closes = {r["symbol"]: r.get("close") for r in client[data_db]["volume_price"].find(
            {"trade_date": latest_date}, {"_id": 0, "symbol": 1, "close": 1}
        )}

    print(f"latest price date: {latest_date}", flush=True)
    for s in db[PAPER_SLEEVES].find({}, {"_id": 0}).sort("sleeve", 1):
        sleeve = s["sleeve"]
        capital = float(s.get("capital", 0.0))
        equity = float(s.get("equity", 0.0))
        print(f"\n[{sleeve}] capital {capital:.0f} cash {s.get('cash',0):.0f} "
              f"mv {s.get('market_value',0):.0f} equity {equity:.0f} "
              f"ret {equity/capital-1 if capital else 0:.2%} "
              f"next_signal {s.get('next_signal_date')} next_exec {s.get('next_execute_date')}", flush=True)
        positions = list(db[PAPER_POSITIONS].find({"sleeve": sleeve, "status": "open"}, {"_id": 0}))
        if positions:
            print(f"  open positions: {len(positions)}", flush=True)
            rows = []
            for p in positions:
                close = closes.get(p["symbol"])
                pnl = (float(close) / p["entry_price"] - 1) if close else float("nan")
                rows.append((p["symbol"], p["shares"], p["entry_price"], close, p["exit_due_date"], pnl))
            rows.sort(key=lambda x: (-(x[5] if x[5] == x[5] else -99)))
            for r in rows:
                print(f"  {r[0]} {r[1]:>6d} entry {r[2]:>9.2f} last {str(r[3]):>9s} exit {r[4]} pnl {r[5]*100:6.2f}%", flush=True)
        eq = list(db[PAPER_EQUITY].find({"sleeve": sleeve}, {"_id": 0}).sort("date", 1))
        if eq:
            series = pd.Series([float(x["equity"]) for x in eq], index=[x["date"] for x in eq])
            mdd = float((series / series.cummax() - 1).min())
            print(f"  equity curve: {len(eq)} days {eq[0]['date']}..{eq[-1]['date']} "
                  f"last {series.iloc[-1]:.0f} ret {series.iloc[-1]/capital-1 if capital else 0:.2%} maxDD {mdd:.2%}", flush=True)

    print("\norders:", {k: db[PAPER_ORDERS].count_documents({"status": k}) for k in ("pending_fill", "filled", "skipped")}, flush=True)


if __name__ == "__main__":
    report()
