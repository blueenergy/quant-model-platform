"""Paper execution: fill pending targets at the open, sell due positions, MTM."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.tools.all_a.calendar import trading_days
from quant_model_platform.tools.all_a.config import (
    BUY_RATE,
    HORIZON,
    MIN_COMMISSION,
    N,
    PAPER_EQUITY,
    PAPER_ORDERS,
    PAPER_POSITIONS,
    PAPER_SLEEVES,
    SELL_RATE,
    STAMP_RATE,
)


def _buy_fee(amount: float) -> float:
    return max(MIN_COMMISSION, amount * BUY_RATE)


def _sell_fee(amount: float) -> float:
    return max(MIN_COMMISSION, amount * SELL_RATE) + amount * STAMP_RATE


def _limit_pct(sym: str) -> float:
    return 0.20 if sym.startswith(("688", "300")) else 0.10


def execute(date: str) -> None:
    cfg = PlatformConfig.from_env()
    client = MongoClient(cfg.mongo_uri, tz_aware=True)
    data_db = cfg.data_db_name
    db = client["quant_trading"]

    cal = trading_days(client, data_db)
    pos = {d: i for i, d in enumerate(cal)}
    if date not in pos:
        raise SystemExit(f"{date} is not a trading day")
    exit_date = cal[pos[date] + HORIZON] if pos[date] + HORIZON < len(cal) else None

    raw = pd.DataFrame(list(client[data_db]["volume_price"].find(
        {"trade_date": date},
        {"_id": 0, "symbol": 1, "open": 1, "high": 1, "low": 1, "close": 1, "pct_chg": 1},
    )))
    if len(raw) < 1000:
        raise SystemExit(f"daily price data not ready for {date} (rows={len(raw)}); retry after sync")
    raw = raw.rename(columns={"symbol": "ts_code"})
    for c in ("open", "high", "low", "close", "pct_chg"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    px = {r.ts_code: r for r in raw.itertuples(index=False)}

    summary = []
    for sleeve_doc in db[PAPER_SLEEVES].find({}, {"_id": 0}):
        sleeve = sleeve_doc["sleeve"]
        cash = float(sleeve_doc.get("cash", 0.0))
        capital = float(sleeve_doc.get("capital", 0.0))

        sold = 0
        for p in db[PAPER_POSITIONS].find({"sleeve": sleeve, "status": "open", "exit_due_date": date}):
            r = px.get(p["symbol"])
            if r is None or pd.isna(r.open):
                continue
            gross = p["shares"] * float(r.open)
            proceeds = gross - _sell_fee(gross)
            cash += proceeds
            db[PAPER_POSITIONS].update_one(
                {"_id": p["_id"]},
                {"$set": {"status": "closed", "exit_price": float(r.open), "exit_date": date,
                          "pnl": proceeds - p["shares"] * p["entry_price"], "closed_at": datetime.utcnow()}},
            )
            sold += 1

        pending = list(db[PAPER_ORDERS].find({"sleeve": sleeve, "status": "pending_fill"}))
        budget = cash
        per_name = budget / N
        filled = skipped = 0
        for o in pending:
            # Only orders whose *execute* date is today; a signal generated after
            # today's close must never fill at today's open.
            exec_date = o.get("execute_date")
            if not exec_date:
                si = pos.get(str(o.get("signal_date") or ""))
                exec_date = cal[si + 1] if si is not None and si + 1 < len(cal) else None
            # Fill on the intended date, or late if the runner missed that day
            # (e.g. the daily sync was not ready); never fill before it.
            if not exec_date or exec_date > date:
                continue
            sym = o["symbol"]
            r = px.get(sym)
            if r is None or pd.isna(r.open):
                db[PAPER_ORDERS].update_one({"_id": o["_id"]}, {"$set": {"status": "skipped", "skip_reason": "no_price", "updated_at": datetime.utcnow()}})
                skipped += 1
                continue
            one_price = (not pd.isna(r.high) and not pd.isna(r.low) and not pd.isna(r.pct_chg)
                         and r.high == r.low and r.pct_chg >= _limit_pct(sym) * 100 * 0.98)
            if one_price:
                db[PAPER_ORDERS].update_one({"_id": o["_id"]}, {"$set": {"status": "skipped", "skip_reason": "one_price_limit_up", "updated_at": datetime.utcnow()}})
                skipped += 1
                continue
            shares = int(min(per_name, budget) // float(r.open) // 100 * 100)
            if shares < 100:
                db[PAPER_ORDERS].update_one({"_id": o["_id"]}, {"$set": {"status": "skipped", "skip_reason": "lot_too_small", "updated_at": datetime.utcnow()}})
                skipped += 1
                continue
            amount = shares * float(r.open)
            cash -= amount + _buy_fee(amount)
            db[PAPER_POSITIONS].insert_one({
                "sleeve": sleeve, "symbol": sym, "shares": shares, "entry_price": float(r.open),
                "entry_date": date, "exit_due_date": exit_date, "status": "open", "created_at": datetime.utcnow(),
            })
            db[PAPER_ORDERS].update_one({"_id": o["_id"]}, {"$set": {"status": "filled", "fill_price": float(r.open), "fill_shares": shares, "updated_at": datetime.utcnow()}})
            filled += 1

        market_value = 0.0
        for p in db[PAPER_POSITIONS].find({"sleeve": sleeve, "status": "open"}):
            r = px.get(p["symbol"])
            close = float(r.close) if r is not None and not pd.isna(r.close) else p["entry_price"]
            market_value += p["shares"] * close
        equity = cash + market_value
        db[PAPER_SLEEVES].update_one(
            {"sleeve": sleeve},
            {"$set": {"cash": cash, "market_value": market_value, "equity": equity, "updated_at": datetime.utcnow()}},
        )
        db[PAPER_EQUITY].delete_many({"date": date, "sleeve": sleeve})
        db[PAPER_EQUITY].insert_one({
            "date": date, "sleeve": sleeve, "cash": cash, "market_value": market_value,
            "equity": equity, "ret": equity / capital - 1 if capital else 0.0, "created_at": datetime.utcnow(),
        })
        summary.append((sleeve, filled, skipped, sold, round(equity, 2)))

    print(f"{'sleeve':>6s} {'filled':>6s} {'skipped':>7s} {'sold':>4s} {'equity':>12s}")
    for row in summary:
        print(f"{row[0]:>6s} {row[1]:6d} {row[2]:7d} {row[3]:4d} {row[4]:12.2f}")
