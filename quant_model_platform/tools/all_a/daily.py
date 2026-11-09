"""Daily all-A paper runner: signal on due dates, then fill/MTM.

Run after the close (e.g. 16:20 CST). A sleeve's signal is generated on
``next_signal_date``; its orders carry ``execute_date`` = next trading day, so
the same-day execute never fills a signal computed after the close.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.tools.all_a.calendar import trading_days
from quant_model_platform.tools.all_a.config import (
    CAPITAL_PER_SLEEVE,
    INITIAL_SCHEDULE,
    PAPER_EQUITY,
    PAPER_ORDERS,
    PAPER_POSITIONS,
    PAPER_SLEEVES,
)
from quant_model_platform.tools.all_a.execute import execute
from quant_model_platform.tools.all_a.panel import build_panel
from quant_model_platform.tools.all_a.signal import generate_signal


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")




def init_sleeves(capital: float = CAPITAL_PER_SLEEVE) -> None:
    cfg = PlatformConfig.from_env()
    client = MongoClient(cfg.mongo_uri, tz_aware=True)
    db = client["quant_trading"]
    cal = trading_days(client, cfg.data_db_name)
    idx = {d: i for i, d in enumerate(cal)}
    now = datetime.utcnow()
    for sleeve, sched in INITIAL_SCHEDULE.items():
        i = idx.get(sched["first_signal"])
        if i is None:
            print(f"skip sleeve {sleeve}: {sched['first_signal']} not in calendar", flush=True)
            continue
        next_sig = cal[i + 20] if sched["signal_done"] and i + 20 < len(cal) else sched["first_signal"]
        next_exec = cal[i + 1] if i + 1 < len(cal) else None
        db[PAPER_SLEEVES].update_one(
            {"sleeve": sleeve},
            {"$set": {"capital": capital, "cash": capital, "market_value": 0.0, "equity": capital,
                      "next_signal_date": next_sig, "next_execute_date": next_exec, "updated_at": now},
             "$setOnInsert": {"sleeve": sleeve, "created_at": now}},
            upsert=True,
        )
        print(f"{sleeve}: next_signal {next_sig} next_exec {next_exec}", flush=True)
    print("initialized sleeves", list(INITIAL_SCHEDULE), flush=True)


def report() -> None:
    cfg = PlatformConfig.from_env()
    client = MongoClient(cfg.mongo_uri, tz_aware=True)
    db = client["quant_trading"]
    print("== sleeves ==", flush=True)
    for s in db[PAPER_SLEEVES].find({}, {"_id": 0}).sort("sleeve", 1):
        print(f"{s['sleeve']}: cap {s.get('capital',0):.0f} cash {s.get('cash',0):.0f} "
              f"mv {s.get('market_value',0):.0f} eq {s.get('equity',0):.0f} "
              f"next_signal {s.get('next_signal_date')} next_exec {s.get('next_execute_date')}", flush=True)
    print("pending", db[PAPER_ORDERS].count_documents({"status": "pending_fill"}),
          "filled", db[PAPER_ORDERS].count_documents({"status": "filled"}),
          "open positions", db[PAPER_POSITIONS].count_documents({"status": "open"}), flush=True)
    he = db[PAPER_EQUITY].find_one({}, {"_id": 0}, sort=[("date", -1)])
    print("latest equity", he, flush=True)


def run(date: str) -> None:
    cfg = PlatformConfig.from_env()
    client = MongoClient(cfg.mongo_uri, tz_aware=True)
    data_db = cfg.data_db_name
    db = client["quant_trading"]
    cal = trading_days(client, data_db)
    if date not in cal:
        raise SystemExit(f"{date} is not a trading day")

    due = list(db[PAPER_SLEEVES].find({"next_signal_date": date}, {"_id": 0}))
    if due:
        panel = build_panel(date)
        for sleeve_doc in due:
            generate_signal(date, sleeve_doc["sleeve"], panel, write=True)
    execute(date)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily", "init", "report", "signal", "execute"], default="daily")
    ap.add_argument("--date", default=None)
    ap.add_argument("--sleeve", default=None)
    ap.add_argument("--panel", default=None)
    args = ap.parse_args()

    if args.mode == "init":
        init_sleeves()
    elif args.mode == "report":
        report()
    elif args.mode == "execute":
        execute(args.date or _today())
    elif args.mode == "signal":
        if not args.sleeve or not args.panel:
            raise SystemExit("--sleeve and --panel are required for signal mode")
        generate_signal(args.date or _today(), args.sleeve, args.panel, write=True)
    else:
        run(args.date or _today())


if __name__ == "__main__":
    main()
