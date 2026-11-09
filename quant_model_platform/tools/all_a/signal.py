"""Train the size-neutral all-A model and write top20 paper targets."""
from __future__ import annotations

import bisect
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pymongo import MongoClient

from quant_model_platform.config import PlatformConfig
from quant_model_platform.tools.all_a.calendar import trading_days
from quant_model_platform.training.models import predict_model, train_lightgbm
from quant_model_platform.tools.all_a.config import (
    CAPITAL_PER_SLEEVE,
    FEATURES,
    LABEL,
    N,
    PAPER_ORDERS,
    PAPER_SLEEVES,
)

PARAMS = {
    "objective": "regression", "metric": "l2", "verbosity": -1,
    "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "deterministic": True, "num_threads": 4, "seed": 42, "num_boost_round": 50,
}


def _industry_stints(client, data_db: str):
    stints: dict[str, list[tuple[str, str, str]]] = {}
    for r in client[data_db]["sw_index_member"].find(
        {"src": "SW2021"},
        {"_id": 0, "ts_code": 1, "con_code": 1, "in_date": 1, "out_date": 1, "l1_name": 1},
    ):
        code = str(r.get("ts_code") or r.get("con_code"))
        in_d = str(r.get("in_date") or "")
        out_d = r.get("out_date")
        out_d = "" if out_d in (None, "None", "") else str(out_d)
        stints.setdefault(code, []).append((in_d, out_d, str(r.get("l1_name") or "UNKNOWN")))
    for code in stints:
        stints[code].sort()
    return stints


def _industry_at(stints, sym: str, date: str) -> str:
    v = stints.get(sym)
    if not v:
        return "UNKNOWN"
    pos = bisect.bisect_right([s[0] for s in v], date) - 1
    if pos < 0:
        return "UNKNOWN"
    in_d, out_d, l1 = v[pos]
    if out_d and date > out_d:
        return "UNKNOWN"
    return l1


def _select_neutral(preds: pd.DataFrame, n: int, stints, date: str) -> list[str]:
    groups = pd.Series([_industry_at(stints, s, date) for s in preds["ts_code"]], index=preds["ts_code"])
    counts = groups.value_counts()
    total = float(counts.sum())
    quotas = {k: int(n * c / total) for k, c in counts.items()}
    rem = n - sum(quotas.values())
    for k, _ in sorted(((k, n * c / total - int(n * c / total)) for k, c in counts.items()), key=lambda x: x[1], reverse=True)[:rem]:
        quotas[k] += 1
    chosen, used = [], set()
    for ind, q in quotas.items():
        if q <= 0:
            continue
        for sym in preds["ts_code"]:
            if sym in used or groups.get(sym) != ind:
                continue
            chosen.append(sym); used.add(sym)
            if len(chosen) >= sum(quotas[kk] for kk in list(quotas)[: list(quotas).index(ind) + 1]):
                break
    if len(chosen) < n:
        for sym in preds["ts_code"]:
            if sym in used:
                continue
            chosen.append(sym); used.add(sym)
            if len(chosen) >= n:
                break
    return chosen[:n]




def generate_signal(signal_date: str, sleeve: str, panel_path: Path,
                    capital: float = CAPITAL_PER_SLEEVE, write: bool = True) -> list[dict]:
    cfg = PlatformConfig.from_env()
    client = MongoClient(cfg.mongo_uri, tz_aware=True)
    data_db = cfg.data_db_name
    db = client["quant_trading"]

    panel = pd.read_parquet(panel_path, columns=["trade_date", "ts_code", LABEL] + FEATURES)
    panel["trade_date"] = panel["trade_date"].astype(str).str.replace("-", "").str[:8]
    panel["ts_code"] = panel["ts_code"].astype(str)
    for c in FEATURES:
        panel[c] = panel.groupby("trade_date", sort=False)[c].rank(pct=True)
    panel[LABEL] = panel.groupby("trade_date", sort=False)[LABEL].rank(pct=True) - 0.5
    train = panel.dropna(subset=[LABEL])
    model = train_lightgbm(train, FEATURES, LABEL, params=PARAMS)
    signal = panel[panel["trade_date"] == signal_date].copy()
    signal["pred"] = predict_model(signal, model)
    signal = signal.dropna(subset=["pred"]).sort_values("pred", ascending=False)

    stints = _industry_stints(client, data_db)
    chosen = _select_neutral(signal, N, stints, signal_date)
    ref = pd.DataFrame(list(client[data_db]["volume_price"].find(
        {"trade_date": signal_date, "symbol": {"$in": chosen}},
        {"_id": 0, "symbol": 1, "close": 1},
    )))
    ref = ref.rename(columns={"symbol": "ts_code"})
    ref["ref_close"] = pd.to_numeric(ref["close"], errors="coerce")
    per_name = capital / N
    ref["target_amount"] = per_name
    ref["target_shares"] = (ref["target_amount"] / ref["ref_close"] // 100 * 100).astype(int)

    cal = trading_days(client, data_db)
    idx = {d: i for i, d in enumerate(cal)}
    i = idx.get(signal_date)
    next_sig = cal[i + 20] if i is not None and i + 20 < len(cal) else None
    next_exec = cal[i + 1] if i is not None and i + 1 < len(cal) else None

    if write and not ref.empty:
        now = datetime.utcnow()
        db[PAPER_ORDERS].delete_many({"sleeve": sleeve, "signal_date": signal_date})
        db[PAPER_ORDERS].insert_many([
            {"sleeve": sleeve, "signal_date": signal_date, "execute_date": next_exec, "symbol": r.ts_code,
             "target_shares": int(r.target_shares), "target_amount": float(r.target_amount),
             "ref_close": float(r.ref_close), "status": "pending_fill", "created_at": now}
            for r in ref.itertuples(index=False)
        ], ordered=False)
        db[PAPER_SLEEVES].update_one(
            {"sleeve": sleeve},
            {"$set": {"next_signal_date": next_sig, "next_execute_date": next_exec, "updated_at": now},
             "$setOnInsert": {"sleeve": sleeve, "capital": capital, "cash": capital, "market_value": 0.0, "equity": capital, "created_at": now}},
            upsert=True,
        )
    print(f"signal {sleeve} {signal_date}: {len(ref)} names, model train rows {len(train)}", flush=True)
    return ref[["ts_code", "ref_close", "target_shares"]].to_dict(orient="records")
