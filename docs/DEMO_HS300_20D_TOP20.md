# Demo：`demo-hs300-20d-top20-1y`

与生产 **成长+周期（动量）加权** 排名策略对比：**ML 模型分** 是否在收益、回撤、换手上有改进。  
**不写 `stock_scores`，不下实盘单。**

相关：[[RUNBOOK_SHADOW.md](RUNBOOK_SHADOW.md)]、[[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)]。

---

## 1. 演示契约（冻结）

| 项 | 值 |
|----|-----|
| 代号 | `demo-hs300-20d-top20-1y` |
| 股票池 | `hs300`（`stock_scores.index_codes`） |
| 基线轨 **A** | `composite_growth_cycle_score`，`growth_cycle_weights=60:40`（与 `entry_filter_mvp_spec` / 组合研究默认一致；上线前与组合计划核对） |
| 模型轨 **B** | `model_id` 默认 `promo-hs300-2bc005de`（`source_job_id` 与 Shadow profile 一致，可 CLI 覆盖） |
| 选股 | **TOP 20**，`selection_mode=fixed_top_n`，**等权** |
| 调仓 | **每 20 个交易日**；信号日 **T**（收盘后可知）；成交 **T+1 开盘** |
| 成本 | 单边 **10 bps**（`transaction_cost=0.001` 往返口径与 ScoreBacktester 对齐） |
| 窗口 | **默认最近 252 个交易日**（约 1 年）；CLI `--years 2` → 504 交易日 |
| 结束日 | 基线：`max(score_date)`（hs300）；模型：与 A 调仓日交集 |
| 标签/持有 | 训练标签 20 日；演示调仓 20 日 → **持有期与协议一致** |

### 1.1 模型分来源（B 轨）

| 区间 | 来源 | Mongo 标记 |
|------|------|------------|
| Shadow 日更上线日起 | `model_predictions`（`shadow: true`） | `score_source=shadow_live` |
| 更早调仓日 | **Backfill 模拟**（见 §4） | `score_source=backfill_pit` 或单独 collection |

**禁止**：用「全历史 Parquet 一次 refit」给所有过去 T 打分（未来信息泄漏）。  
Forward **8–12 周 Shadow 观察**仍只认 `shadow_live` 段，不能用 backfill 冒充。

---

## 2. 交付物

| 交付 | 说明 |
|------|------|
| CLI | `python -m quant_model_platform.tools.demo_hs300_20d_top20` |
| 输出目录 | `--out-dir`：`summary.json`、`equity_curve.csv`、`rebalances.csv`、`metadata.json` |
| 文档 | 本文件 + RUNBOOK 一节「Demo 回测」 |
| 可选 API | `GET /api/model-training/demos/hs300-20d-top20`（Phase D3） |
| 可选 UI | 训练域 Tab「相对成长周期 TOP20」（Phase D3） |

### 2.1 `summary.json`（示例字段）

```json
{
  "demo_id": "demo-hs300-20d-top20-1y",
  "window": {"start": "YYYYMMDD", "end": "YYYYMMDD", "trading_days": 252},
  "rebalance_days": 20,
  "top_n": 20,
  "track_a": {"score_column": "composite_growth_cycle_score", "total_return": 0.0, "max_drawdown": 0.0, "ann_vol": 0.0, "avg_turnover": 0.0},
  "track_b": {"model_id": "promo-hs300-2bc005de", "total_return": 0.0, "max_drawdown": 0.0, "shadow_live_from": "YYYYMMDD"},
  "comparison": {"excess_return_b_minus_a": 0.0, "avg_overlap_ratio": 0.0},
  "caveats": ["backfill_pit_before=..."]
}
```

---

## 3. 架构

```text
                    ┌─────────────────────────────────────┐
                    │  demo_hs300_20d_top20.py            │
                    └─────────────────────────────────────┘
           ┌────────────────┬────────────────┬──────────────────┐
           ▼                ▼                ▼                  ▼
   rebalance calendar   Track A scores    Track B scores    ScoreBacktester
   (20d grid)          stock_scores      shadow + backfill  (stock-scoring-system)
                       Mongo finance     Mongo quant_ml
```

- **回测引擎**：优先复用 `stock-scoring-system/analysis/core/score_backtester.py`（`BacktestConfig` + `ScoreBacktester`），避免自写收益公式。
- **编排仓库**：`quant-model-platform`（CLI、读 `quant_ml`、调 backfill）；对 `stock-scoring-system` 为 **可选依赖**（同 monorepo 路径或 `PYTHONPATH`）。

---

## 4. Backfill（B 轨历史段）

对每个调仓日 **T**（仅当无 `shadow_live` 分或 `--force-backfill`）：

1. 从 Mongo 拉 hs300、`score_date <= T` 的 `stock_scores` + 价量，拼 **PIT 面板**（或物化到临时 Parquet，`end_date=T`）。
2. 用 **晋升 job 配方**（`model_kind`、`feature_names`、`model_params`）在 `label` 可算样本上 **refit**（仅用 `trade_date` 行且标签基于 ≤T 的信息；与 `fit_full_sample` 相同但 **面板截断到 T**）。
3. 对 **当日截面** `trade_date=T` 预测 → 写入内存或 `model_predictions` 辅助表（`source=backfill_pit`）。
4. **不调仓日**不算分（仅 ~12 次/年 refit，性能可接受）。

可选优化（Phase D1.5）：缓存「截至 T 的 refit 系数」按周增量；非 MVP 必须。

---

## 5. 开发阶段

### Phase D0 — 前置（约 2–3 天）

| ID | 任务 | 负责人/仓 | 验收 |
|----|------|-----------|------|
| D0.1 | 确认实盘 `score_column`、`top_n`（10 vs 20）、`growth_cycle_weights` | 业务 + 组合计划 | 写入本 doc §1 脚注 |
| D0.2 | Phase 1.5 pipeline 部署（20:00） | infra | `hs300-current.end_date` 跟进 scorer |
| D0.3 | 拉 1 年 hs300 `score_date` 覆盖报告（缺失日列表） | 脚本/手工 | 调仓日 A 轨缺失 <5% |
| D0.4 | 记录 Shadow live 起始日 `shadow_live_from` | 运维 | 写入 `metadata.json` |

### Phase D1 — CLI 核心（约 1.5–2 周）

| ID | 任务 | 验收 |
|----|------|------|
| D1.1 | `demo/rebalance.py`：生成 20 日调仓日序列（窗口内） | 单元测试：计数 ≈ `floor(N/20)` |
| D1.2 | `demo/track_a.py`：按日读 `composite_growth_cycle_score`，TOP20 | 与手工 Mongo 查询一致 1 日 |
| D1.3 | `demo/track_b.py`：优先读 `model_predictions`；缺失走 backfill §4 | 标记 `score_source` |
| D1.4 | `demo/backfill_pit.py`：截断面板 + refit + 预测单日 | 测试：T 早于训练窗末无未来列 |
| D1.5 | 对接 `ScoreBacktester`（同等权、20d、成本） | 与 `entry_filter_mvp` 同窗 A 轨误差 <1e-6 |
| D1.6 | CLI + `summary.json` / CSV 输出 | `pytest tests/test_demo_hs300_20d_top20.py` |
| D1.7 | `--years 1|2`、`--model-id`、`--source-job-id`、`--out-dir` | 文档 `--help` |

**M1 里程碑**：一条命令产出 1 年 A/B 对比表 + 回撤。

### Phase D1.5 — 性能与缓存（可选，约 3 天）

| ID | 任务 |
|----|------|
| D1.5.1 | Backfill 并行按调仓日（进程池 ≤4） |
| D1.5.2 | 可选：backfill 结果写入 `quant_ml.model_predictions_backfill` 避免重复算 |

### Phase D2 — 策略对齐与消融（约 3–5 天）

| ID | 任务 |
|----|------|
| D2.1 | 第三轨 **C**：仅 growth+cycle 60:40 手算列（与 A 应一致，回归测试） |
| D2.2 | 与 `process_portfolio_research_jobs` 同参数跑 1 条研究 job，对比 A 轨净值 |
| D2.3 | `metadata.json` 自动写 caveat（backfill 区间、shadow_live 区间） |

### Phase D3 — API + UI（约 1 周，可延后）

| ID | 任务 | 仓 |
|----|------|-----|
| D3.1 | `GET .../demos/hs300-20d-top20?years=1` 触发或读缓存 | quantFinance |
| D3.2 | 训练页 Tab：双净值、回撤、重叠率表 | dashboard |
| D3.3 | 链接 `promotion-context` + `source_job_id` | dashboard |

### Phase D4 — 组合研究统一入口（可选）

| ID | 任务 |
|----|------|
| D4.1 | 研究 Job 模板：hs300、20d、TOP20、`MODEL_SCORE_ADAPTER` |
| D4.2 | 与 D1 净值交叉校验 |

---

## 6. 测试策略

| 层级 | 内容 |
|------|------|
| 单元 | 调仓日历；单日 TOP20；backfill 截断无泄漏（合成小面板） |
| 集成 | Mem Mongo fixture；1 个调仓日 A/B 分数 |
| 回归 | A 轨与 ScoreBacktester 直接调用同参数 |
| 手工 | aipoc 跑 1 年，`summary.json` 人工 spot-check 3 个调仓日持仓 |

---

## 7. 排期（示意）

| 周 | 目标 |
|----|------|
| W1 | D0 + D1.1–D1.3 |
| W2 | D1.4–D1.7，**M1** 内部分享 |
| W3 | D2；可选 D3 启动 |
| W4+ | D3 上线；D4 按需 |

---

## 8. 风险

| 风险 | 缓解 |
|------|------|
| Backfill 慢（12× refit/年） | 仅调仓日；D1.5 并行 |
| A 轨 `composite_growth_cycle` 与实盘权重不一致 | D0.1 核对 |
| 用户误解 backfill = Shadow 已满 1 年 | UI/报告强制展示 `shadow_live_from` |
| `stock-scoring-system` 导入重 | 子进程 CLI 或薄封装 HTTP 调 researcher |

---

## 9. CLI 草案

```bash
python -m quant_model_platform.tools.demo_hs300_20d_top20 \
  --years 1 \
  --top-n 20 \
  --rebalance-days 20 \
  --score-column composite_growth_cycle_score \
  --growth-cycle-weights 60:40 \
  --model-id promo-hs300-2bc005de \
  --source-job-id 2bc005de-1ca1-4146-bd03-1e19fc31abfd \
  --out-dir /tmp/demo-hs300-1y
```

环境：`MONGO_URI`、`MONGO_DB`、`QUANT_ML_DB`、S3（若 backfill 走 Parquet）。

---

## 10. 未决（实施前确认）

- [ ] 实盘 TOP **10** 还是 **20**（demo 默认 20，可 `--top-n 10`）
- [ ] `growth_cycle_weights` 是否全策略统一 60:40
- [ ] Backfill 是否持久化到 Mongo 或仅 demo 本地 CSV
