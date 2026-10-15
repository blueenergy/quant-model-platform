# Demo：hs300 模型分 vs 成长周期规则分

**长期 ML 计划的评估基准**：在 hs300 上，模型分是否**稳定优于**生产的成长+周期加权规则分。  
主证据是扣成本后的**中性化 IC / top-K 逐日超额**；净值与回撤只作沟通，**不作为结论依据**。  
**只读生产数据；不写 `stock_scores`，不下实盘单。**

相关：[[RUNBOOK_SHADOW.md](RUNBOOK_SHADOW.md)]、[[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)]。

---

## 1. 测量契约（冻结 2026-10-15，B0）

> 本节参数已于 2026-10-15 逐条对**生产实况**核实，出处见 §1.3；与 draft 版不符处已就地更正。
> 契约只定义「测量什么、按什么口径算」，实现进度见 §5。

| 项 | 值 | 依据 |
|----|-----|------|
| 股票池 | `hs300`，取 `finance.stock_scores` 的**当日** `index_codes` 成员（天然 as-of，不用固定名单） | 实盘 plan `live_trading_default` |
| 基线轨 **A** | **自算** `0.6 × growth_score + 0.4 × cycle_score`（可用权重归一；缺一维则取另一维） | 实盘 `score_type=growth_cycle_weighted`，`growth_weight=0.6` / `cycle_weight=0.4` |
| 模型轨 **B** | `model_id` 由 Shadow 登记提供，CLI 可覆盖；backfill 段用**晋升 job 的配方**（从 `model_training_jobs` 读，不手抄） | — |
| 选股 | TOP **N**，`selection_mode=fixed_top_n`，**等权** | — |
| N（主/敏感） | 主 **N=10**（与实盘一致）；**N=20** 作敏感性 | 实盘 `top_n=10` |
| 调仓（主/敏感） | 主 **20 交易日**（与标签 horizon 一致）；**30 交易日**（与实盘一致）作敏感性 | 标签 `LABEL_HORIZON_DAYS=20`；实盘 `rebalance_days=30` |
| 成交 | 信号日 **T**（收盘后可知）→ **T+1 开盘** | 协议 `execution_price=next_open` |
| 成本 | **费用 + 滑点**两项，口径见 §1.2（**不再**写作「单边 10 bps」） | 协议 + `alpha_factor_backtest.CostAssumptions` |
| 标签 | `label_fwd_20d` = 池内 **demean** 后的超额，不是原始收益 | `dataset/labels.py::attach_label_column` 默认 `neutralize="demean"` |
| 训练/评估切分 | **purge ≥ 60 交易日**，复用 `evaluation/walk_forward.py` | 协议 `MAX_PURGE_DAYS=60` |
| 主指标 | 扣成本后**中性化 IC / rank-IC**、TOP-K 相对池等权的**逐日**超额 | 协议「成本」「模型晋级」两节 |
| 次指标 | 净值、回撤、换手 —— **仅作沟通**，须标注独立期数（n≈12/年）无统计功效 | — |
| 对照轨 | **R**：随机分数，相对池等权超额应 ≈ 0（否则 harness 有泄漏或记账错） | 协议置换检验 |
| 窗口 | 默认 **504 交易日（2 年）**，另支持 `start/end` 显式区间 | — |
| 结束日 | 基线：`max(score_date)`（hs300）；模型：与 A 调仓日交集 | — |

**指标优先级**：主证据是 IC / top-K 逐日超额（约 250 日 × 300 名 ≈ 7.5 万观测，有功效）；净值差只能作为**同名调仓日配对**的统计量，附置换/bootstrap p 值。1 年窗口的净值差**不足以**支持任何「模型更好」的结论。

**调仓周期与标签的错配**：标签 horizon 是 20 日，实盘调仓是 30 日。主口径取 20 日（与标签一致），30 日作为「实盘会拿到什么」的敏感性 —— 若 20 日口径都赢不了，30 日不必看。

### 1.1 模型分来源（B 轨）

**B 轨全部本机自产**。K8s/VOLC 是**独立环境、不可访问**，其 `quant_ml`（含既有 `model_predictions` 与 Shadow 观察）本机够不着 ⇒ demo **不引用任何远端产物**。

| 区间 | 来源 | Mongo 标记 |
|------|------|------------|
| 本机 Shadow 启用之后 | 本机 `quant_ml.model_predictions` | `score_source=shadow_live` |
| 更早调仓日 | 本机 **PIT backfill**（见 §4），按协议 purge=60 | `score_source=backfill_pit` |

⚠️ 本机 `quant_ml` 现为**空库**（`model_training_jobs` / `results` / `predictions` / `manifests` 全 0，从未在本机训练过）⇒ **没有现成 `model_id` 可引用**，B 轨第一步必须是在本机跑通一次训练（§5 B3a）。

**边界按日期切，不按「当天有没有分」切** —— 半覆盖的截面不得当完整截面用。

**禁止**：用「全历史 Parquet 一次 refit」给所有过去 T 打分（未来信息泄漏）。  
Forward **8–12 周 Shadow 观察**若要做，须在**本机**新起（本机 `MODEL_SHADOW_PROFILES` 目前未设），且只认 `shadow_live` 段，不能用 backfill 冒充。  
**可测替代**：在 Shadow 与 backfill 重叠的调仓日做 rank correlation —— 复现得好，长曲线才可信（§5 B3b）。

### 1.2 成本口径（2026-10-15 核实）

仓库现存 **4 套**口径，draft 的「单边 10 bps」不属于其中任何一套：

| 口径 | 出处 | 一次 100% 换手的成本 |
|------|------|----------------------|
| 协议（只定义滑点） | `protocol.DEFAULT_SLIPPAGE_BPS=10`（单边） | 滑点往返 20 bps，**未含费用** |
| 因子回测（唯一完整实现） | `alpha_factor_backtest.CostAssumptions.round_trip_rate()` | 费用 7 bps + 滑点 20 bps = **27 bps** |
| 组合回测 / 实盘 plan（legacy） | `trade_fee_model.period_cost_rate` + `transaction_cost=0.001` | **10 bps**，且**不含滑点** |
| 组合回测（显式费率预设） | 买 1bp + 卖 1bp + 印花税 5bp | **7 bps**（此时 `transaction_cost` 被 `from_params` **忽略**） |
| draft 的 `demo/portfolio.py` | `ONE_WAY_COST=0.001` × 2 侧 | 20 bps |

**采用**：`费用（往返 7 bps）+ 单边滑点 10 bps` = 一次全换手 **27 bps**，即 `alpha_factor_backtest` 的口径 —— 它是唯一同时落地了协议滑点常量与真实费率的地方，且对高换手轨偏保守（模型轨换手通常更高，成本从宽更不容易自欺）。

`ScoreBacktester` **没有任何成本字段**（纯 gross 输出），draft 里「与 ScoreBacktester 对齐」的说法作废。

### 1.3 口径来源（核实记录）

- **实盘参数**：`quant_trading.portfolio_trade_plans` 最新 plan（`preset_id=live_trading_default`，`params_hash=7cd2a2143a5f`）—— hs300 / 60:40 / `top_n=10` / `rebalance_days=30` / `execution_price=next_open` / `transaction_cost=0.001`。
- **预置对照**：`portfolio_strategy_param_presets` 共 6 条，研究面多为 10 日调仓，`growth_cycle_index_enhanced_csi1000_top10_20d` 为 20 日。
- **`finance.stock_scores` 没有 `composite_growth_cycle_score` 列**（实际字段：`composite_score` / `growth_score` / `cycle_score` / `technical_score` / `value_score` / `fundamental_score` / `money_flow_score` …）⇒ A 轨必须自算加权；doc 原稿「优先读存储列」在生产恒为死分支。
- **成本语义**：`analysis/core/trade_fee_model.py::from_params` / `period_cost_rate`；`portfolio_backtester` 与 `score_backtester` **均不含滑点**。
- **数据可得性**：本机 `finance.stock_scores` 656 万条、hs300 85.2 万条（2015-01-30 → 2026-10-14）✅；`quant_data.volume_price` 1447 万条（2010-01-04 → 2026-10-15）✅。本机 `quant_ml` **从未训练过**（全 0，`MODEL_SHADOW_PROFILES` 未设）—— B 轨须本机自产，详见 §3.1。

---

## 2. 交付物

| 交付 | 说明 |
|------|------|
| CLI | `python -m quant_model_platform.tools.demo_hs300_20d_top20` |
| 输出目录 | `--out-dir` **现有**：`summary.json`、`equity_curve.csv`、`rebalances.csv`；**计划**：`metrics_by_date.csv`（逐日主指标，B2）、`metadata.json`（provenance，B4） |
| 文档 | 本文件 + RUNBOOK 一节「Demo 回测」 |
| 后续（可选） | `GET /api/model-training/demos/...` + 训练域 Tab（API/UI，非 MVP） |

### 2.1 `summary.json`（目标字段）

```json
{
  "demo_id": "demo-hs300-gc-vs-model",
  "contract": {"frozen": "2026-10-15", "purge_days": 60, "label_neutralize": "demean",
               "cost_round_trip_bps": 27, "execution_price": "next_open"},
  "window": {"start": "YYYYMMDD", "end": "YYYYMMDD", "trading_days": 504},
  "params": {"universe": "hs300", "top_n": 10, "rebalance_days": 20, "growth_cycle": "60:40"},
  "primary": {"rank_ic_mean": 0.0, "rank_ic_ir": 0.0, "topk_excess_mean_bps": 0.0,
              "paired_p_value": null, "periods": 0},
  "track_a": {"label": "规则分 60:40"},
  "track_b": {"model_id": "...", "shadow_live_from": "YYYYMMDD", "scored_days": {"shadow_live": 0, "backfill_pit": 0}},
  "control_r": {"topk_excess_mean_bps": 0.0},
  "nav": {"total_return_a": 0.0, "total_return_b": 0.0, "max_drawdown_grid": 0.0,
          "note": "调仓网格口径，非日频；n 小、无统计功效"},
  "backfill_validation": {"overlap_days": 0, "rank_corr_vs_shadow": null},
  "provenance": {"code_sha": "", "panel_snapshot_id": "", "source_job_id": ""},
  "caveats": []
}
```

净值字段保留，但报告与 UI 必须**先展示主指标**，并在净值旁标注口径与功效限制。

---

## 3. 架构与运行位置

```text
                    ┌─────────────────────────────────────┐
                    │  demo_hs300_20d_top20.py            │
                    └─────────────────────────────────────┘
        ┌───────────────┬────────────────┬──────────────────┐
        ▼               ▼                ▼                  ▼
  rebalance grid   Track A 自算      Track B 读数      metrics + NAV
  (20d / 30d)      finance.stock_    本机 quant_ml     IC / top-K 超额
                   scores(本机✅)    + PIT backfill    + 成本 + 配对统计
```

- **运行位置**：**全部本机（WSL）**。K8s/VOLC 是独立环境、不可访问，demo **不引用其任何产物**。
- **`stock-scoring-system` 不作运行时依赖**：trainer 镜像只含本仓，装不进去。它只作**对账基准**（gross、as-of universe）。它的 forward return 带 `exit_limit_delay_days` / 止损 / regime 分支，把这些拉进 demo 会让「差异来自模型」这个结论不可解释。
- **A 轨数据**：本机 `finance.stock_scores`，按 `score_date` 取当日 hs300 成员。
- **B 轨数据**：本机 `quant_ml`（由本机 trainer 容器产出）+ 本机 PIT backfill（§4）。

### 3.1 本机环境（核实 2026-10-15）

| 零件 | 状态 |
|------|------|
| `quant-model-trainer` 容器 | ✅ 运行中，主进程 `workers.run_loop`，worker `idle`、5s 轮询、心跳正常 |
| Mongo | ✅ 容器内 `mongodb:27017`（本机 `quant-mongodb`）；`MONGO_DB=finance`、`QUANT_ML_DB=quant_ml` |
| 训练数据源 | ✅ `finance.stock_scores` hs300 **85.2 万条，2015-01-30 → 2026-10-14** |
| 价量 | ✅ `quant_data.volume_price` **1447 万条，2010-01-04 → 2026-10-15** |
| Parquet 落地 | ✅ **文件系统** `/data/quant_ml/parquet`（bind mount `quant-infrastructure/apps/data/quant_ml`，可写）；**本机无 S3/MinIO 依赖** |
| 提交入口 | ✅ 本机 `quant-api`：`POST /api/model-training/jobs` 等 6 条路由 |
| 存量训练产物 | ❌ 从未训练过（`quant_ml` 6 集合，除 worker_status 外全 0） |
| 资源 | 宿主 16 vCPU；trainer 限额 4 vCPU / 32 GB |

> 注意：本机 Mongo 返回的是 **naive UTC**，而宿主 `date` 是 CST（+8）。比对心跳/`created_at` 时必须 +8，否则会把「5 秒前」误读成「8 小时前」。

**窗口含义**：score 数据从 2015-01 起，样本充分。默认 2 年是**快速迭代**取向，不是数据上限。

> **窗口深度上限受 B 轨 PIT backfill 成本约束**：每个调仓日一次 refit，10 年 ≈ 122 次；4 vCPU 下须实测单次耗时再定。

---

## 4. Backfill（B 轨历史段）

对每个调仓日 **T**（仅当该日无 `shadow_live` 分，或显式 `--force-backfill`）：

1. 从本机 `finance.stock_scores` + `quant_data.volume_price` 拼 **PIT 面板**（`trade_date <= T`），或复用物化 Parquet 后截断到 T。
2. 取**本机训练 job 的配方**（`model_kind` / `feature_names` / `model_params`，从本机 `quant_ml.model_training_jobs` 读），在可算标签的样本上 refit：面板截断到 T，且训练窗末再退 **purge = 60 交易日**（协议 `MAX_PURGE_DAYS`，复用 `evaluation/walk_forward.py` 的口径，勿自写）。
3. 对**当日截面** `trade_date = T` 预测，标记 `score_source=backfill_pit`。
4. **非调仓日不算分**（≈12 次 refit/年 × 池数）。

**重叠期校验（必做）**：在 Shadow 与 backfill 都有的调仓日，比较两边分数的 rank correlation。复现不好 ⇒ 长曲线不可用于任何结论，只能当管道验证。

---

## 5. 实施批次

| 批次 | 内容 | 规模 | 验收 |
|------|------|------|------|
| **B0** | 冻结本契约（本节 §1）；修 doc 内的失实表述 | 0.5d | 本文件 |
| **B1** | 修 demo 记账 bug：缺失截面=持仓不变且不计换手（区分「无截面」与「前瞻未完成」）、末段不完整期剔出 NAV、`--score-column` 真正生效、`--years` 不足时明确报错；清死代码 | 1d | 单测 + 对照轨 R 相对池等权超额 ≈ 0 |
| **B2** | A 轨端到端：读 `stock_scores`（as-of 成员）→ 20 日调仓 → 主指标 + 净值；与 `ScoreBacktester.calculate_topk_summary` 同参逐日对账 | 1d | `top_vs_universe` / `universe_mean_return` 逐日一致 |
| **B3a** | **本机跑通训练闭环**：物化 hs300 快照（本机 `stock_scores` + `volume_price` → 本机 parquet root）→ 提交本机 job → worker 训练 → 本机 `quant_ml` 产出 `model_id` 与 `model_predictions` | 1d | 本机 `model_training_results` 有结果；快照落在 `apps/data/quant_ml/parquet` |
| **B3b** | B 轨接本机 `quant_ml`：按 `shadow_live_from` **日期**切片；backfill 换协议口径（purge=60 + 本机 job 配方）；重叠期 rank correlation | 1.5d | 重叠期相关性达标 |
| **B4** | 池参数化（hs300 / a500 / csi1000）、窗口 `start/end`、`metadata.json` provenance、与 `process_portfolio_research_jobs` 交叉校验 | 1.5d | 两条链路 A 轨一致 |

后续（非 MVP）：API + UI；`MODEL_SCORE_ADAPTER` 走查；日频 NAV 与日频回撤。

---

## 6. 测试策略

| 层级 | 内容 |
|------|------|
| 单元 | 调仓日历；单日 TOP-N；缺失截面语义；末段完整期判定；purge 断言（训练窗末 ≤ T − 80 交易日，horizon 20 + purge 60） |
| 集成 | 合成小面板跑通 A/B/R 三轨；`--bundle` 离线夹具 |
| 对账 | A 轨与 `ScoreBacktester` 同参数逐日一致（B2） |
| **harness 自证** | **对照轨 R 相对池等权超额 ≈ 0**；可选泄漏探针（把 T 日分数换成 T+1 日分数，harness 应显著变好） |
| 手工 | aipoc 跑全窗口，`summary.json` 人工 spot-check 3 个调仓日持仓 |

---

## 7. 风险

| 风险 | 缓解 |
|------|------|
| 12 期/年的净值差被当结论 | 报告以主指标打头；净值旁强制标注期数与功效 |
| Backfill 与 Shadow 分布不一致 | 重叠期 rank correlation 校验；不达标则长曲线只作管道验证 |
| 成本口径分歧（4 套并存） | §1.2 写死采用「费用 + 单边滑点」，并在 summary 里回显常量 |
| 本机从未训练过（`quant_ml` 空），B 轨无现成分可用 | B3a 先跑通本机闭环；A 轨与 harness（R 轨）不依赖 B 轨，可并行落地 |
| 本机 PIT backfill 计算量（每调仓日一次 refit） | 先实测单次 refit 耗时再定窗口深度；必要时降频或复用物化面板截断 |
| 训练标签（20d demean 超额）与实盘调仓（30d）错配 | 主口径 20d；30d 只作敏感性并标注错配 |
| `--years` 数据不足被静默降级 | B1 起改为显式报错 / 明确 warn |

---

## 8. CLI 草案

```bash
python -m quant_model_platform.tools.demo_hs300_20d_top20 \
  --universe hs300 \
  --start 20241001 --end 20261014 \
  --top-n 10 \
  --rebalance-days 20 \
  --growth-cycle-weights 60:40 \
  --model-id promo-hs300-2bc005de \
  --source-job-id <从 model_training_jobs 读，勿手抄> \
  --control-track random \
  --out-dir /tmp/demo-hs300
```

环境：本机 `MONGO_URI`（`finance` / `quant_data`）、`QUANT_ML_DB=quant_ml`、`QUANT_ML_PARQUET_ROOT=/data/quant_ml/parquet`。**无 S3、无远端依赖**。

---

## 9. 待拍板

- [ ] 主口径 **N=10 / 调仓 20d** 是否认可（原稿为 TOP20 / 20d；实盘是 TOP10 / 30d）
- [ ] `demo_id` 是否改名去 N（现名 `demo-hs300-20d-top20-1y` 与主口径不符）
- [x] ~~B 轨读数通道~~ → **本机自产**（K8s 不可访问；见 §1.1 / §3.1）
- [ ] 窗口：默认 2 年（快速迭代）还是拉长（score 数据 2015-01 起、价量 2010 起都够）；受 PIT backfill 成本约束，建议先实测单次 refit 耗时
- [ ] B3a 物化范围：只物化 hs300，还是顺带 a500/csi1000（B4 池参数化要用）
