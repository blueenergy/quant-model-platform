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
| 成交 | 信号日 **T**（收盘后可知）→ **T+1 开盘**；价格必须是**前复权**（`open × adj_factor`，因子 as-of 取当日之前最后一个） | 协议 `execution_price=next_open`；`ScoreBacktester` 读 `adjust="hfq"` |
| 成本 | **费用 + 滑点**两项，口径见 §1.2（**不再**写作「单边 10 bps」） | 协议 + `alpha_factor_backtest.CostAssumptions` |
| 标签 | `label_fwd_20d` = 池内 **demean** 后的超额，不是原始收益；**底价须先复权**（见 §1.4） | `dataset/labels.py::attach_label_column` 默认 `neutralize="demean"` |
| 训练/评估切分 | **purge ≥ 60 交易日**，复用 `evaluation/walk_forward.py` | 协议 `MAX_PURGE_DAYS=60` |
| 主指标 | 扣成本后**中性化 IC / rank-IC**、TOP-K 相对池等权的**逐日**超额 | 协议「成本」「模型晋级」两节 |
| 次指标 | 净值、回撤、换手 —— **仅作沟通**，须标注独立期数（n≈12/年）无统计功效 | — |
| 对照轨 | **R**：随机分数，相对池等权超额应 ≈ 0（否则 harness 有泄漏或记账错） | 协议置换检验 |
| 缺失截面 | 信号日**无可用截面** ⇒ 顺延上一期持仓，换手与成本均为 0（不空仓、不算新开仓） | B1 |
| 前瞻未完成 | 持仓**无 T+20 收益** ⇒ 该期剔出 NAV 并计入 `dropped_periods`（不作为 0 收益复合） | B1 |
| 部分定价 | 仅部分持仓有收益 ⇒ 在**有报价子集**上取均值，同时计入 `partial_periods` | B1 |
| 窗口 | 默认 **504 交易日（2 年）**，另支持 `start/end` 显式区间 | — |
| 结束日 | 基线：`max(score_date)`（hs300）；模型：与 A 调仓日交集 | — |

**指标优先级**：主证据是 IC / top-K 逐日超额（约 250 日 × 300 名 ≈ 7.5 万观测，有功效）；净值差只能作为**同名调仓日配对**的统计量，附置换/bootstrap p 值。1 年窗口的净值差**不足以**支持任何「模型更好」的结论。

**R 轨的零假设是 gross 的**：R 也要付成本，用净收益去检验「≈ 0」等于在检验成本模型而非记账；判据取 `gross_t_stat`（|t| ≤ 3），成本作为两条轨都付的确定性拖累另列。窗口不足时 R 会**报错而不是缩窗口**（B1）。

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

### 1.4 平台标签的复权缺陷（2026-10-15 修，B2 复核）

B0 冻结时点，`dataset/production.py::load_open_prices` 读的是**未复权**的 `volume_price.open`。信号日买入到持有期末卖出之间若发生送转/分红，这段价差会被当成真实收益写进标签 —— 而 `ScoreBacktester` 对的是 `adjust="hfq"`，两边测的不是同一件事。

- **修法**（`add8d17`）：`open × adj_factor`，因子按 `(symbol, trade_date)` as-of 取「该日之前最后一个」，并以窗口起点前 **45 天**的因子做种子，保证每个 symbol 的首行也有值。
- **影响面**：2y hs300 窗口内 **98.4%** 的价量行因子 ≠ 1。极端例 `002594.SZ` 2025-07-01：原始 −67.45% ≈ 3:1 送转比例，复权后 −1.20%；对照例 `601318.SH` 同日两种口径都是 +6.79%（不跨除权）。
- **实现坑**：`pandas 3.0.6` 的 `merge_asof` 不支持按 symbol 分组（`on` 键必须全局有序），改用「因子行与价格键交错后 `groupby.ffill`」。
- **已复跑 B3a**（同快照、同超参）：模型 walk-forward mean IC 从 −0.0201 → −0.0034，基线 0.0617 → 0.0748，仍是 6/18 fold、仍 `ic_below_baseline`。**两侧同向改善、排序未动** ⇒ B3a 的阴性结论不是标签缺陷造成的。

demo 侧**不复用**平台这段代码（`demo/sources.py` 自带 as-of 因子 join），否则「对账」就成了自己对自己。

---

## 2. 交付物

| 交付 | 说明 |
|------|------|
| CLI | `python -m quant_model_platform.tools.demo_hs300_20d_top20` |
| 输出目录 | `--out-dir` **现有**：`summary.json`、`equity_curve.csv`、`rebalances.csv`；**计划**：`metrics_by_date.csv`（逐日主指标，B2）、`metadata.json`（provenance，B4） |
| 文档 | 本文件 + RUNBOOK 一节「Demo 回测」 |
| 后续（可选） | `GET /api/model-training/demos/...` + 训练域 Tab（API/UI，非 MVP） |

### 2.1 `summary.json`（目标字段）

> 现状（B2）：`demo_id` / `window` / `params`（`top_n`、`rebalance_days`、`score_column`）/ `track_a` / `track_b` / `track_r` / `control_r` / `comparison` / `cost` / `source` / `caveats` 已按此落地；`contract` / `primary`（IC 那一组）/ `nav` / `backfill_validation` / `provenance` 仍是 B4 的目标字段。

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
| 存量训练产物 | B3a 前 ❌ 从未训练过（`quant_ml` 6 集合，除 worker_status 外全 0）；**B3a 后** ✅ 本机已产出 job / result / manifest，`model_predictions` 仍 0（`MODEL_SHADOW_PROFILES` 未设） |
| 资源 | 宿主 16 vCPU；trainer 限额 4 vCPU / 32 GB |

> 时区约定（2026-10-15 统一）：库内时间戳一律 **UTC**；平台**读侧已开 `tz_aware=True`**、**写侧统一走 `timeutil.utcnow()`**，新代码拿到的是 **aware UTC**，显示用 `timeutil.to_cst()`。若用裸 pymongo（未开 tz_aware）读，拿到的是 **naive UTC**，直接和宿主 `date`（CST，+8）相减会差 8 小时 —— 足以把「5 秒前」的心跳误判成「停摆 8 小时」。

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
| **B1** ✅ | 修 demo 记账 bug：缺失截面=持仓不变且不计换手（区分「无截面」与「前瞻未完成」）、末段不完整期剔出 NAV、`--score-column` 真正生效、`--years` 不足时明确报错；清死代码；落地对照轨 **R** 与成本 27bps | 1d | 单测 51 全绿 + 对照轨 R 相对池等权超额 ≈ 0（且池带溢价时**必须报警**） |
| **B2** ✅ | A 轨端到端：`--from-mongo` 读 `stock_scores`（as-of 成员）+ `quant_data` hfq 价 → 调仓网格 + 三轨净值；与 `ScoreBacktester` 真码逐 (日期, 股票) 对账 | 1d | 见 §5.1 —— 收益层 8848/8850 逐位一致，聚合层 `universe_mean_return` 12/12 一致 |
| **B3a** ✅ | **本机跑通训练闭环**：物化 hs300 快照（本机 `stock_scores` + `volume_price` → 本机 parquet root）→ 提交本机 job → worker 训练 | 1d | ✅ 快照落在 `apps/data/quant_ml/parquet`，`model_training_results` 已有结果（同参重跑 `artifact_hash` 一致 ⇒ 确定性）。⚠️ `model_predictions` 仍 0：本机未设 `MODEL_SHADOW_PROFILES`，预测段留给 B3b |
| **B3b** | B 轨接本机 `quant_ml`：按 `shadow_live_from` **日期**切片；backfill 换协议口径（purge=60 + 本机 job 配方）；重叠期 rank correlation | 1.5d | 重叠期相关性达标 |
| **B4** | 池参数化（hs300 / a500 / csi1000）、窗口 `start/end`、`metadata.json` provenance、与 `process_portfolio_research_jobs` 交叉校验 | 1.5d | 两条链路 A 轨一致 |

后续（非 MVP）：API + UI；`MODEL_SCORE_ADAPTER` 走查；日频 NAV 与日频回撤。

### 5.1 B2 对账证据（2026-10-15，`5decfed`）

跑法：trainer 容器里 `PYTHONPATH` 同时挂本仓、`stock-scoring-system`、`data-access-lib/src`，两边**读同一批 hfq 价**，`ScoreBacktester` 走自己的 `_load_symbol_price_history` + `_calculate_forward_returns`；窗口 hs300 20250923–20261014（252 个交易日、13 个调仓日）。

| 层 | 结果 |
|----|------|
| 价格清洗 | 354/354 symbol 与 SBT 输出**完全相同**（正 close、open 回退 close、排序） |
| 前瞻收益 | 8850 个 (调仓日, 股票) 对中 **8848 逐位一致**（|Δ| ≤ 1e-12） |
| 池均值 | 12/12 个调仓日 `universe_mean_return` 与 demo 池均值**完全相同** |
| top-K 均值 | 10/12 逐位一致；2 个日的差异见下 |

两处差异都已定位，都不是记账错：

1. **复牌一字板**（`600482.SH` 2024-11-11、`603019.SH` 2025-06-10）：信号日该股停牌、无 bar，SBT 的价格窗从复牌 bar 起算 ⇒ 没有前收盘 ⇒ **一字涨停的不可买判据被跳过**，它会给一个开盘根本买不到的一字板定价。demo 的窗口回溯到停牌前，直接拒绝。**这是 demo 的规则更正确，属刻意分歧**，已由 `tests/test_demo_sources.py` 固定。
2. **边界同分**：2 个日期的第 20/21 名分数**完全相等**（79.0000000000 / 79.4800000000）。A 轨分数是 2 位小数的 60:40 合成，撞分常见。demo 的 tie-break 是写死的「同分按 symbol 升序」（`demo/ranking.py::top_n_symbols`），SBT 用 `sort_values` 的不稳定排序 —— 取谁取决于喂进去的行序，**本身不是一条规则**。单期 gross 影响可达 1.5pp，见 §7。

> 对账基准刻意**不进入运行时依赖**（§3）：trainer 镜像只装本仓，`stock-scoring-system` 只在对账时挂进去。因此这条对账**不进 CI**，结论记录在本节与 vault。

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
| **边界同分**：A 轨是 2 位小数合成分，第 N/N+1 名撞分时换谁进组合可让单期 gross 摆动 ~1.5pp（实测 12 期里 2 期） | demo 的 tie-break 写死「同分按 symbol 升序」并随每组分数一起可复现；对照口径的差异已在 §5.1 量化。**解读单期净值时必须先看该期有没有撞分** —— 撞分期的净值差不构成证据 |
| Backfill 与 Shadow 分布不一致 | 重叠期 rank correlation 校验；不达标则长曲线只作管道验证 |
| 成本口径分歧（4 套并存） | §1.2 写死采用「费用 + 单边滑点」，并在 summary 里回显常量 |
| 本机从未训练过（`quant_ml` 空），B 轨无现成分可用 | B3a 先跑通本机闭环；A 轨与 harness（R 轨）不依赖 B 轨，可并行落地 |
| 本机 PIT backfill 计算量（每调仓日一次 refit） | 先实测单次 refit 耗时再定窗口深度；必要时降频或复用物化面板截断 |
| 训练标签（20d demean 超额）与实盘调仓（30d）错配 | 主口径 20d；30d 只作敏感性并标注错配 |
| ~~`--years` 数据不足被静默降级~~ | ✅ B1 已改为**显式报错**（`window_trading_days` 抛出可用天数与区间，CLI 退出码 2） |
| 单条 shadow 行顶掉整段 backfill | ✅ B1 已改为**按 `shadow_live_from` 日期切**；live 日缺分则**报错**而非回退 refit |

---

## 8. CLI

**已实现（B1/B2）** —— 三条路径：离线 `--bundle` / `--dates-file`（零 Mongo 依赖）、`--from-mongo`（本机直连）：

```bash
# 本机真实数据（B2）
python -m quant_model_platform.tools.demo_hs300_20d_top20 --from-mongo \
  --index-code hs300 --start-date 20241001 --end-date 20261014 --years 1 \
  --top-n 20 --rebalance-days 20 \
  --score-column ""            # 空 = A 轨自算 60:40；命名则读 stock_scores 该列
  --model-snapshot /data/quant_ml/parquet/snapshots/hs300_production_20241001_20261014.parquet \
  --model-kind ridge           # B 轨：有 shadow 行用 shadow 行，否则按 PIT refit
  --shadow-live-from ""        # 非空则从该日起只认已存 shadow 分
  --control-seed 20261015      # R 轨可换种子复核，不必设开关：R 恒开
  --out-dir /tmp/demo-hs300
```

- `--start-date/--end-date` 是**取数范围**；窗口是其中**最后 `--years` 个交易日**（不足则报错退出 2）。
- `--mongo-uri/--mongo-db/--data-db` 可覆盖环境变量；`summary.source` 记的是**生效库名**，不是 flag。
- `demo_id` 由参数派生（`demo-{index}-{rebalance}d-top{n}-{years}y`），A/B/R 三轨的 `top_n` 相同。
- **默认 `--top-n 20` 沿用文件与标题的原稿口径；§1 冻结的主口径是 N=10** —— 两者差一个 flag，等 §9 拍板（见 §9 第 1 条）。

**计划（B4）** —— `--universe` 参数化（hs300 / a500 / csi1000）与 `metadata.json` provenance。

环境：本机 `MONGO_URI`（`finance` / `quant_data`）、`QUANT_ML_DB=quant_ml`、`QUANT_ML_PARQUET_ROOT=/data/quant_ml/parquet`。**无 S3、无远端依赖**。

---

## 9. 待拍板

- [ ] 主口径 **N=10 / 调仓 20d** 是否认可（原稿为 TOP20 / 20d；实盘是 TOP10 / 30d）
  - 现状：CLI 默认仍是 **N=20 / 20d**（与文件名、标题、原稿一致）；主口径差一个 `--top-n 10`。**是否把默认改成 N=10 请拍板** —— 不拍板就保持默认不动的原则。
- [x] ~~`demo_id` 是否改名~~ → 改为**按参数派生**：`demo-{index}-{rebalance}d-top{n}-{years}y`（如 `demo-hs300-20d-top20-1y`），任何一次跑都自描述，不再是写死的名字。
- [x] ~~B 轨读数通道~~ → **本机自产**（K8s 不可访问；见 §1.1 / §3.1）
- [ ] 窗口：默认 2 年（快速迭代）还是拉长（score 数据 2015-01 起、价量 2010 起都够）；受 PIT backfill 成本约束，建议先实测单次 refit 耗时
- [ ] B3a 物化范围：只物化 hs300，还是顺带 a500/csi1000（B4 池参数化要用）
