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
| 股票池 | 默认 `hs300`，取 `finance.stock_scores` 的**当日** `index_codes` 成员（天然 as-of，不用固定名单）；`--index-code` 可换，但**只有 hs300 / csi500 / csi1000 有 2015 起的全历史**（a500 起 2024-09-30，csi2000 起 2026-10-09） | 实盘 plan `live_trading_default`；覆盖实测见 §5.3 |
| 基线轨 **A** | **自算** `0.6 × growth_score + 0.4 × cycle_score`（可用权重归一；缺一维则取另一维） | 实盘 `score_type=growth_cycle_weighted`，`growth_weight=0.6` / `cycle_weight=0.4` |
| 模型轨 **B** | `model_id` 由 Shadow 登记提供，CLI 可覆盖；backfill 段用**晋升 job 的配方**（从 `model_training_jobs` 读，不手抄） | — |
| 选股 | TOP **N**，`selection_mode=fixed_top_n`，**等权** | — |
| N（主/敏感） | 主 **N=10**（与实盘一致）；**N=20** 作敏感性 | 实盘 `top_n=10` |
| 调仓（主/敏感） | 主 **20 交易日**（与标签 horizon 一致）；**30 交易日**（与实盘一致）作敏感性 | 标签 `LABEL_HORIZON_DAYS=20`；实盘 `rebalance_days=30` |
| 成交 | 信号日 **T**（收盘后可知）→ **T+1 开盘**；价格必须是**前复权**（`open × adj_factor`，因子 as-of 取当日之前最后一个） | 协议 `execution_price=next_open`；`ScoreBacktester` 读 `adjust="hfq"` |
| 成本 | **费用 + 滑点**两项，口径见 §1.2（**不再**写作「单边 10 bps」） | 协议 + `alpha_factor_backtest.CostAssumptions` |
| 标签 | `label_fwd_20d` = **池内 demean** 后的超额，不是原始收益；**底价须先复权**（见 §1.4）。**池内 demean** = 逐日减去**当天全体有标签样本的等权均值**（`groupby(date)[label] - mean`）；那个均值**就是等权池收益**，所以减完剩下的是「相对 hs300 等权」的超额 —— 也正是对照轨 R 检验的那条基线。口径细节：`pandas.mean()` 跳过 NaN ⇒ 基线只含**当天有标签**的名字 | `dataset/labels.py::attach_label_column` 默认 `neutralize="demean"` |
| 训练/评估切分 | **purge ≥ 60 交易日**，复用 `evaluation/walk_forward.py` | 协议 `MAX_PURGE_DAYS=60` |
| 主指标 | 扣成本后**中性化 IC / rank-IC**、TOP-K 相对池等权的**逐日**超额 | 协议「成本」「模型晋级」两节 |
| 次指标 | 净值、回撤、换手 —— **仅作沟通**，须标注独立期数（n≈12/年）无统计功效 | — |
| 对照轨 | **R**：随机分数，相对池等权超额应 ≈ 0（否则 harness 有泄漏或记账错） | 协议置换检验 |
| 缺失截面 | 信号日**无可用截面** ⇒ 顺延上一期持仓，换手与成本均为 0（不空仓、不算新开仓） | B1 |
| 前瞻未完成 | 持仓**无 T+20 收益** ⇒ 该期剔出 NAV 并计入 `dropped_periods`（不作为 0 收益复合） | B1 |
| 部分定价 | 仅部分持仓有收益 ⇒ 在**有报价子集**上取均值，同时计入 `partial_periods` | B1 |
| 窗口 | **`--start-date`/`--end-date` 覆盖的全部历史**（`--years 0`，B4 起的默认）；`--years N` 截最后 N 年 | §5.3：12 期样本无功效 |
| 窗口 ≠ 面板 | **评估窗口**（demo 跑多久）与**物化面板窗口**（喂给模型的训练历史）是两件事，报告里分别记 | B4 `metadata.json` |
| 结束日 | 基线：`max(score_date)`（hs300）；模型：与 A 调仓日交集 | — |

**指标优先级**：主证据是 IC / top-K 逐日超额（约 250 日 × 300 名 ≈ 7.5 万观测，有功效）；净值差只能作为**同名调仓日配对**的统计量，附置换/bootstrap p 值。1 年窗口的净值差**不足以**支持任何「模型更好」的结论。

> ⚠️ **B4 后仍未落地**：上面的「主指标」IC / rank-IC 序列**没有实现**（见 §2.1）。现在能拿到的只有每期收益的 `t_stat` / `gross_t_stat` 与 `comparison.excess_return_b_minus_a`，加上 §5.2 那组**重叠期** rank correlation。**在这条补上之前，本契约的"主证据"实际上是空的**，任何结论都只能靠次指标 —— 这正是 §5.2/§5.3 反复出现"结论随旋钮翻转"的原因。这是 B5 的第一优先级。

**R 轨的零假设是 gross 的**：R 也要付成本，用净收益去检验「≈ 0」等于在检验成本模型而非记账；判据取 `gross_t_stat`（|t| ≤ 3），成本作为两条轨都付的确定性拖累另列。窗口不足时 R 会**报错而不是缩窗口**（B1）。

**调仓周期与标签的错配**：标签 horizon 是 20 日，实盘调仓是 30 日。主口径取 20 日（与标签一致），30 日作为「实盘会拿到什么」的敏感性 —— 若 20 日口径都赢不了，30 日不必看。

### 1.1 模型分来源（B 轨）

**B 轨全部本机自产**。K8s/VOLC 是**独立环境、不可访问**，其 `quant_ml`（含既有 `model_predictions` 与 Shadow 观察）本机够不着 ⇒ demo **不引用任何远端产物**。

| 区间 | 来源 | Mongo 标记 |
|------|------|------------|
| 本机 Shadow 启用之后 | 本机 `quant_ml.model_predictions` | `score_source=shadow_live` |
| 更早调仓日 | 本机 **PIT backfill**（见 §4），按协议 purge=60 | `score_source=backfill_pit` |

⚠️ ~~本机 `quant_ml` 现为空库~~ → **2026-10-15 起已有本机产物**：B3a 跑通训练闭环（job / result / 规范快照），B3b 跑通 shadow（`model_predictions` 300 条）。B 轨的 `model_id` 由本机 job 派生（`promo-{index}-{short_job}`），**不引用任何远端 `model_id`**。

**边界按日期切，不按「当天有没有分」切** —— 半覆盖的截面不得当完整截面用。

**禁止**：用「全历史 Parquet 一次 refit」给所有过去 T 打分（未来信息泄漏）。  
Forward **8–12 周 Shadow 观察**若要做，须在**本机**新起（本机 `model_shadow_profiles` 已登记 `hs300-20d-ridge`，日更尚未挂 cron），且只认 `shadow_live` 段，不能用 backfill 冒充。  
**可测替代**：在 Shadow 与 backfill 重叠的调仓日做 rank correlation —— 复现得好，长曲线才可信（B3b 已落地，结果见 §5.2）。

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
| 输出目录 | `--out-dir`：`summary.json`、`equity_curve.csv`、`rebalances.csv`、**`metadata.json`**（B4：provenance，且**最后写** ⇒ 它的 digest 覆盖其余三个）；**计划**：`metrics_by_date.csv`（逐日主指标，B2） |
| 文档 | 本文件 + RUNBOOK 一节「Demo 回测」 |
| 后续（可选） | `GET /api/model-training/demos/...` + 训练域 Tab（API/UI，非 MVP） |

### 2.1 产物字段（B4 后实况）

原稿把 provenance 放在 `summary.json` 里。B4 改成**两个文件**：`summary.json` 只装**这一次评估的结果**，`metadata.json` 只装**这次跑的环境与协议**（见 §5.3 第 1 点）。理由是 provenance 里含产物摘要，而"摘要自己"必须最后算 —— 混在一个文件里做不到。

`summary.json`（实测键位，冒烟产物 `/tmp/demo_smoke_host`）：

```
demo_id / status / years / top_n / rebalance_days / backfill_purge_days / score_column
model_id / source_job_id
source{kind, mongo_db, data_db, index_code, coverage_start, coverage_end}   ← 生效库名，不是 flag
model_source{kind, source_job_id, snapshot_id, parquet_path, model_id, model_kind,
             live_rows, live_dates, shadow_live_from}
model_overlap          ← 原稿的 backfill_validation：live↔backfill rank correlation
cost{per_side_bps, round_trip_bps, source, note}
window_start / window_end / window_trading_days / rebalance_dates[] / rebalance_count
track_a / track_b / track_r  ← total_return, max_drawdown, ann_vol, mean_period_return,
                               t_stat, gross_mean_period_return, gross_t_stat,
                               avg_turnover, periods, rebalance_days, dropped_periods,
                               partial_periods, carried_periods, score_column
control_r{seed, t_stat, t_max, mean_period_return, net_total_return, periods, passed, note}
comparison{excess_return_b_minus_a, avg_overlap_ratio, avg_overlap_baseline_vs_random}
caveats[]
```

两处与原稿不同，记在这里免得再被当成"未实现"：

- **`primary`（IC 那一组）不存在**，B4 也没做。现值是 `t_stat` / `gross_t_stat`（每期收益的 t）与 `comparison.excess_return_b_minus_a`；rank IC 只有 §5.2 的**重叠期**那组诊断，不是逐日 IC 序列。
- **`nav` 不存在**：净值在 `equity_curve.csv`（调仓网格口径，**非日频**）。`caveats[]` 里带上期数与功效限制，报告与 UI 必须**先展示主指标**，净值旁标注口径 —— 这条纪律不变。

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
| 存量训练产物 | B3a 前 ❌ 从未训练过（`quant_ml` 6 集合，除 worker_status 外全 0）；**B3a 后** ✅ job / result / 规范快照；**B3b 后** ✅ `model_predictions` 300 条（`model_id=promo-hs300-b7618410`，`trade_date=20261014`），`model_shadow_profiles` 1 条 |
| 资源 | 宿主 16 vCPU；trainer 限额 4 vCPU / 32 GB |

> 时区约定（2026-10-15 统一）：库内时间戳一律 **UTC**；平台**读侧已开 `tz_aware=True`**、**写侧统一走 `timeutil.utcnow()`**，新代码拿到的是 **aware UTC**，显示用 `timeutil.to_cst()`。若用裸 pymongo（未开 tz_aware）读，拿到的是 **naive UTC**，直接和宿主 `date`（CST，+8）相减会差 8 小时 —— 足以把「5 秒前」的心跳误判成「停摆 8 小时」。

**窗口含义**：score 数据从 2015-01 起（§5.3 的池覆盖表），样本充分。**B4 起默认全历史**（`--years 0`）—— 2 年那档只是快速迭代取向，从来不是数据上限。

> **窗口深度不受 B 轨 backfill 成本约束**（实测）：每个调仓日一次 refit，13 个调仓日 + overlap 检查共 ~25 次 refit 端到端约 **10s**（4 vCPU），全历史 ~142 次 ⇒ **分钟量级**。真正的约束是统计功效（§5.3 第 4 点），不是算力。

---

## 4. Backfill（B 轨历史段）

对每个调仓日 **T**（该日 < `--shadow-live-from`；live 日缺分**报错**，不回退 refit）：

1. 从本机 `finance.stock_scores` + `quant_data.volume_price` 拼 **PIT 面板**（`trade_date <= T`），或复用物化 Parquet 后截断到 T（`--source-job-id` 走**规范快照**）。
2. 取**本机训练 job 的配方**（`model_kind` / `feature_names` / `model_params` / `label_col`，从本机 `quant_ml.model_training_jobs` 读，**不手抄**），在可算标签的样本上 refit：面板截断到 T，且训练窗末再退 **purge = 60 交易日**（协议 `MAX_PURGE_DAYS`，`demo/backfill_pit.py::pit_train_dates`）。
3. 对**当日截面** `trade_date = T` 预测，标记 `score_source=backfill_pit`。
4. **非调仓日不算分**（≈12 次 refit/年 × 池数；实测 13 个调仓日 + overlap 检查共 ~25 次 refit 端到端约 **10s**，全历史 ~142 次 ⇒ 分钟量级）。

> **purge 的双重身份（2026-10-15 补）**：`purge=60` 约束的是**评估切分**，不是实盘 refit —— 实盘 shadow 按定义用**全部有标签的行**（等效 `purge=0`，训练窗末 = T−21）。所以 backfill 与 live 分**本来就不是同一件事**，`--backfill-purge` 只是把这道差量摆到台面上（敏感性见 §5.2）。改这个值等于改 B 轨的口径，不是调优。

**重叠期校验（必做，B3b 已落地）**：在 Shadow 与 backfill 都有的调仓日比较两边分数的 rank correlation（`--check-overlap`，判据 `MIN_OVERLAP_SPEARMAN=0.8`）。复现不好 ⇒ 长曲线不可用于任何结论，只能当管道验证。结果见 §5.2。

---

## 5. 实施批次

| 批次 | 内容 | 规模 | 验收 |
|------|------|------|------|
| **B0** | 冻结本契约（本节 §1）；修 doc 内的失实表述 | 0.5d | 本文件 |
| **B1** ✅ | 修 demo 记账 bug：缺失截面=持仓不变且不计换手（区分「无截面」与「前瞻未完成」）、末段不完整期剔出 NAV、`--score-column` 真正生效、`--years` 不足时明确报错；清死代码；落地对照轨 **R** 与成本 27bps | 1d | 单测 51 全绿 + 对照轨 R 相对池等权超额 ≈ 0（且池带溢价时**必须报警**） |
| **B2** ✅ | A 轨端到端：`--from-mongo` 读 `stock_scores`（as-of 成员）+ `quant_data` hfq 价 → 调仓网格 + 三轨净值；与 `ScoreBacktester` 真码逐 (日期, 股票) 对账 | 1d | 见 §5.1 —— 收益层 8848/8850 逐位一致，聚合层 `universe_mean_return` 12/12 一致 |
| **B3a** ✅ | **本机跑通训练闭环**：物化 hs300 快照（本机 `stock_scores` + `volume_price` → 本机 parquet root）→ 提交本机 job → worker 训练 | 1d | ✅ 快照落在 `apps/data/quant_ml/parquet`，`model_training_results` 已有结果（同参重跑 `artifact_hash` 一致 ⇒ 确定性）。⚠️ `model_predictions` 仍 0：本机未设 `MODEL_SHADOW_PROFILES`，预测段留给 B3b |
| **B3b** ✅ | B 轨接本机 `quant_ml`：`--source-job-id` 解析**配方 + 规范快照**；backfill 换协议口径（purge=60）；读**真** `model_predictions`；重叠期 rank correlation | 1.5d | 见 §5.2 —— live/backfill ρ=0.8702（n=1 天，冒烟级）；**purge 敏感性：同一面板同一模型，B−A 可在 −10.03pp 与 +12.11pp/年之间取值** |
| **B4** ✅ | 池参数化（`--index-code`）、窗口 `start/end`（`--years 0` = 全历史）、`metadata.json` provenance、与 `process_portfolio_research_jobs` 交叉校验 | 1.5d | 见 §5.3（参数化 + 池覆盖 + 功效）与 §5.4（交叉校验）——A 轨一致性到 SBT 层（§5.1）；两条链路**成本口径本就不同**，端到端数字不可能相等 |

后续（非 MVP）：API + UI；`MODEL_SCORE_ADAPTER` 走查；日频 NAV 与日频回撤。

### 5.1 B2 对账证据（2026-10-15，`5decfed`）

跑法：trainer 容器里 `PYTHONPATH` 同时挂本仓、`stock-scoring-system`、`data-access-lib/src`，两边**读同一批 hfq 价**，`ScoreBacktester`（下称 **SBT**，`stock-scoring-system/analysis/core/score_backtester.py`，只作对账基准）走自己的 `_load_symbol_price_history` + `_calculate_forward_returns`；窗口 hs300 20250923–20261014（252 个交易日、13 个调仓日）。

| 层 | 结果 |
|----|------|
| 价格清洗 | 354/354 symbol 与 SBT 输出**完全相同**（正 close、open 回退 close、排序） |
| 前瞻收益 | 8850 个 (调仓日, 股票) 对中 **8848 逐位一致**（\|Δ\| ≤ 1e-12） |
| 池均值 | 12/12 个调仓日 `universe_mean_return` 与 demo 池均值**完全相同** |
| top-K 均值 | 10/12 逐位一致；2 个日的差异见下 |

两处差异都已定位，都不是记账错：

1. **复牌一字板**（`600482.SH` 2024-11-11、`603019.SH` 2025-06-10）：信号日该股停牌、无 bar，SBT 的价格窗从复牌 bar 起算 ⇒ 没有前收盘 ⇒ **一字涨停的不可买判据被跳过**，它会给一个开盘根本买不到的一字板定价。demo 的窗口回溯到停牌前，直接拒绝。**这是 demo 的规则更正确，属刻意分歧**，已由 `tests/test_demo_sources.py` 固定。
2. **边界同分**：2 个日期的第 20/21 名分数**完全相等**（79.0000000000 / 79.4800000000）。A 轨分数是 2 位小数的 60:40 合成，撞分常见。demo 的 tie-break 是写死的「同分按 symbol 升序」（`demo/ranking.py::top_n_symbols`），SBT 用 `sort_values` 的不稳定排序 —— 取谁取决于喂进去的行序，**本身不是一条规则**。单期 gross 影响可达 1.5pp，见 §7。

> 对账基准刻意**不进入运行时依赖**（§3）：trainer 镜像只装本仓，`stock-scoring-system` 只在对账时挂进去。因此这条对账**不进 CI**，结论记录在本节与 vault。

### 5.2 B3b 证据（2026-10-15）

**B 轨现在真的接在本机 `quant_ml` 上**

- **配方 + 面板**：`--source-job-id b7618410-…` → 与 shadow 走**同一个解析器**（`shadow.resolve.merge_job_into_shadow_params` + `load_training_panel`），拿到的是**规范快照** `hs300_current.parquet`（`snapshot_id=hs300-current`，147,000 行 = 490 日 × 300 名）。跑之前先核过：该文件与 B3a 的 job 快照 **md5 完全相同**（`f42a8386…`）⇒ 两次跑的 B 轨差**不来自面板**。
- **真 live 分**：登记 `model_shadow_profiles/hs300-20d-ridge` → `python -m quant_model_platform.workers.shadow_loop --once` → `model_predictions` **300 条**（`model_id=promo-hs300-b7618410`，`trade_date=20261014`）。此前该集合恒为 0。
- **切片**：本跑 `--shadow-live-from 20261014`，13 个调仓日全部早于边界 ⇒ B 轨仍全走 backfill；live 段要等本地 shadow 日更跑起来才长出来（**尚未挂 cron**）。

**重叠期 rank correlation（契约 §4 的验收项）**

| 比较 | 天数 | 名数 | mean ρ | min ρ | 判据 0.8 |
|---|---|---|---|---|---|
| live 分 vs backfill（同日 20261014） | 1 | 300 | **0.8702** | 0.8702 | 过，但 **n=1 只是冒烟** |
| purge=0（= live 配方）vs purge=60，逐调仓日 | 13 | 300 | 0.8473 | **0.2255** | **不过**（2/13 天 ρ<0.4） |

第二行是本次新增的诊断，把 §7「Backfill 与 Shadow 分布不一致」那条风险量化了：**训练窗差 60 天，排名就会被重排**，个别日子（20251029 ρ=0.23、20250923 ρ=0.36）几乎没什么共同意见。

**purge 敏感性 —— 本节最该看的东西**

B0 冻结的 `purge ≥ 60` 直到 B3b 才真的落在 backfill 上（此前只退了 `horizon+1 = 21` 天）。补上之后，同一块面板、同一个模型：

| 训练窗末距 T | flag | B 每期 | t | B−A（年化） | 组合重叠 |
|---|---|---|---|---|---|
| 21 交易日（**旧行为** = live 配方） | `--backfill-purge 0` | +3.0007% | 0.763 | **−10.03pp** | 72.7% |
| 42 交易日 | `--backfill-purge 21` | +3.6420% | 0.928 | +0.23pp | 65.4% |
| 81 交易日（**契约主口径**） | `--backfill-purge 60`（默认） | +4.2153% | 1.155 | **+12.11pp** | 53.8% |

A 轨三次**逐位相同**（+3.5532%/期，t=0.9358，gross t=0.9586）⇒ 面板、记账、调仓网格都没动，**差异全部来自禁运期**。`--backfill-purge 0` 精确复现了 B2 那跑的 B（+3.0007%/t=0.763/−10.03pp），证明这次重构本身不改行为。

**结论（须随之一起引用）**：在这 12 期样本上，「B 打不打得过 A」**不由数据决定，而由禁运期决定** —— −10pp 到 +12pp 全都能取到，且三条的 t 都 ≤ 1.16。这不支持任何方向的结论；它说明在动 N / 窗口这些旋钮之前，purge 先得讲清楚。**B 轨目前只能算「管道通了」，不能算「有效」。**

### 5.3 B4 证据（2026-10-15）：参数化、provenance、窗口与功效

**1. 旋钮全部进 provenance，窗口默认改为「手上全部历史」**

`--start-date/--end-date/--years/--top-n/--rebalance-days/--index-code/--backfill-purge` 原样记进 `metadata.json` 的 `requested`。`--years` 默认从 2 改成 **0 = 用 `--start-date/--end-date` 覆盖到的每一个交易日**（不再由 CLI 猜一个窗口）。`--top-n` 默认从 20 改成 **10**（主口径，§9 已拍板）。

`demo_id` 随之改成**真实窗口边界**而不是 `{years}y`：`--years 0` 之后，"3y" 这种标签会和它实际跑的数据对不上，而标签是对数据的断言。冒烟产物：`demo-hs300-20d-top10-20250923-20261014`。

`metadata.json` **最后写**，所以它的 `outputs[]` digest 覆盖同目录其余全部产物（`summary.json` / `equity_curve.csv` / `rebalances.csv`，含 size + sha256）；自身不在其中。另含：`argv`、`image`（`image_tag` + OCI `revision`）、`protocol`（`label_horizon_days=20`、`max_purge_days=60`、`backfill_purge_days`、`slippage_bps`、`round_trip_bps=27`、`control_seed`、`control_t_max`）、`window`（首尾日 + `rebalance_count`）、`data`（`track_b` 实际打分的日数与名数）、`model`（`model_id` / `source_job_id`）。

**2. 池覆盖：契约原计划的 a500 撑不起长窗口**（2026-10-15 实测 `finance.stock_scores`）

| `--index-code` | 行数 | 起始 | 是否够全历史 |
|---|---|---|---|
| `hs300` | 852,300 | **2015-01-30** | ✅（2,841 日 × 300 名） |
| `csi500` | 1,420,495 | **2015-01-30** | ✅ |
| `csi1000` | 2,841,000 | **2015-01-30** | ✅ |
| `a500` | 245,500 | 2024-09-30 | ❌ 只有 ~2 年 |
| `star50` | 75,100 | 2020-07-31 | ❌ |
| `csi2000` | 8,000 | 2026-10-09 | ❌ 只有 4 天 |

⇒ 长窗口只能在 **hs300 / csi500 / csi1000** 三个池上做。a500 的"全历史"实际是 2024-09-30 起（指数本身 2024 年才发布），**换池等于换窗口**，两者不能分别调。

**3. 窗口 ≠ 面板**：**评估窗口**是 demo 在多久的数据上打分/记账（由 `--start-date/--end-date` 决定）；**物化面板窗口**是喂给模型的训练历史（由物化 job 的快照决定，`hs300_current.parquet` = 490 日）。B 轨每期还在面板内部按 `purge` 截训练窗，所以**三者是三件事**，报告里分别记（`window` / `data` / `protocol.backfill_purge_days`）。

**4. 功效：1 年窗口 = 12 期，这是所有"翻转"的根源**

| 窗口 | 交易日 | 调仓期 | 相对 12 期的 t 倍数 |
|---|---|---|---|
| 1 年（旧默认） | 252 | 12 | 1× |
| 3 年 | 756 | ~37 | ~1.8× |
| **全历史 2015-01-30 起** | **2,841** | **~142** | **~3.4×** |

t 按 √期数 缩放，所以把窗口从 1 年放到全历史，同样一个 B−A 的 t 大约 ×3.4 —— 这不是"更好看"，是**能不能被数据支持**的分界。下面第 5 点是同一窗口换 `--top-n` 的实测，用来钉住"12 期决定不了任何事"。

**5. 同一 1 年窗口、同样 `purge=60`，只换 `--top-n`**（12 期，单位统一为「每期」）

| 配置 | A 每期 (t) | B 每期 (t) | **B−A** | 组合重叠 | R（gross t） |
|---|---|---|---|---|---|
| `--top-n 20` | +3.5532% (0.9358) | +4.2153% (1.155) | **+0.66 pp/期** | 53.8% | — |
| `--top-n 10`（主口径） | +1.6526% (0.4204) | +5.1367% (1.0957) | **+4.07 pp/期**（累计 +48.79pp） | 48.46% | 0.7477 ✅ |

换一个 N，A 掉一半多（+3.55 → +1.65）、B 反涨（+4.22 → +5.14），B−A 从 +0.66 变 +4.07 pp/期 —— 两边 t 都 < 1.2。**同一个窗口里，动旋钮的幅度大于数据本身的信号**，这就是为什么 B4 必须先扩窗口再谈结论。

### 5.4 B4 交叉校验：`process_portfolio_research_jobs`（2026-10-15）

**它是什么**：`stock-scoring-system/tools/process_portfolio_research_jobs.py:567 run_research_job` → `analysis/sweep/portfolio_sweep_report.py:403 run_parameter_sweep` → `ScoreBacktester` → `PortfolioBacktester`，结果落 `quant_trading.portfolio_research_jobs/_results`。默认池 `csi1000`、`score_column=composite_growth_cycle_score`、`horizon/rebalance=20`、`top_n=[10,20,50,100]`。

**与 demo 共享的代码 = 0**（双向 grep 均 0 命中，是刻意的，见 §3）；共享的只有两张**只读**原始表 `finance.stock_scores`、`quant_data.volume_price`（+ `stock_adj_factor`）。

| 轴 | portfolio research | demo | 判定 |
|---|---|---|---|
| 模型 | 无（纯规则分参数扫描） | track B 真模型 | 不可比 |
| purge | **全仓不存在**（`purge`/`embargo` 只剩 `Jenkinsfile` 的 `pip cache purge`） | `MAX_PURGE_DAYS = 60` | 结构差异，不是同量纲冲突 |
| 对照轨 | **无**（无 placebo / shuffle / 随机分） | track R + `CONTROL_T_MAX = 3.0` | demo 更强 |
| 成本 | **10 bps，不含滑点**（legacy `transaction_cost=0.001`；一旦传显式费率，`TradeFeeModel.from_params` 会**忽略**它） | **27 bps**（费用 7 + 单边滑点 10×2） | **唯一实质冲突**，§1.2 已冻结取 27 |
| 前瞻收益 | **原始** `exit/entry − 1`（`score_backtester.py:979`），超额留到指标层算 `top_mean − universe_mean` | 评估层同样原始（`sources.py`）；`demean` 只用于**训练标签**（`dataset/labels.py`） | **不矛盾**——两条链路的评估层口径一致 |
| 股票池 PIT | 降级阶梯：asof 快照 → 今日成分，并把 `universe_pit_quality` 记进 `data_watermark` | 分数文档自带的 `index_codes`（与 `score_backtester.py` 的 pushdown 同口径） | demo 少依赖一层快照，但也**没有** survivorship 告警机制 |

**验收项「两条链路 A 轨一致」的成立范围**：成立，**但只到 SBT 层**——§5.1 的 354/354 价格清洗、8848/8850 前瞻收益、12/12 池均值逐位一致，对的是 demo 与 research **同一个回测引擎**在同一份数据上的行为。**没有**从 `run_parameter_sweep` 端到端复跑一个同窗口数字，因为两条链路成本口径本就不同（10 vs 27 bps），端到端净值**不可能**相等；能逐位对账的只有剥掉成本的那一层。**跨链路引用数字时必须换算成本，或只用 gross。**

**顺带一条**：research 的排序键是 `index_excess_cumulative_return − 0.5·|dd| − 0.15·turnover − 0.1·active_share`（`portfolio_sweep_report.py:894`），**带网格选择偏差**（网格越大，"最优行"越乐观）。这跟 §5.3 第 5 点里 demo「同一个 12 期窗口，B−A 能随旋钮从 +0.66 挑到 +4.07 pp/期」是同一个坑。

---

## 6. 测试策略

| 层级 | 内容 |
|------|------|
| 单元 | 调仓日历；单日 TOP-N；缺失截面语义；末段完整期判定；**purge 断言**（✅ B3b：`test_default_purge_is_the_protocol_embargo` 钉住训练窗末 = T − horizon − purge，另有 `purge=0` = live 配方的对照）；重叠期 rank correlation（同序=1、反序=失败、名数不足不硬算） |
| 集成 | 合成小面板跑通 A/B/R 三轨；`--bundle` 离线夹具；`model_predictions` 读取按 `model_id`/`shadow` 过滤（假集合） |
| 对账 | A 轨与 `ScoreBacktester` 同参数逐日一致（B2）；B 轨 live 分 vs backfill rank correlation（B3b，§5.2） |
| **harness 自证** | **对照轨 R 相对池等权超额 ≈ 0**；可选泄漏探针（把 T 日分数换成 T+1 日分数，harness 应显著变好） |
| 手工 | aipoc 跑全窗口，`summary.json` 人工 spot-check 3 个调仓日持仓 |

---

## 7. 风险

| 风险 | 缓解 |
|------|------|
| 12 期/年的净值差被当结论 | 报告以主指标打头；净值旁强制标注期数与功效 |
| **结论被配置而非数据决定**（2026-10-15 实测）：同一个 1 年窗口、同样 12 期，只看「B 每期 − A 每期」（`mean_period_return` 之差，同一单位）：`purge` 取 0 / 21 / 60 → **−0.55 / +0.09 / +0.66 pp/期**；固定 `purge=60` 再动 `--top-n` 20 → 10 → **+0.66 / +4.07 pp/期**。全部 t < 1.2 | ✅ B4 已把窗口放到全历史（§5.3）：138 期的功效是 12 期的 ~3.5 倍（t 按 √期数 缩放）。**在此之前，任何"某档配置更好"的读法都不成立** |
| **边界同分**：A 轨是 2 位小数合成分，第 N/N+1 名撞分时换谁进组合可让单期 gross 摆动 ~1.5pp（实测 12 期里 2 期） | demo 的 tie-break 写死「同分按 symbol 升序」并随每组分数一起可复现；对照口径的差异已在 §5.1 量化。**解读单期净值时必须先看该期有没有撞分** —— 撞分期的净值差不构成证据 |
| **B 轨结论随禁运期翻转**（2026-10-15 实测）：同一面板同一模型，`purge=0/21/60` 对应 B−A = **−10.03 / +0.23 / +12.11pp** 每年，三条 t ≤ 1.16 | 报告 B 时必须同时给三档敏感性（§5.2）；**不准单独引用任一档当结论**。契约主口径仍是 purge=60，因为它是冻结值，不是因为它是最好看的那个 |
| Backfill 与 Shadow 分布不一致 | ✅ B3b 已量化：live vs backfill ρ=0.8702（n=1）；purge=0 vs 60 逐日 mean 0.8473 / min 0.2255 ⇒ 长曲线只作管道验证 |
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
# 本机真实数据（B2/B3b/B4）—— 推荐：B 轨配方与面板都从 quant_ml 的 job 解析
python -m quant_model_platform.tools.demo_hs300_20d_top20 --from-mongo \
  --index-code hs300 --start-date 20150130 --end-date 20261014 \
  --top-n 10 --rebalance-days 20 \
  --score-column ""            # 空 = A 轨自算 60:40；命名则读 stock_scores 该列
  --source-job-id b7618410-…   # B 轨：从 model_training_jobs 读配方 + 规范快照
  --shadow-live-from 20261014  # 该日及以后只认 model_predictions，缺分报错
  --backfill-purge 60          # 协议禁运期；默认 60，降档只为做敏感性
  --check-overlap              # 额外输出 live/purge 的 rank correlation（多几次 refit）
  --control-seed 20261015      # R 轨可换种子复核，不必设开关：R 恒开
  --out-dir /tmp/demo-hs300

# 离线替代：不给 --source-job-id 时，用 --model-snapshot <parquet> + --model-kind 顶替
```

**Shadow 侧（本机）**：先 `materialize_snapshot --canonical --register-mongo`（规范快照是
`model_shadow_profiles` 的前置），再 `register_shadow_profile --profile-id … --source-job-id …`，
然后 `python -m quant_model_platform.workers.shadow_loop --once`。**日更 cron 尚未挂**。

- `--start-date/--end-date` 是**取数范围**；`--years 0`（**B4 起的默认**）= 取数范围里的每一个交易日，
  `--years N` = 只取最后 N 年（历史不足 N 年**报错退出 2**，从不静默缩窗）。
- `--mongo-uri/--mongo-db/--data-db` 可覆盖环境变量；`summary.source` 记的是**生效库名**，不是 flag。
- `demo_id` 由参数派生且用**真实窗口边界**：`demo-{index}-{rebalance}d-top{n}-{首日}-{末日}`。
  A/B/R 三轨共用同一个 `top_n`。
- **默认 `--top-n 10` 就是 §1 冻结的主口径**（B4 起）。

**B4 已落地**：`--index-code` 参数化（长窗口只有 `hs300`/`csi500`/`csi1000` 可选，§5.3）+ `metadata.json` provenance。

环境：本机 `MONGO_URI`（`finance` / `quant_data`）、`QUANT_ML_DB=quant_ml`、`QUANT_ML_PARQUET_ROOT=/data/quant_ml/parquet`。**无 S3、无远端依赖**。

---

## 9. 待拍板

- [ ] **B 轨口径（B3b 新出，优先级最高）**：backfill 用 `purge=60`（契约冻结值）还是 `purge=0`（= 实盘 shadow 的 refit 方式）？
  - 现状：`--backfill-purge` 默认 **60**（照契约）。同一 12 期窗口上，两档的 B−A 是 **−0.55 与 +0.66 pp/期**（§5.2），而**实盘 shadow 走的是 purge=0 那条**。
  - 这不是调参问题，是「B 轨到底在测谁」的定位问题：测「协议下的模型」还是「实盘会拿到的模型」。**请拍板，不拍板就保持 60 不动**；B4 建议用**全历史**这一跑把三档重做一遍再定（12 期上三档的 t 全 ≤ 1.16，分不出）。
- [x] ~~主口径 **N=10 / 调仓 20d** 是否认可~~ → **2026-10-15 拍板认可**。CLI 默认已从 20 改为 **10**（§5.3 第 1 点）；20 保留为敏感性档（§5.3 第 5 点：同一窗口两档 B−A 差 6 倍）。
- [x] ~~`demo_id` 是否改名~~ → 改为按参数派生；B4 起用**真实窗口边界**而非 `{years}y`（`--years 0` 之后年份标签会和数据对不上）。
- [x] ~~B 轨读数通道~~ → **本机自产**（K8s 不可访问；见 §1.1 / §3.1）
- [x] ~~窗口：默认 2 年还是拉长~~ → **2026-10-15 决定拉长到全历史**，`--years 0` 成为默认（§5.3 第 4 点：12 期 → ~142 期，t 功效 ×3.4）。成本不是约束（13 个调仓日实测端到端约 10s；全历史 ~142 次 refit ⇒ 分钟量级）。
  - ⚠️ **换池 = 换窗口**：`a500` 的分数只从 2024-09-30 起，`csi2000` 只从 2026-10-09 起 ⇒ 长窗口实际只能在 `hs300`/`csi500`/`csi1000` 上做（§5.3 第 2 点）。
- [ ] B3a 物化范围：目前只物化了 `hs300`（面板 `hs300_current.parquet`，490 日）。换 `csi500`/`csi1000` 跑长窗口**需要各自物化一份面板**（B 轨要面板才能 refit）。
- [ ] 本机 shadow **日更 cron** 要不要挂（不挂 ⇒ live 段永远是 1 天、§5.2 的 ρ 永远 n=1；挂 ⇒ 8–12 周后才有可用的 live 观察段）
