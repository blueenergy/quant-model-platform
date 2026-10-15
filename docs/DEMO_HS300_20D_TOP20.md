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
| 主指标 | **rank-IC（Spearman）与 Pearson IC**，在**调仓日**网格上算；TOP-K 相对**池等权**的逐期超额（gross 与 net 并列）。**IC 不扣成本**（见下）；「中性化」在这里是恒等变换（见下） | 协议「模型晋级」；口径订正见 §5.6 |
| 次指标 | 净值、回撤、换手 —— **仅作沟通**，须标注独立期数（n≈12/年）无统计功效 | — |
| 对照轨 | **R**：随机分数，相对池等权超额应 ≈ 0（否则 harness 有泄漏或记账错） | 协议置换检验 |
| 缺失截面 | 信号日**无可用截面** ⇒ 顺延上一期持仓，换手与成本均为 0（不空仓、不算新开仓） | B1 |
| 前瞻未完成 | 持仓**无 T+20 收益** ⇒ 该期剔出 NAV 并计入 `dropped_periods`（不作为 0 收益复合） | B1 |
| 部分定价 | 仅部分持仓有收益 ⇒ 在**有报价子集**上取均值，同时计入 `partial_periods` | B1 |
| 窗口 | **`--start-date`/`--end-date` 覆盖的全部历史**（`--years 0`，B4 起的默认）；`--years N` 截最后 N 年 | §5.3：12 期样本无功效 |
| 窗口 ≠ 面板 | **评估窗口**（demo 跑多久）与**物化面板窗口**（喂给模型的训练历史）是两件事，报告里分别记 | B4 `metadata.json` |
| 结束日 | 基线：`max(score_date)`（hs300）；模型：与 A 调仓日交集 | — |

**指标优先级**：主证据是 IC 与 top-K 超额，在**调仓日**网格上算（不是每个交易日 —— 20 日前瞻重叠，日频 2,591 个 IC 的**有效独立样本**只有 ~130，与调仓网格信息量相同，而调仓网格的点**不重叠**，普通 t 检验直接有效）；净值差只能作为**同名调仓日配对**的统计量，附 p 值与 95% CI（实现给出的是**配对 t**，不是原稿写的置换/bootstrap —— 调仓网格的点不重叠，普通 t 就成立，见 §5.6 第 1 点）。1 年窗口的净值差**不足以**支持任何「模型更好」的结论。

> ⚠️ **功效的单位是日历跨度，不是观测数**（订正前文的「7.5 万观测」）：250 日 × 300 名 = 7.5 万是**观测数不是独立观测数**，同一天的 300 个名字共享同一段行情；把有效样本读成 7.5 万会把每个 t 放大 √300 ≈ 17 倍。**只有日历跨度买功效** —— 见 §5.5 第 5 点（换调仓频率救不了）和 §5.6 第 3 点（IC 这把尺子实测有多灵敏）。

> ✅ **B5 已落地（2026-10-15）**：`summary.json["primary"]` 与逐日明细 `metrics_by_date.csv` 都在，结果见 §5.6。B4 时点这里写的「主指标没有实现」**已失效**。
>
> ⚠️ 一条**读法纪律**必须跟着主指标走：主指标问的是**与净值不同的问题**（整条截面的排序 vs 头部 10 名能不能变成钱），所以两者结论不一致时**不存在「谁错了」**，只存在「问的是哪个问题」。B4 时写在这里的「主指标比净值灵敏得多」是**计划书的推断，没有被实测兑现**（IC 的 MDE₈₀ = 0.0459 vs 实际效应 0.020–0.031）—— 细节与原因见 §5.6 第 3 点。
>
> ⚠️ 另外两处限定词必须一起读：**IC 不扣成本**（成本作用在组合换手上，加不到单票收益上，所以成本只进 TOP-K 超额族）；**「中性化」在这里是恒等变换** —— 池内 demean 是**每个日期一个常数**，而 IC 在**单个日期内**算，相关系数对加常数不变，故中性化 IC ≡ 原始 IC，**精确相等**（推论：demo 在原始收益上算的 IC 与平台在 `label_fwd_20d` 上算的 `baseline_ic` 可直接对拍）。真正的行业中性和市值中性**不在此列**，且 panel 的 `log_market_cap` 是 100% NaN，做不了。

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
| 输出目录 | `--out-dir`：`summary.json`、`equity_curve.csv`、`rebalances.csv`、**`metrics_by_date.csv`**（✅ B5 已交付，逐日主指标）、**`metadata.json`**（B4：provenance，且**最后写** ⇒ 它的 digest 覆盖其余四个） |
| 文档 | 本文件 ✅；RUNBOOK 一节「Demo 回测」**未写**（2026-10-15 核：三个 RUNBOOK 里都没有 demo 段落，§8 的 CLI 示例目前是本文件内唯一的运行说明） |
| 后续（可选） | `GET /api/model-training/demos/...` + 训练域 Tab（API/UI，非 MVP） |

### 2.1 产物字段（B5 后实况）

原稿把 provenance 放在 `summary.json` 里。B4 改成**两个文件**：`summary.json` 只装**这一次评估的结果**，`metadata.json` 只装**这次跑的环境与协议**（见 §5.3 第 1 点）。理由是 provenance 里含产物摘要，而"摘要自己"必须最后算 —— 混在一个文件里做不到。

`summary.json`（实测键位，冒烟产物 `/tmp/demo_smoke_host`）：

```
demo_id / status / years / top_n / rebalance_days / backfill_purge_days / score_column
model_id / source_job_id
source{kind, mongo_db, data_db, index_code, coverage_start, coverage_end}   ← 生效库名，不是 flag
model_source{kind, source_job_id, snapshot_id, parquet_path, model_id, model_kind,
             live_rows, live_dates, shadow_live_from}
model_overlap          ← 原稿的 backfill_validation：live↔backfill rank correlation
model_training{job_id, result_id, promotion, permutation_p, baseline_ic, folds{...}, note}
                       ← 源 job 自己的走查证据，逐字引用而非重算；无 job 时为 null（B5）
primary{grid, pool, join, tracks{baseline|model|control_random}, comparisons, control,
        power, cost, caveats}                                    ← B5 主指标，见 §5.6
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

- ✅ **`primary`（IC 那一组）已存在**（B5，2026-10-15）。B4 时点写的"不存在"已失效。`metrics_by_date.csv` 列：
  `trade_date, track, pool_basis, n_pool, pool_return, pool_return_wide, n_wide, n_scored, n_joined, n_priced, pricing, carried, ic_rank, ic_pearson, topk_mean, excess_gross, cost, excess_net` —— 一行一个 `(调仓日, 轨)`，池那几列每行重复（基准若中途变了，不可能被忽略）。**缺数写空单元格，不写 `None`**。`topk_mean` 从净值链路的 `gross_return` **抄**，不重算 ⇒ 与 `rebalances.csv` 逐位一致。
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

**窗口含义**：score 数据从 2015-01 起（§5.3 的池覆盖表），**行数**充分。**B4 起默认全历史**（`--years 0`）—— 2 年那档只是快速迭代取向，从来不是数据上限。但 **「数据够多」≠「功效够」**：全历史跑完（§5.5）测出本 demo 的最小可测效应是 **~24%/yr 的池超额**，约束来自**日历跨度**而非行数。

> **窗口深度不受 B 轨 backfill 成本约束**（实测）：每个调仓日一次 refit，13 个调仓日 + overlap 检查共 ~25 次 refit 端到端约 **10s**（4 vCPU），全历史 130 个调仓日 ⇒ **分钟量级**。真正的约束既不是算力也不是行数，是**日历跨度**（§5.5 第 5 点）。

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
| **B4** ✅ | 池参数化（`--index-code`）、窗口 `start/end`（`--years 0` = 全历史）、`metadata.json` provenance、与 `process_portfolio_research_jobs` 交叉校验 | 1.5d | 见 §5.3（参数化 + 池覆盖 + 功效）、§5.4（交叉校验）、**§5.5（全历史 129 期实跑）**——A 轨一致性到 SBT 层（§5.1）；两条链路**成本口径本就不同**，端到端数字不可能相等 |
| **B5a** ✅ | **补主指标**：`demo/primary.py` + `summary.json["primary"]` + `metrics_by_date.csv`；`evaluation/metrics.py` 的 `spearman_ic` 参数化为 `cross_sectional_ic(method=…)`（行为不变） | 1d | 见 §5.6 —— 逐调仓日 IC 与池超额都在；⚠️ **但不等于「功效更高」**：IC 的 MDE₈₀ = 0.0459 而实际效应 0.020–0.031，对 A 自己的水平净值的 t（1.62）甚至高于 IC 的 t（1.23）（§5.6 第 3 点） |
| **B5b** ✅ | `model_training` 组：把源 job 自己的走查判决（`promotion` / `permutation_p` / `baseline_ic` / 折汇总）引进 summary，逐字引用不重算 | 0.5d | 见 §5.6 第 5 点 —— ⚠️ 首次记录 **`promotion.passed = false`（`ic_below_baseline`）**：B4/B5 跑的这个模型是被自家门禁判否的 |
| **B5c** | `--parquet-resolution` 开关，并把生效模式记进 provenance（§5.5 的 canonical 覆盖坑；目前只能靠 `-e MODEL_SHADOW_PARQUET_RESOLUTION=job` 绕） | 0.5d | — |
| **B5d** | 决定 shadow 日更 cron（**是决策不是代码**；不挂 ⇒ live 段永远 1 天、§5.2 的 ρ 永远 n=1） | — | 见 §9 |

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

> **⚠️ 已被 §5.5 覆盖（2026-10-15 同日，全历史 129 期）**：同一对比在 129 期上是 **+0.1415pp/期，t=0.41，p=0.684**，削掉两端各 5 期后点估计 **+0.0071pp**。**本节这张表里 −10pp/+12pp 的跨度是 12 期窗口的噪声，不是 purge 的效应量**；保留本节作为「12 期能骗到什么程度」的标本，**不要**再引用它的量级。purge 的裁决见 §5.5 第 3 点 / §9。

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

> **实测回填（§5.5，2026-10-15）**：全历史实际拿到 **130 调仓日 / 129 完整期**（比上表的 ~142 少，因为窗口起点还要让出 81 日的 embargo）。**×3.4 的功效买回来了，但仍然不够**——A − 池 的 t 只到 1.58。本表只讲「扩窗口怎么改善功效」，**不要**把它读成「扩了窗口就能下结论」；真实上限见 §5.5 第 5 点。

**5. 同一 1 年窗口、同样 `purge=60`，只换 `--top-n`**（12 期，单位统一为「每期」）

| 配置 | A 每期 (t) | B 每期 (t) | **B−A** | 组合重叠 | R（gross t） |
|---|---|---|---|---|---|
| `--top-n 20` | +3.5532% (0.9358) | +4.2153% (1.155) | **+0.66 pp/期** | 53.8% | — |
| `--top-n 10`（主口径） | +1.6526% (0.4204) | +5.1367% (1.0957) | **+4.07 pp/期**（累计 +48.79pp） | 48.46% | 0.7477 ✅ |

换一个 N，A 掉一半多（+3.55 → +1.65）、B 反涨（+4.22 → +5.14），B−A 从 +0.66 变 +4.07 pp/期 —— 两边 t 都 < 1.2。**同一个窗口里，动旋钮的幅度大于数据本身的信号**，这就是为什么 B4 必须先扩窗口再谈结论。

> **全历史跑完之后（§5.5 第 7 点）**：129 期上 **A +1.2033 > B +0.8319 pp/期**，排名和这张 1 年表**相反**。所以本表的 +4.07pp 不是「N=10 更好」，是 12 期噪声；**`--top-n` 的效应同样没被这条数据分辨出来**。

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

### 5.5 B4 全历史结果（2026-10-15，`sha-00a4f45`）：首次在 129 期上做结论

**跑法**（两档只差一个旋钮）

| 项 | 值 |
|---|---|
| 面板 | `hs300_production_20150130_20261014.parquet`，snapshot `2ceba697-6c98-42b4-b61d-e515b32478ba`，**852,300 行 / 2,841 日 / 667 名**（`hs300_current.parquet` 只有 490 日，故意没覆盖，见 §9） |
| job | `434591a2-9dd7-476f-b00a-da2eb4c6c08d`，`ridge`，train 60 / test 20 / **purge 60**，`artifact_hash=d70fe691…` |
| 评估窗口 | **20160215 → 20261014**：2,591 交易日 / 130 调仓日 / **129 完整期** |
| 旋钮 | `--backfill-purge 60`（契约） vs `--backfill-purge 0`（live 配方） |
| 产物 | `apps/data/quant_ml/demo-fullhistory-p60/`、`…-p0/`（各含 `metadata.json` provenance） |

> **窗口起点 20160215 是在看结果之前定的**，理由不是挑好看的：面板首日 20150130，而 B 轨每个调仓日要退 **horizon 20 + purge 60 = 81 交易日**才够训练窗，故窗口起点必须 ≥ 面板第 250 个交易日附近；取 20160215（面板第 250 日）。这是**先定规则再看数**，不是事后择优。

> **必须显式拧的第二个旋钮**：`--source-job-id` 默认走 **canonical 覆盖**（`shadow/resolve.py:50` → `apply_canonical_parquet_resolution`），会把 job 的面板换成 `hs300_current.parquet`。历史回放必须 `docker compose run -e MODEL_SHADOW_PARQUET_RESOLUTION=job`（trainer 服务的 `environment:` 没列这个变量，不加 `-e` compose 不会转发）。**默认覆盖对 live 是对的、对 replay 是错的**——live 永远该训最新快照，而 replay 的整个意义是复现那次实验。

**1. 三轨结果**（单位统一为「每期」，即每个 20 日调仓期；pp = 百分点）

| | A（规则 60:40） | **B(`purge=60`)** | B(`purge=0`) | R（随机对照） |
|---|---|---|---|---|
| 每期 net | **+1.2033pp** | +0.8319pp | +0.6904pp | −0.0251pp |
| 每期 gross | **+1.3337pp** | +0.9780pp | +0.8373pp | +0.2351pp |
| 累计 net | **+166.18%** | +108.41% | +58.99% | −20.79% |
| max drawdown | −71.72% | −58.36% | −57.49% | −41.88% |
| 平均单边换手 | 0.4862 | 0.5438 | 0.5477 | 0.9677 |
| t（net） | 1.46 | 1.30 | 0.99 | −0.05 |
| **t（gross）** | **1.62** | **1.53** | **1.21** | **0.48** ✅ |

R 的 gross 是**池等权本身**（期望超额为 0），所以上表里「B 的 gross − R 的 gross」就是 B 相对池的 gross 超额。

**2. 配对检验**（n=129，按 `trade_date` 配对；`rebalances.csv` 逐期可取。**不要用 `summary.json` 的 `comparison.excess_return_b_minus_a`**——那是**累计**差，和每期差不是一回事）

| 对比（每期） | 差值 | t | p | 95% CI（pp/期） | 年化 CI |
|---|---|---|---|---|---|
| B(60) − A | −0.3714pp | −0.56 | 0.578 | [−1.691, +0.948] | [−20.5, +11.5]%/yr |
| B(0) − A | −0.5130pp | −0.88 | 0.382 | [−1.670, +0.644] | [−20.2, +7.8]%/yr |
| B(60) − R | +0.7429pp | +1.42 | 0.157 | [−0.291, +1.777] | [−3.5, +21.5]%/yr |
| **A − R** | **+1.0986pp** | **+1.58** | **0.116** | **[−0.276, +2.473]** | **[−3.3, +29.9]%/yr** |
| **B(60) − B(0)** | **+0.1415pp** | **+0.41** | **0.684** | **[−0.543, +0.824]** | **[−6.6, +10.0]%/yr** |

> ⚠️ **上面两行「− 池」的标签是错的，已改为「− R」（2026-10-15，B5 订正）。** 当时基准还没实现，用的是**对照轨 R 的 gross 收益**当池等权的代理 —— R 从池里随机抽 10 名，它的 gross 在期望上就是池等权。B5 补上按**当日截面**直接算的基准后，量到 A − 池 = **+1.0261pp（t=1.62, p=0.109）**，与代理值 +1.0986pp 只差 0.07pp：**代理有效，§5.5 的结论没被新口径推翻**（两个都验过，见 §5.6 第 2 点）。两者不等的原因是 R 自己那 10 名的抽样噪声**加进了差里**，所以代理版的 sd 更宽（7.89pp vs 7.213pp）—— 这正是下面第 5 点的 MDE 要跟着改的原因。

**3. purge 这一档：** 不是一个结论，是一个**零结果**，而且是被三重证据钉住的零结果——

- 点估计 +0.1415pp/期 在**削掉两端各 5 期后只剩 +0.0071pp**（119 期）。也就是说这个"优势"完全由 3–4 期撑着：最大两期是 `20260630` **+33.83pp** 和 `20260601` **−14.97pp**（同一段行情的两面）。
- 符号检验 **67/129**，binomial p=0.725 —— 连方向都不稳定。
- 95% CI 是 **[−0.543, +0.824] pp/期 ≈ [−6.6, +10.0]%/yr**：**任何大于 ~±10%/yr 的 purge 效应都可以排除**，但没有更细的分辨率。

⇒ §9 那条「purge 取 60 还是 0」**在数据上无法用收益差来裁决**。既然两者都无泄漏（`purge=0` 时训练窗末 = T−20、标签窗 [T−40, T−20]，与评估期 [T, T+20] 已有 20 日净间隔），就该按**其他理由**定：契约的 60 更保守且是冻结值，**保留 60**；但必须在契约里写清 **live 用的是 0，所以 Track B 量的是「保守变体」，不是 live 本体**。**不要因为 +0.14pp 去改这个旋钮**——那正是 §5.3 第 5 点和 §5.4 结尾警告的网格选择偏差。

**4. 内置一致性检查（顺带证明了管道是确定性的）**：两档之间 **`baseline` 与 `control_random` 的逐期收益逐位相同**（`max|p60−p0| = 0.000e+00`），只有 `model` 动了。说明 purge 旋钮**确实只喂 B 轨**，A/R 两轨与价格清洗、成本、日历全部可复现。另：`equity_curve.csv` 三列终值 − 1 与 `summary.json` 的 `track_*.total_return` **6/6 逐位相符**（`baseline`=A、`model`=B、`random`=R）。

**5. 功效：净值这把尺子只分辨得出「年化 ~22% 以上」的池超额**

A − R 的每期标准差是 **7.89pp**（n=129）⇒ se = 0.695pp；而**按当日截面直接算**的 A − 池标准差是 **7.213pp**（B5 实测）⇒ se = 0.635pp。后者是主指标的口径（基准是确定的，没有抽样噪声），所以主口径的 **MDE₈₀ = 2.8016 × 0.635 = 1.78pp/期**，年化 = `1.78 × 252/20` = **22.4%/yr**。（本文件早前写的 23.6%/yr、以及用 7.89pp 算出的 1.95pp/24.6%/yr，都是拿 R 当池代理时的数；B5 有了精确基准后一律用 1.78pp/22.4%。报数一律用**算出来的**那个。）

> ⚠️ **B4 时写在这里的话「IC 用满 300 名而非 10 名，MDE 小一个量级」没有兑现**，见 §5.6 第 3 点：同一批数据上 IC 的 MDE₈₀ 是 0.0459，而实际存在的效应只有 0.020–0.031；对 A 自己的水平，**净值的 t（1.62）甚至高于 IC 的 t（1.23）**。净值这把尺子粗是真的，但**换 IC 并没有换来预期的分辨率** —— 原因和结论都在 §5.6。

**换调仓频率救不了**——令持有期 h，则 sd ∝ √h、n ∝ 1/h，故 se = sd/√n **∝ h**：年化 MDE 与 h 无关。20 日一档算出来 22.4%/yr，5 日一档同样是 22.4%/yr。**期数不是自由度，日历跨度才是。**

| 想测出的池超额 | 需要的期数 | 需要的日历跨度（现有 10.7 年） |
|---|---|---|
| 2.0pp/期（25.2%/yr） | 102 | 8.1 年（**刚好够**） |
| 1.0pp/期（12.6%/yr） | 409 | **32.5 年** |
| 0.5pp/期（6.3%/yr） | 1,633 | 129.6 年 |

（表按主口径 sd = 7.213pp、`n = (2.8016 × 7.213 / 目标pp)²`、1 期 = 20 交易日算。）

**6. 本轮能说什么、不能说什么**

- ✅ **R 轨通过**（gross t=0.48，远低于 `CONTROL_T_MAX=3.0`）：harness 在 gross 层面**没有** look-ahead 或换手计费缺陷。这是三轨里唯一被证成的结论。
- ⚠️ **A 有正超额但未显著**：+1.0986pp/期 vs 池，t=1.58，p=0.116。方向稳定、量级合理（≈年化 13%），但**没到显著**。
- ⚠️ **B 看不出比 A 好**：点估计低 0.37pp/期，p=0.578，CI 宽到 [−1.69, +0.95]。**「B 不如 A」和「B 好于 A」都在这条数据的 CI 里**，唯一能说的是**现有 129 期不支持 B 优于 A**。
- ❌ **主指标补上了，但没有收窄「B vs A」的区间**（B5，同日，§5.6）：本节以上结论都建立在次要指标上，B5 是它们的检验。检验的结果是——**A 的池超额被两条路互相确认**（+1.0261pp vs 代理版 +1.0986pp），而 **B vs A 仍然判不出来**，且 IC 与净值给出的方向相反。**「B 好于 A」在这份数据里没有任何一条口径支持**。
- ❌ **`model_overlap` 为 `null`**：本次 `live_rows=0`（新 job 还没有 shadow 实盘分），重叠期校验**没跑**。该检查需要一段 live 段才成立（§9 'shadow cron' 那条）。
- ❌ **无 purged/embargo 之外的多重检验修正**：本轮只动了 1 个旋钮 ×2 档，没有网格；但把 `--top-n`、窗口、池一起扫就是网格，须按 §5.4 的警告处理。

**7. 与 §5.3 第 5 点的对照**：同一个 B−A，12 期窗口下能随 `--top-n` 在 +0.66 ~ +4.07pp/期间跳；129 期下 A 与 B 的 CI 宽度是 ±1.3pp/期。**扩窗口买到的是「CI 收窄」，不是「数字变好」**——这一轮 A 和 B 的排名甚至和 1 年窗口时相反（1 年窗口 B 是 +5.14 > A +1.65；全历史 A +1.20 > B +0.83）。**12 期窗口给出的方向本身就是噪声**，这正是先扩窗口再谈结论的理由。

**8. Track B 量的到底是「谁」——它测的是配方，不是 live 那个 artifact**

demo 的 B 轨**不复用 live 的模型文件**：它从 job 读**配方**（`model_kind` / `feature_names` / `model_params` / `label_col`，ridge 的 `alpha=1.0` 也在内），然后在每个调仓日**从零 refit**（§4）——这是 PIT 强制要求的，否则拿 2026-10 训好的系数去预测 2016 就是彻底的前视。所以 **129 期那张表是「这条配方在走一遍历史时会怎样」，不是「live 系统当时拿到了什么」。**

它与 live 实际那条路有**三处**不同，不是一处：

| # | 维度 | demo Track B | live（`model_shadow_profiles/hs300-20d-ridge`） | 能不能对齐 |
|---|---|---|---|---|
| 1 | **面板深度** | 全历史 2,841 日 | 490 日（`hs300_current.parquet`，即 job `b7618410` 的训练面板） | 能，但**对上就没窗口**（24 期，§5.3） |
| 2 | **禁运期** | 60 日 | 0 日 | 能，就是 §9 那条（已判：不可分辨） |
| 3 | **refit 节奏** | 每个调仓日重训 | 用**晋级的 artifact** 直接打分 | **不能**——PIT 下必须 refit；差距由 §5.2 的 ρ=0.87 界定 |

⇒ **第 3 条是结构性的，无法消除**；第 1 条是个真取舍：**「跟 live 对齐」和「有足够窗口下结论」是互斥的**，demo 选了后者（正确的选择，因为 24 期什么都测不出）。**引用 Track B 的数字时必须带上这三条**，尤其别把 129 期的 B 当成「live 的业绩预告」。

**9. 顺带核过：canonical 解析没被这次物化污染**（2026-10-15）。新物化的全历史面板在 `dataset_manifests` 里 `channel=None`，**不匹配** canonical 的兜底查询 `{index_code, channel: production_current}`，所以 `load_current_manifest` 仍稳定解析到 `hs300-current`（490 日）。**live shadow 的训练面板没被改动。**

---

### 5.6 B5 主指标结果（2026-10-15）：IC 补上了，但**没有**换来预期的分辨率

**跑法**（与 §5.5 同一套参数，产物 `apps/data/quant_ml/demo-fullhistory-p60-b5/`）

| 项 | 值 |
|---|---|
| 命令 | 同 §5.5，**加** `--source-job-id 434591a2-…`（读训练证据）与 `--out-dir …-b5`；新增产物 `metrics_by_date.csv` |
| 环境 | `-e MODEL_SHADOW_PARQUET_RESOLUTION=job`（必须显式，同 §5.5） |
| ⚠️ **必须挂工作树** | `-v /home/shuyolin/trading/quant-model-platform:/app -w /app` |

> 🔥 **最后那条是这次踩到的坑，值得单独记**：`quant-model-trainer` 服务**不挂仓库**（`quant-infrastructure/apps/docker-compose.yml` 里它的 `volumes:` 只有 `./logs/…` 和 `./data/quant_ml`），代码来自**镜像里那层 `/app`**。所以 `docker compose run … quant-model-trainer -m quant_model_platform.tools.…` 跑的是**镜像那个 sha 的代码**，不是工作树 —— 未提交的改动**完全不生效，而且不报错**：退出码 0、日志正常、产物齐全，只有内容悄悄退回旧版。
>
> **判据**（这次就是这个）：新写的 `metrics_by_date.csv` 没出现在产物目录，且 `summary.json` **字节数和上一次一模一样**（6511）—— 新增一个 JSON 组不可能不涨字节。冒烟验证：`--entrypoint python quant-model-trainer -c "import quant_model_platform.demo.primary as p; print(p.__file__)"`，要看到 `/app/...`。**B5 的第一次全历史重跑就是白跑的**，结论来自挂载后那一版。

**1. 三条口径订正落成实现**（把 §1 第 28 行的措辞兑现）

- **IC 不扣成本**：成本作用在组合换手上，加不到单票收益上 ⇒ 成本只进 TOP-K 超额族（`cost.applied_to = "topk_excess_net only"`）。
- **「中性化」是恒等变换**：池内 demean 是**每个日期一个常数**，IC 在**单个日期内**算，相关系数对加常数不变 ⇒ 中性化 IC ≡ 原始 IC，**精确相等**。已用测试钉住（将来若有人改成按行业/市值残差化，测试立刻红）。
- **网格 = 调仓日**：`grid.n_dates = 130`，且 `min_session_gap = 20 = horizon` ⇒ 点**不重叠**，普通配对 t 直接有效，不需要 Newey-West。IC 有 **129** 期，少的那一期是最后一期（前瞻窗未完成 ⇒ `pricing == "none"` ⇒ 出不了 IC，也不按 0 复合）。
- **基准 = 当日截面 ∩ 有报价**（`pool.basis = "day_cross_section"`，均值 **263.6** 名），**不是**窗口并集（**600.2** 名）。并集口径并列记录，实测 **+6.85pp/期 vs 当日截面 +0.31pp/期**，差 **+6.54pp/期**；而且这差价**不是常数**：

  | 年 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 并集 − 截面（pp/期） | +25.95 | +17.05 | +13.13 | +7.91 | +2.73 | +2.20 | +1.05 | +0.79 | +0.10 | +0.38 | −0.35 |

  逐年衰减到 0 的**形状**说明它不是「生存偏差」这类固定偏置，而是**成分股前视**：面板横跨 10 年成分，并集篮子里一直混着「后来才被纳入 hs300」的票，而纳入前的涨幅被算了进来 —— 一个当年**没有人能持有**的篮子。**拿并集当基准会把每个超额平移约 −6.5pp**。这是 B5 引入的最大陷阱，也是它只并列展示、不当基准的原因。
- ⚠️ **一处必须说明的窄化**：契约 §1 说的「当天全体有标签样本」，在 demo 里落成的是「**当天有分数的名字** ∩ 有报价」—— 分数是横截面的定义者，没有分数就没有排序可言，所以这是**窄化，不是等号**。平台自己的 `label_fwd_20d` demean 用的是 panel 的全体有标签样本（`baseline_ic` 的 `n_days = 2821` 就是这么来的）。两处因此**只能对拍、不会相等**，别把差值当 bug（§1 第 46 行说的「可直接对拍」指的就是这一点）。
- 一条如实记的裂缝：`EXIT_LIMIT_DELAY_DAYS = 5` 会让个别票的出场推迟最多 5 个交易日 ⇒ 少数标的的 20 日前瞻窗**轻微重叠**，上限约一期的 25%，二阶量；已写进 `grid.note`，不假装不存在。

**2. 三轨主指标**（n = 129 期；pp = 百分点/期）

| | rank-IC 均值 | sd | t | MDE₈₀ | Pearson IC 均值 | TOP-10 池超额 gross | t | MDE₈₀ |
|---|---|---|---|---|---|---|---|---|
| A 基线 | +0.0201 | 0.1861 | +1.23 | 0.0459 | **+0.0279** | **+1.0261pp** | **+1.62** | 1.7792pp |
| B 模型 | **+0.0310** | 0.1991 | +1.77 | 0.0491 | +0.0248 | +0.6704pp | +1.34 | 1.3995pp |
| R 对照 | −0.0021 | 0.0616 | −0.39 | 0.0152 | −0.0062 | −0.0725pp | −0.27 | 0.7504pp |

**R 的两个独立检验都通过**：t = −0.39；解析零假设（独立打分下，单日 rank-IC 的抽样 sd ≈ `1/√(m−1)` = 0.0579，D 天均值的 sd = 0.0051）下，实测 −0.0021 落在 3σ 内（`control.passed = true`）。这测的是**打分→标签这条 join**，与 §5.5 的 `control_r`（测账本与成本模型）是两件事，两个都留着。

**配对差（model − baseline，按 `trade_date` 配对，n = 129）**

| 口径 | 差值 | t | p | 95% CI | MDE₈₀ |
|---|---|---|---|---|---|
| rank-IC | **+0.0109** | +0.58 | 0.563 | [−0.0262, +0.0479] | 0.0525 |
| Pearson IC | **−0.0032** | −0.18 | 0.857 | [−0.0380, +0.0316] | 0.0493 |
| TOP-10 池超额 gross | **−0.36pp/期** | −0.53 | 0.594 | [−0.0167, +0.0096] | 1.866pp |

⇒ **三个差全部跨零，没有一条能判方向。**

**顺带**：整张表 9 个对比里**唯一跨出零的**是 `baseline_minus_control_random` 的 **Pearson IC**：+0.0341，t = +2.14，**p = 0.035**，CI [+0.0025, +0.0657] ⇒ **A 轨确实有可测的截面预测力**（与 §5.5 里 A 的池超额 +1.0261pp / t=1.62 同向，但这一条真的跨了零）。⚠️ 但 9 个对比里冒出 1 个 p=0.035，按族错误率看（≈ 0.32）**不构成发现** —— 它值得记一笔是因为**与 §5.5 的独立证据同向**，不是因为那个 p 值本身。

**3. 功效：IC 确实更细，但没有细到能判 B vs A —— 计划书的预期没有兑现**

- **净值这把尺子**：A − 池的 sd 7.213pp ⇒ MDE₈₀ = **1.78pp/期 = 22.4%/yr**（§5.5 第 5 点已按此订正）。
- **IC 这把尺子**：sd 0.1861 ⇒ MDE₈₀ = **0.0459**。而**数据里真正存在的效应是 0.0201（A）与 0.0310（B）** ⇒ 分别是 MDE 的 **44% 与 68%** —— **连 A 自己那条 IC 都不足以在本窗口达到 80% 功效**。IC 不是万能尺：10.7 年也只够看见 IC ≈ 0.046 那种量级的东西。
- **对 A 自己的水平，净值的 t（1.62）反而高于 IC 的 t（1.23）**（IR 0.142/期 vs 0.108/期）。⇒ 计划书里「IC 是唯一有功效的尺子」这句，**在这个窗口上没有兑现**。补主指标的价值在于**问对了问题**，不在于**测得更准**。
- 为什么会这样，有一个**可查的**观测：**R 的 IC sd（0.0616）几乎等于解析零假设（1/√(m−1) = 0.0579），而 A 的 IC sd 是它的 3 倍（0.1861）**。随机分数的 IC 波动就是纯抽样噪声；规则分数的 IC 波动主要来自**「打分与收益的关系本身在变」** —— 那一部分方差是**信号的时间变化**，不是样本不够，**换更细的尺子也消不掉**。
- **计划书里 t ≈ 2.8 的预期是「借来的效应量」**：它按源 job 的 `baseline_ic.mean_ic = 0.0493` 反推。demo 自己的基线 mean IC 实测 **+0.0201** —— 同样名义上的「规则分 IC」，换池、换网格、换窗口差 **2.5 倍**。这是 §5.5 第 8 点「Track B 测的是配方」的又一个读数：**效应量不可跨口径搬运**，那一次搬运正好把预期抬高了。
- 回到「B − A」：两把尺子的相对分辨率**接近**（IC 要 0.0525 才够 80% 功效、实测 0.0109 ⇒ 差 4.8 倍；净值要 1.87pp、实测 0.36pp ⇒ 差 5.2 倍）。**不是尺子选错了** —— 这份数据的日历跨度（10.7 年 / 129 个不重叠期）就是这个分辨率。

**4. 三个口径指向不同方向 —— 但三者可以自洽**

| 口径 | 问的问题 | model − baseline | 方向 |
|---|---|---|---|
| rank-IC（Spearman） | 整条截面的**排序** | +0.0109 | 模型更好 |
| Pearson IC | 排序 + **数值极值** | −0.0032 | 模型更差 |
| TOP-10 池超额 | **只吃最头部**（能变成钱的那部分） | −0.36pp/期 | 模型更差 |

三者按「对极值的敏感度」排成一条线，方向的变化因此**不是矛盾**，而是同一个故事：**模型把中段排得更好、把最头部（同时也是数值极值）排得更差**。

**必须一起读的一句话**：IC 在当天 ~263 名上算，策略只交易 TOP-10 —— **「排序更好」与「选出的 10 只更赚钱」是两个问题**，而只有后者能变成钱。既然三个差都跨零，这个「故事」是**读法**而不是**发现**；能确定的只有：**没有任何一条口径支持「B 优于 A」**。

**5. 源 job 的训练证据（`model_training` 组）—— 一件 B5 之前没记在任何地方的事**

- **`promotion = {passed: false, reason: "ic_below_baseline", model_mean_ic: 0.0234, baseline: 0.0493}`** ⇒ **B4/B5 跑的这个模型，是被自家门禁判否的。**
- `permutation_p = 0.0196`（置换检验通过）⇒ 判否的依据是**相对基线不够**，不是「没有信号」。
- 136 折归约：model mean_ic **0.0234** / sd 0.1814 / 正比例 0.537；baseline **0.0552** / sd 0.1637 / 正比例 0.654；差 **−0.0319**。**离散度才是重点** —— sd 0.18 是均值的 8 倍，短窗口点估计基本是噪声，这正是那份文档的整个论点。
- 一处**口径不对称**（已写进 `model_training.note`）：门禁拿「模型各折均值的均值」去比 `baseline_ic.mean_ic`（**逐日池化**的均值）。同一份数据的两种平均差 12%（`folds.baseline.mean_ic` **0.0552** vs `baseline_ic.mean_ic` **0.0493**）。判否方向不因此改变（0.0234 离两者都远），但**比大小前要先看是不是同一种平均**。
- `n_days_summed = 2701` 是**日频、重叠**的计数 ⇒ 不是样本量，且与 demo 的网格不同。`comparable_to_primary: false` 就是为这句话设的。

**6. 本轮能说什么**

- ✅ **R 轨在 IC 上也通过**（t = −0.39，且落在解析零假设 3σ 内）⇒ 打分→标签链路无泄漏，与 §5.5 的 gross 检验**互相独立**。
- ✅ **A 的池超额被两条路互相确认**：精确基准 **+1.0261pp**（t=1.62）vs 代理基准（A−R）**+1.0986pp**（t=1.58）⇒ §5.5 的结论**没被新口径推翻**。
- ✅ **A 有可测的截面预测力**（Pearson IC vs R：p = 0.035）—— 但按 9 个对比的多重性看**不构成发现**。
- ⚠️ **B vs A 仍然判不出来**，且三个口径方向不一致。「B 好于 A」在**任何**一条口径上都没有支持；能说的仍然只是 §5.5 那句「**不支持 B 优于 A**」。
- ❌ **B4 的模型是被门禁判否的**（`ic_below_baseline`）—— 引用 B 轨数字时必须带上这条，与 §5.5 第 8 点的三条口径差异并列。
- ❌ **主指标没有换来分辨率** —— 这是 B5 最该记住的一条。**后续任何加因子 / 换因子的工作，都不要假设「IC 一定比净值灵敏」**，要像 §5.5 第 5 点那样先算 MDE 再谈能不能测。

**7. 一处实现订正（写下来是因为它差点混过去）**：`tracks[*].pearson_ic` 曾经**三轨全为 null** —— 行里的列叫 `ic_pearson`，而汇总那块读的是 `pearson_ic`，键名对不上，`_clean` 把 NaN 变成了 `null`，读起来像「没数据」。**125 个测试全绿**，因为没有任何一个测试查过 summary 的这个块（`comparisons` 用的是对的那个键，所以配对差正常）。已修，并补了一条**通用**测试：summary 的每个族都逐一对回 CSV 列、重算均值与 `n_dates` —— 这类键名漂移以后再出现会立刻红。

---

## 6. 测试策略

| 层级 | 内容 |
|------|------|
| 单元 | 调仓日历；单日 TOP-N；缺失截面语义；末段完整期判定；**purge 断言**（✅ B3b：`test_default_purge_is_the_protocol_embargo` 钉住训练窗末 = T − horizon − purge，另有 `purge=0` = live 配方的对照）；重叠期 rank correlation（同序=1、反序=失败、名数不足不硬算） |
| 集成 | 合成小面板跑通 A/B/R 三轨；`--bundle` 离线夹具；`model_predictions` 读取按 `model_id`/`shadow` 过滤（假集合） |
| 对账 | A 轨与 `ScoreBacktester` 同参数逐日一致（B2）；B 轨 live 分 vs backfill rank correlation（B3b，§5.2） |
| **harness 自证** | **对照轨 R 相对池等权超额 ≈ 0**；可选泄漏探针（把 T 日分数换成 T+1 日分数，harness 应显著变好） |
| **主指标（B5）** | `tests/test_demo_primary.py` 20 例：单调分数 ⇒ IC **精确** ±1；**池内 demean 不移动任何 IC**（钉住 §1 的恒等式，若有人改成按行业残差化会立刻红）；池排除「在窗口并集里但不在当日截面里」的名字；薄截面（4 名）出行但无 IC；无报价日无 IC 也不当 0 收益；carried 日无打分事件故无 IC；`excess_net = excess_gross − cost` 精确；随机分数的 mean rank-IC 落在解析零假设 `1/√((m−1)·D)` 的 3σ 内；`effective_n == n_dates` 且 **`pooled_observations` 只数基线一条截面**（不是三条轨相加）；MDE 按 `2.8016·sd/√n`；**`json.dumps(..., allow_nan=False)` 不抛**（子串搜 NaN 会误命中说明文字）；空网格报"无"而不是 0；**每个汇总族逐一对回它读的 CSV 列并重算均值**（键名漂移会红，见 §5.6 第 7 点） |
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

**B5 已落地**：`summary.json` 新增 `primary` / `model_training` 两组，新增逐期明细 `metrics_by_date.csv`（长表，一行一个 `(trade_date, track)`）。

> ⚠️ **在容器里跑本节的命令时，必须挂工作树**：`-v /home/shuyolin/trading/quant-model-platform:/app -w /app`。`quant-model-trainer` 服务的 compose 定义**不挂仓库**，代码来自镜像里那层 `/app` ⇒ 不挂 = 静默跑**镜像那个 sha 的代码**（退出码 0、产物齐全、内容退回旧版）。判据与冒烟命令见 §5.6 的跑法表。

环境：本机 `MONGO_URI`（`finance` / `quant_data`）、`QUANT_ML_DB=quant_ml`、`QUANT_ML_PARQUET_ROOT=/data/quant_ml/parquet`。**无 S3、无远端依赖**。

---

## 9. 待拍板

- [x] ~~**B 轨口径**：backfill 用 `purge=60`（契约冻结值）还是 `purge=0`（= 实盘 shadow 的 refit 方式）？~~ → **2026-10-15 全历史（129 期）判定：数据裁不了这个，按非收益理由定，保留 60**（§5.5 第 3 点）。
  - 129 期上 B(60) − B(0) = **+0.1415pp/期，t=0.41，p=0.684，95% CI [−0.543, +0.824]pp/期 ≈ [−6.6, +10.0]%/yr**；削掉两端各 5 期后点估计只剩 **+0.0071pp**（119 期），符号检验 67/129（p=0.725）。
  - ⇒ **±10%/yr 以上的 purge 效应可以排除；更细的分辨率这条数据给不了。** 判据从「哪个收益高」换成「哪个更保守」：60 是冻结值且已无泄漏疑虑（`purge=0` 时训练窗末 = T−20，标签窗 [T−40,T−20] 与评估期 [T,T+20] 已有 20 日净间隔），**保留 60**。
  - ⚠️ 但契约必须同时写明：**live 走的是 0，所以 Track B 量的是「保守变体」，不等于 live 本体**。**不要拿 +0.14pp 去改这个旋钮**（§5.3 第 5 点 / §5.4 结尾的网格选择偏差）。
  - 12 期窗口时两档看着差 −0.55 与 +0.66pp/期（§5.2），**那是窗口噪声**，不是本旋钮的效应量。
- [x] ~~主口径 **N=10 / 调仓 20d** 是否认可~~ → **2026-10-15 拍板认可**。CLI 默认已从 20 改为 **10**（§5.3 第 1 点）；20 保留为敏感性档（§5.3 第 5 点：同一窗口两档 B−A 差 6 倍）。
- [x] ~~`demo_id` 是否改名~~ → 改为按参数派生；B4 起用**真实窗口边界**而非 `{years}y`（`--years 0` 之后年份标签会和数据对不上）。
- [x] ~~B 轨读数通道~~ → **本机自产**（K8s 不可访问；见 §1.1 / §3.1）
- [x] ~~窗口：默认 2 年还是拉长~~ → **2026-10-15 决定拉长到全历史**，`--years 0` 成为默认（§5.3 第 4 点：12 期 → ~142 期，t 功效 ×3.4）。成本不是约束（13 个调仓日实测端到端约 10s；全历史 ~142 次 refit ⇒ 分钟量级）。
  - ⚠️ **换池 = 换窗口**：`a500` 的分数只从 2024-09-30 起，`csi2000` 只从 2026-10-09 起 ⇒ 长窗口实际只能在 `hs300`/`csi500`/`csi1000` 上做（§5.3 第 2 点）。
- [ ] B3a 物化范围：目前只物化了 `hs300`（面板 `hs300_current.parquet`，490 日）。换 `csi500`/`csi1000` 跑长窗口**需要各自物化一份面板**（B 轨要面板才能 refit）。
- [ ] 本机 shadow **日更 cron** 要不要挂（不挂 ⇒ live 段永远是 1 天、§5.2 的 ρ 永远 n=1；挂 ⇒ 8–12 周后才有可用的 live 观察段）。**这也是本次 `model_overlap=null` 的原因**（§5.5 第 6 点）。**= B5d**，2026-10-15 未拍板。
- [ ] **B5c：`--parquet-resolution` 开关**（0.5d，纯代码，不必拍板但没人做）—— 现在历史回放必须靠 `docker compose run -e MODEL_SHADOW_PARQUET_RESOLUTION=job` 手动绕，而这个"必须手动绕"只能从 §5.5 那段文字知道，跑出来的 `metadata.json` **不记生效模式**。不补这条，一份产物事后无法自证它用的是 job 面板还是 canonical 面板。
- [ ] **要不要把 canonical `hs300_current.parquet`（490 日）换成全历史面板**（2026-10-15 新增，**会改 live 训练输入，需明确授权**）。
  - 现状：全历史面板已物化并登记（`2ceba697-…`，`channel=None`），**canonical 未被改动**（§5.5 第 9 点），live 仍训 490 日。
  - 这是 §5.5 第 8 点的第 1 条：换掉就等于让 live 也训全历史，**「demo 与 live 的面板深度不一致」这条差异会消失**。
  - ⚠️ 但**「面板该多深」本身是个未测的旋钮**——本轮只测了 purge，没测深度。按 §5.3 第 5 点 / §5.4 的教训，**不该在没测之前改 live 的行为**。若要做，先按 §5.5 的两档跑法加一档「490 日面板 + purge 60」对照（`--start-date` 起于 2024-10，只有 24 期，功效不足 ⇒ 大概率测不动，那就**维持现状不动**）。
