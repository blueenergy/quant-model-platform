# ML 研究协议（v0.1）

本协议锁定标签、PIT、时间切分、成本与模型晋级门槛；实现见 `quant_model_platform/protocol.py`。

## 标签

- **主标签**：20 交易日持有期、**股票池内市值加权中性**后的超额收益（相对池内等权或市值基准，由 `label_neutralize` 配置）。
- **成交价**：信号日 T 收盘后可知信息；**T+1 开盘**成交（`execution_price=next_open`），与 `ScoreBacktester` 默认一致。
- **前瞻窗口**：标签使用未来 20 个交易日收益；训练特征仅使用 T 日及之前 PIT 数据。

## PIT 与成分

- 指数成分、财报字段以**公告日/生效日**为准，禁止未来成分与未公告财报。
- 数据集 manifest 记录 `source_revision`（Mongo oplog 或快照时间）与 `pit_policy` 版本。

## 时间切分与 Purge

- 评估采用 **purged walk-forward**：训练窗与测试窗之间插入 **purge ≥ 60 交易日**，避免标签泄漏。
- 历史 2020–2026 仅作滚动验证，不宣称未触碰样本外。

## 成本

- 默认单边滑点 **10 bps**；换手按 Top-K 组合相对上一期变化估算。
- 晋级报告必须给出**扣成本后**的中性化 IC / Top-K 模拟收益。

## 模型晋级（Shadow 前）

训练 job 需通过：

1. 多个 walk-forward 折上，扣成本中性化 IC 均值 **> 0** 且不低于规则基线。
2. 至少 **两个股票池**（如 CSI1000、HS300）同向为正。
3. FDR 校正后显著因子/特征比例不依赖单一年份；置换检验 p-value 阈值 0.05。

## Shadow（8–12 周）

- 仅写 `quant_ml.model_predictions` 与制品元数据；不写 `stock_scores`。
- 监控覆盖率、分数漂移、模拟收益；失败可删除 Deployment 完整回滚。

## 可选接入组合研究

- 仅当 Shadow 与 walk-forward 门禁均通过后，启用 `MODEL_SCORE_ADAPTER_ENABLED=1` 加载模型分；默认关闭。
