# quant-model-platform

旁路 CPU 模型训练平台：只读生产数据，写入独立 `quant_ml` 库与 S3/Parquet 快照，不改动 `stock_scores` / 交易链路。

## 文档

- 研究协议：[`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md)

## 本地开发

```bash
cd quant-model-platform
pip install -e ".[dev]"
python -m pytest -q
```

在 `stock-scoring-system` 镜像中构建前，从 monorepo 同步源码：

```bash
cd stock-scoring-system && ./scripts/copy-quant-model-platform.sh
```

## 运行 worker（需 Mongo `quant_ml` 与可选 Parquet 路径）

```bash
export MONGO_URI=...
export QUANT_ML_DB=quant_ml
python -m quant_model_platform.workers.run_loop --worker-id dev-1
```

Shadow 日批预测：

```bash
python -m quant_model_platform.workers.shadow_loop --worker-id shadow-1
```
