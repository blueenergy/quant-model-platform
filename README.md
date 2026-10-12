# quant-model-platform

旁路 CPU 模型训练平台：只读生产数据，写入独立 `quant_ml` 库与 S3/Parquet 快照，不改动 `stock_scores` / 交易链路。

## 文档

- 研究协议：[`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md)

## 仓库

- GitHub（private）：https://github.com/blueenergy/quant-model-platform

## 本地开发

```bash
git clone git@github.com:blueenergy/quant-model-platform.git
cd quant-model-platform
pip install -e ".[dev]"
python -m pytest -q
```

生产镜像 **`quant-model-trainer`** 由本仓库 CI 构建（`Dockerfile` → ghcr → ACR relay）。

```bash
docker build -t quant-model-trainer:local .
```

## Parquet 与 S3（K8s）

`MODEL_TRAINING_RUNTIME=external_k8s` 时，训练结束会将快照上传到  
`s3://<bucket>/<QUANT_ML_S3_PREFIX>/snapshots/<index_code>/<snapshot_id>.parquet`  
（S3 凭证可与 `PORTFOLIO_RESEARCH_S3_*` 共用）。后续 job 在 `params.parquet_path` 填该 `s3://` URI 时，worker 会下载到 `QUANT_ML_PARQUET_ROOT/cache/` 并复用本地缓存。

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
