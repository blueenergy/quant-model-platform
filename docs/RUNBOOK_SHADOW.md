# Shadow 旁路运维（Phase 1）

Shadow 写入 `quant_ml.model_predictions`（`shadow: true`），**不**写 `stock_scores`。

## 配置过线模型

任选其一：

### A. 环境变量 `MODEL_SHADOW_PROFILES`（推荐 Cron）

JSON 数组，每个元素至少包含 `source_job_id`（训练 UI 里过线 job 的 `job_id`）：

```json
[
  {"profile_id": "hs300-ridge", "source_job_id": "<hs300-job-uuid>"},
  {"profile_id": "a500-ridge", "source_job_id": "<a500-job-uuid>"}
]
```

可选：`model_id`、`enabled: false`、`use_saved_artifact: true`（默认每日按同配方 **全样本 refit**）。

### B. Mongo `model_shadow_profiles`

```bash
python -m quant_model_platform.tools.register_shadow_profile \
  --profile-id hs300-ridge \
  --source-job-id <job-uuid>
```

未设置 `MODEL_SHADOW_PROFILES` 时，Cron `--once` 会跑所有 `enabled` 的 profile。

## 手动跑一次

```bash
export MONGO_URI=...
export QUANT_ML_PARQUET_BACKEND=s3
# ... S3 env 与 quant-api 一致
export MODEL_SHADOW_PROFILES='[{"source_job_id":"<uuid>"}]'
python -m quant_model_platform.workers.shadow_loop --once
```

开发合成数据：`python -m quant_model_platform.workers.shadow_loop --synthetic`

## K8s

```bash
cd quant-infrastructure/k8s/quant-finance-stack/jobs
./run-quant-model-shadow.sh --dry-run   # 检查镜像 tag
# 在 CronJob 或 Secret 中注入 MODEL_SHADOW_PROFILES 后：
./run-quant-model-shadow.sh
```

## Mongo 验收

```javascript
// 某日各 model 条数
db.model_predictions.aggregate([
  { $match: { shadow: true, trade_date: "20251010" } },
  { $group: { _id: "$model_id", n: { $sum: 1 } } },
]);

// 最近写入
db.model_predictions.find({ shadow: true }).sort({ created_at: -1 }).limit(5);
```

## 故障排查

| 现象 | 处理 |
|------|------|
| `no shadow profiles configured` | 设置 `MODEL_SHADOW_PROFILES` 或注册 `model_shadow_profiles` |
| `unknown source_job_id` | 确认 job 在 `model_training_jobs` |
| S3 下载失败 | 检查 `QUANT_ML_S3_*` / `PORTFOLIO_RESEARCH_S3_*` |
| `n_predictions: 0` | Parquet 无最新 `trade_date` 行；重跑物化 Job 延长 `end_date` |

## 与训练制品

训练完成会在 `model_training_results.artifact_uri` 保存全样本模型（joblib）。Shadow 默认 **refit**（快照扩展后分数与训练配方一致）；设 `use_saved_artifact: true` 可改为加载制品。
