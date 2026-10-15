# Shadow 旁路运维（Phase 1 / 1.5）

与成长+周期策略对比的 1 年 demo 规格见 [DEMO_HS300_20D_TOP20.md](DEMO_HS300_20D_TOP20.md)。

Shadow 写入 `quant_ml.model_predictions`（`shadow: true`），**不**写 `stock_scores`。

## 观察期启动条件（Phase 1.5）

旁路要产生 **按交易日前进** 的样本外序列，需同时满足：

1. **quant-scorer** 已写入当日 `stock_scores`（工作日 ~19:00）。
2. **canonical 物化**：hs300、a500 各 **一个** S3 对象 `.../snapshots/{index}/current.parquet`，Mongo `dataset_manifests` 中 `snapshot_id={index}-current`、`channel=production_current`。
3. **日更 pipeline**（工作日 **20:00** 上海）：物化 `end_date latest` → `shadow_loop --once`。
4. **验收**：每个 `model_id` 的最新 `trade_date` 与 manifest 的 `end_date` 一致，且随交易日 **单调前进**。

若 `trade_date` 长期停在同一日（如 `20251231`），说明仍在用冻结训练快照，**不计入** 8–12 周观察期。

## Canonical 快照（每指数一份 current）

```bash
python -m quant_model_platform.tools.materialize_snapshot \
  --index-code hs300 --start-date 20230101 --end-date latest \
  --canonical --register-mongo
# 同上 a500
```

- S3：`s3://<bucket>/quant_ml/snapshots/hs300/current.parquet`（覆盖写）
- `snapshot_id`：`hs300-current` / `a500-current`
- 训练 job 历史 uuid 快照保留，Shadow **默认**读 current（`MODEL_SHADOW_PARQUET_RESOLUTION=canonical`）

## 配置过线模型

### A. `MODEL_SHADOW_PROFILES`（Cron）

```json
[
  {"profile_id": "hs300-ridge", "source_job_id": "<hs300-job-uuid>"},
  {"profile_id": "a500-ridge", "source_job_id": "<a500-job-uuid>"}
]
```

仍用 `source_job_id` 绑定配方；**数据路径**由 canonical manifest 覆盖。

### B. Mongo `model_shadow_profiles`

```bash
python -m quant_model_platform.tools.register_shadow_profile \
  --profile-id hs300-ridge --source-job-id <job-uuid>
```

## K8s 日更 pipeline

CronJob 定义在 **quant-infrastructure** 的 `k8s/quant-finance-stack/base-workers/quant-ml-daily-shadow-pipeline-cronjob.yaml`（勿在 `jobs/` 下重复一份 YAML）。与日更时间表（data-engine → scorer 19:00 → ML 20:00）见 `k8s/quant-finance-stack/jobs/README.md`。

```bash
cd quant-infrastructure/k8s/quant-finance-stack/jobs
./run-daily-ml-shadow-pipeline.sh --apply-cron   # CronJob 20:00 + 挂起旧 shadow-only Cron
./run-daily-ml-shadow-pipeline.sh --run-once     # 手动跑一轮
```

Secret：`quant-model-shadow-profiles`（`MODEL_SHADOW_PROFILES`）。

仅 Shadow、不物化（调试）：`./run-quant-model-shadow.sh --run-once`（需已 current 物化）。

## Mongo / kubectl 验收（无 mongosh）

```bash
kubectl -n aipoc exec deploy/quant-model-trainer -- python -c "
from quant_model_platform.config import PlatformConfig
from quant_model_platform.db import get_ml_collections
from quant_model_platform.shadow.dataset import load_current_manifest
cfg = PlatformConfig.from_env()
cols = get_ml_collections(config=cfg)
for ix in ('hs300', 'a500'):
    m = load_current_manifest(cols, ix)
    print(ix, 'manifest end_date', m and m.get('end_date'))
preds = cols['model_predictions']
for mid in ('promo-hs300-', 'promo-a500-'):
    pipe = [
        {'\$match': {'model_id': {'\$regex': f'^{mid}'}, 'shadow': True}},
        {'\$group': {'_id': '\$trade_date', 'n': {'\$sum': 1}}},
        {'\$sort': {'_id': -1}},
        {'\$limit': 5},
    ]
    print(mid, list(preds.aggregate(pipe)))
"
```

清理开发期合成分（可选）：

```javascript
db.model_predictions.deleteMany({ shadow: true, model_id: "shadow-csi1000" })
```

## 故障排查

| 现象 | 处理 |
|------|------|
| `no canonical dataset_manifests` | 跑 `--canonical --register-mongo` |
| `end_date ... behind latest stock_scores` | scorer 未跑或物化失败；查 pipeline Job 日志 |
| `no shadow profiles configured` | `MODEL_SHADOW_PROFILES` / `model_shadow_profiles` |
| S3 失败 | `QUANT_ML_S3_*` / `PORTFOLIO_RESEARCH_S3_*` |

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL_SHADOW_PARQUET_RESOLUTION` | `canonical` | 设 `job` 则用训练 job 内固定 parquet（测试/复现） |
| `MODEL_SHADOW_MAX_STALE_CALENDAR_DAYS` | `3` | manifest `end_date` 落后 `stock_scores` 超过则 fail |

## 与训练制品

训练 `artifact_uri` 为全样本 joblib。Shadow 默认 **refit**；`use_saved_artifact: true` 可加载制品。
