# Real data on aipoc K8s (stock_scores → Parquet → train)

## Prerequisites

- `quant-model-trainer` image includes production dataset code (rebuild + bump tag).
- Pod env: `MONGO_URI`, `MONGO_DB` (usually `finance`), `DATA_DB_NAME=quant_data`, S3 vars (see trainer Deployment).
- `stock_scores` rows tagged with `index_codes` for your universe (scorer must have run).

## Step 1 — Materialize snapshot (recommended)

Edit dates in `quant-infrastructure/k8s/quant-finance-stack/jobs/quant-ml-materialize-snapshot.yaml`, then:

```bash
K8S_NAMESPACE=aipoc
kubectl -n "$K8S_NAMESPACE" apply -f quant-infrastructure/k8s/quant-finance-stack/jobs/quant-ml-materialize-snapshot.yaml
kubectl -n "$K8S_NAMESPACE" logs job/quant-ml-materialize-snapshot -f
```

Stdout prints `snapshot_id` and `s3://...` `parquet_path`. Manifest is inserted into `quant_ml.dataset_manifests` when `--register-mongo` is set.

Or exec on the running trainer:

```bash
kubectl -n aipoc exec deploy/quant-model-trainer -- \
  python -m quant_model_platform.tools.materialize_snapshot \
  --index-code csi1000 --start-date 20230101 --end-date 20251231 --register-mongo
```

## Step 2 — Training job

**A. Reuse S3 snapshot** (fast reruns):

POST `/api/model-training/jobs` with `parquet_path`, `snapshot_id`, and walk-forward windows (`train_days` / `test_days` / `purge_days`).

**B. On-the-fly materialize in worker** (heavy):

```json
{
  "name": "csi1000 production",
  "index_code": "csi1000",
  "model_kind": "ridge",
  "data_source": "production_scores",
  "start_date": "20230101",
  "end_date": "20251231",
  "train_days": 60,
  "test_days": 20,
  "purge_days": 60,
  "universe_count": 2
}
```

Worker reads `finance.stock_scores` + `quant_data.volume_price`, uploads Parquet to S3, then trains.

## Step 3 — Promotion gates

- Need **≥3** valid walk-forward folds → calendar must cover `train+purge+test` many times over.
- Need **≥2** universes in `universe_count` (run hs300 + csi1000 jobs or extend pipeline).
- IC vs baseline + permutation test still apply.

## MVP feature coverage

Production v0 uses **score dimensions** mapped to the registry (`growth` / `cycle` / `value` / `fundamental→quality`). Price-derived factors (`momentum_*`, fundamentals) are NaN until a later materializer version.
