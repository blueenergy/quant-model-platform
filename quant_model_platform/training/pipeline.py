"""End-to-end training job: dataset -> walk-forward -> gates."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quant_model_platform.config import PlatformConfig
from quant_model_platform.dataset.features import registry_names
from quant_model_platform.training.artifacts import artifact_uri_for_result, save_trained_model
from quant_model_platform.training.fit import fit_full_sample_model
from quant_model_platform.training.panel_loader import load_training_panel
from quant_model_platform.evaluation.gates import evaluate_fold, passes_promotion_gates
from quant_model_platform.evaluation.metrics import permutation_test_ic, spearman_ic, summarize_ic
from quant_model_platform.evaluation.walk_forward import purged_walk_forward_folds, split_frame_by_dates
from quant_model_platform.queue import clean_for_mongo
from quant_model_platform.training.models import predict_model, train_model


def _artifact_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_training_job(
    params: Dict[str, Any],
    *,
    parquet_root: Path,
    config: Optional[PlatformConfig] = None,
) -> Dict[str, Any]:
    """Execute one training job document (synthetic or parquet snapshot)."""
    cfg = config or PlatformConfig.from_env()
    if config is None:
        cfg = PlatformConfig(
            mongo_uri=cfg.mongo_uri,
            quant_ml_db=cfg.quant_ml_db,
            mongo_db=cfg.mongo_db,
            data_db_name=cfg.data_db_name,
            parquet_root=str(parquet_root),
            s3_bucket=cfg.s3_bucket,
            s3_prefix=cfg.s3_prefix,
            runtime=cfg.runtime,
            worker_pool=cfg.worker_pool,
        )
    model_kind = str(params.get("model_kind", "ridge"))
    label_col = str(params.get("label_col", "label_fwd_20d"))
    panel, snapshot_id, parquet_path = load_training_panel(params, parquet_root=parquet_root, config=cfg)

    feature_names = list(params.get("feature_names") or registry_names())
    dates = sorted(panel["trade_date"].unique().tolist())
    folds = purged_walk_forward_folds(
        dates,
        train_days=int(params.get("train_days", 12)),
        test_days=int(params.get("test_days", 5)),
        purge_days=int(params.get("purge_days", 3)),
    )
    fold_reports: List[Dict[str, Any]] = []
    models_trained = []
    for fold in folds:
        train = split_frame_by_dates(panel, fold.train_dates)
        test = split_frame_by_dates(panel, fold.test_dates)
        if train[label_col].notna().sum() < 10 or test[label_col].notna().sum() < 5:
            continue
        baseline = train_model("rule_baseline", train, feature_names, label_col)
        model = train_model(model_kind, train, feature_names, label_col, params=params.get("model_params"))
        test = test.copy()
        test["pred"] = predict_model(test, model)
        test["baseline_pred"] = predict_model(test, baseline)
        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "baseline": evaluate_fold(test, "baseline_pred", label_col),
                "model": evaluate_fold(test, "pred", label_col),
            }
        )
        models_trained.append(model.kind)

    # Full-sample baseline vs model for promotion summary
    full_baseline = train_model("rule_baseline", panel.dropna(subset=[label_col]), feature_names, label_col)
    full_model = fit_full_sample_model(panel, params, label_col=label_col, feature_names=feature_names)
    scored = panel.dropna(subset=[label_col]).copy()
    scored["pred"] = predict_model(scored, full_model)
    scored["baseline_pred"] = predict_model(scored, full_baseline)
    baseline_ic = summarize_ic(spearman_ic(scored, "baseline_pred", label_col))
    perm_p = permutation_test_ic(scored, "pred", label_col, n_perm=int(params.get("n_perm", 50)))
    promotion = passes_promotion_gates(
        [f["model"] for f in fold_reports],
        baseline_mean_ic=float(baseline_ic.get("mean_ic") or 0.0),
        universe_count=int(params.get("universe_count", 1)),
        permutation_p=perm_p,
    )

    result_id = str(uuid.uuid4())
    artifact_uri = save_trained_model(full_model, artifact_uri_for_result(result_id, config=cfg))
    artifact_hash = _artifact_hash(
        {
            "model_kind": model_kind,
            "snapshot_id": snapshot_id,
            "feature_names": feature_names,
            "fold_count": len(fold_reports),
        }
    )
    return clean_for_mongo(
        {
            "result_id": result_id,
            "snapshot_id": snapshot_id,
            "parquet_path": parquet_path,
            "model_kind": model_kind,
            "fold_reports": fold_reports,
            "promotion": promotion,
            "permutation_p": perm_p,
            "baseline_ic": baseline_ic,
            "artifact_hash": artifact_hash,
            "artifact_uri": artifact_uri,
            "code_version": params.get("code_version", "0.1.0"),
        }
    )
