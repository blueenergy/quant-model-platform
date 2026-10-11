from pathlib import Path

from quant_model_platform.training.pipeline import run_training_job


def test_training_pipeline_smoke(tmp_path: Path):
    result = run_training_job(
        {
            "index_code": "csi1000",
            "model_kind": "ridge",
            "train_days": 8,
            "test_days": 4,
            "purge_days": 2,
            "universe_count": 2,
            "n_perm": 20,
        },
        parquet_root=tmp_path,
    )
    assert result["result_id"]
    assert "promotion" in result
    assert "artifact_hash" in result
