"""Persist TrainedModel to S3 or local path (joblib)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from quant_model_platform.config import PlatformConfig
from quant_model_platform.storage.s3 import download_file, is_s3_uri, should_use_s3_parquet, upload_file
from quant_model_platform.training.models import TrainedModel

log = logging.getLogger(__name__)


def artifact_object_key(result_id: str, *, prefix: str) -> str:
    base = prefix.strip("/") or "quant_ml"
    return f"{base}/artifacts/{result_id}.joblib"


def artifact_uri_for_result(result_id: str, config: Optional[PlatformConfig] = None) -> str:
    from quant_model_platform.storage.s3 import _setting

    cfg = config or PlatformConfig.from_env()
    if should_use_s3_parquet(config=cfg):
        bucket = _setting("BUCKET")
        key = artifact_object_key(result_id, prefix=cfg.s3_prefix)
        return f"s3://{bucket}/{key}"
    root = Path(cfg.parquet_root) / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"{result_id}.joblib")


def save_trained_model(model: TrainedModel, path_or_uri: str) -> str:
    import joblib

    if is_s3_uri(path_or_uri):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            joblib.dump(model, tmp_path)
            upload_file(tmp_path, path_or_uri)
        finally:
            tmp_path.unlink(missing_ok=True)
        return path_or_uri
    path = Path(path_or_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("saved model artifact to %s", path)
    return str(path)


def load_trained_model(path_or_uri: str, *, cache_root: Path) -> TrainedModel:
    import joblib

    if is_s3_uri(path_or_uri):
        bucket_key = path_or_uri.replace("s3://", "").replace("/", "__")
        local = cache_root / "artifacts" / bucket_key
        if not local.is_file():
            download_file(path_or_uri, local)
        path = local
    else:
        path = Path(path_or_uri)
    if not path.is_file():
        raise FileNotFoundError(f"model artifact not found: {path_or_uri}")
    model = joblib.load(path)
    if not isinstance(model, TrainedModel):
        raise TypeError(f"expected TrainedModel, got {type(model)}")
    return model
