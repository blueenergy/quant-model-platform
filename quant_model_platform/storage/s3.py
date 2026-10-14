"""S3-backed Parquet snapshots (K8s: survive pod restarts)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from quant_model_platform.config import PlatformConfig

logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _setting(suffix: str) -> str:
    """QUANT_ML_S3_* first, then PORTFOLIO_RESEARCH_S3_* (shared cluster secret)."""
    v = _env(f"QUANT_ML_S3_{suffix}")
    if v:
        return v
    return _env(f"PORTFOLIO_RESEARCH_S3_{suffix}")


def s3_env_complete() -> bool:
    return bool(
        _setting("ENDPOINT")
        and _setting("BUCKET")
        and _setting("ACCESS_KEY")
        and _setting("SECRET_KEY")
    )


def should_use_s3_parquet(config: Optional[PlatformConfig] = None) -> bool:
    cfg = config or PlatformConfig.from_env()
    override = _env("QUANT_ML_PARQUET_BACKEND").lower()
    if override == "local":
        return False
    if override == "s3":
        if not s3_env_complete():
            raise RuntimeError("QUANT_ML_PARQUET_BACKEND=s3 but S3 env is incomplete")
        return True
    runtime = (cfg.runtime or _env("MODEL_TRAINING_RUNTIME") or "local").lower()
    if runtime == "external_k8s":
        if not s3_env_complete():
            raise RuntimeError("MODEL_TRAINING_RUNTIME=external_k8s requires S3 for durable Parquet")
        return True
    return False


def _s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=_setting("ENDPOINT"),
        aws_access_key_id=_setting("ACCESS_KEY"),
        aws_secret_access_key=_setting("SECRET_KEY"),
        region_name=_setting("REGION") or "us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def canonical_snapshot_id(index_code: str) -> str:
    return f"{index_code}-current"


def snapshot_object_key(index_code: str, snapshot_id: str, *, prefix: str) -> str:
    base = prefix.strip("/") or "quant_ml"
    return f"{base}/snapshots/{index_code}/{snapshot_id}.parquet"


def canonical_snapshot_object_key(index_code: str, *, prefix: str) -> str:
    base = prefix.strip("/") or "quant_ml"
    return f"{base}/snapshots/{index_code}/current.parquet"


def s3_uri_for_canonical_snapshot(index_code: str, config: Optional[PlatformConfig] = None) -> str:
    cfg = config or PlatformConfig.from_env()
    bucket = _setting("BUCKET")
    key = canonical_snapshot_object_key(index_code, prefix=cfg.s3_prefix)
    return f"s3://{bucket}/{key}"


def s3_uri_for_snapshot(index_code: str, snapshot_id: str, config: Optional[PlatformConfig] = None) -> str:
    cfg = config or PlatformConfig.from_env()
    bucket = _setting("BUCKET")
    key = snapshot_object_key(index_code, snapshot_id, prefix=cfg.s3_prefix)
    return f"s3://{bucket}/{key}"


def is_s3_uri(path: str) -> bool:
    return str(path or "").startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"invalid s3 uri: {uri}")
    key = parsed.path.lstrip("/")
    return parsed.netloc, key


def _cache_path_for_key(cache_root: Path, bucket: str, key: str) -> Path:
    safe = key.replace("/", "__")
    return cache_root / bucket / safe


def upload_file(local_path: Path, s3_uri: str) -> str:
    bucket, key = _parse_s3_uri(s3_uri)
    _s3_client().upload_file(str(local_path), bucket, key)
    logger.info("uploaded parquet to %s", s3_uri)
    return s3_uri


def download_file(s3_uri: str, local_path: Path) -> Path:
    bucket, key = _parse_s3_uri(s3_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _s3_client().download_file(bucket, key, str(local_path))
    logger.info("downloaded parquet from %s to %s", s3_uri, local_path)
    return local_path


def publish_local_parquet(
    local_path: Path,
    *,
    index_code: str,
    snapshot_id: str,
    config: Optional[PlatformConfig] = None,
) -> str:
    """Upload local Parquet when S3 is enabled; return canonical path (s3 uri or local)."""
    if not should_use_s3_parquet(config):
        return str(local_path)
    uri = s3_uri_for_snapshot(index_code, snapshot_id, config=config)
    upload_file(local_path, uri)
    return uri


def publish_canonical_local_parquet(
    local_path: Path,
    *,
    index_code: str,
    config: Optional[PlatformConfig] = None,
) -> str:
    """Overwrite the per-index current.parquet object when S3 is enabled."""
    if not should_use_s3_parquet(config):
        canonical_local = local_path.parent / f"{index_code}_current.parquet"
        if canonical_local.resolve() != local_path.resolve():
            import shutil

            shutil.copy2(local_path, canonical_local)
        return str(canonical_local)
    uri = s3_uri_for_canonical_snapshot(index_code, config=config)
    upload_file(local_path, uri)
    return uri


def ensure_local_parquet(path_or_uri: str, *, cache_root: Path) -> Path:
    """Resolve training path: local path as-is, s3:// downloads into cache (reused if present)."""
    if not is_s3_uri(path_or_uri):
        return Path(path_or_uri)
    bucket, key = _parse_s3_uri(path_or_uri)
    cached = _cache_path_for_key(cache_root, bucket, key)
    if cached.is_file():
        return cached
    return download_file(path_or_uri, cached)
