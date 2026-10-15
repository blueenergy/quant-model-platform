"""What a demo run read, with which recipe, under which protocol constants.

A number without this is not reproducible: the same flags against a moved
`stock_scores`, a rebuilt panel or a different image give a different answer and
nothing in `summary.json` would say which of the three changed
(docs/DEMO_HS300_20D_TOP20.md section 7).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from quant_model_platform.demo.portfolio import (
    CONTROL_SEED,
    CONTROL_T_MAX,
    COST_PER_SIDE,
    ROUND_TRIP_COST_RATE,
    SELF_COMPUTED_SCORE,
)
from quant_model_platform.protocol import (
    DEFAULT_EXECUTION_PRICE,
    DEFAULT_SLIPPAGE_BPS,
    LABEL_HORIZON_DAYS,
    MAX_PURGE_DAYS,
)
from quant_model_platform.timeutil import to_cst, utcnow

#: Where a caller can hand in the image that produced the run. Inside the built
#: image there is no ``.git``, and the OCI labels that name the revision
#: (``org.opencontainers.image.revision``) are not visible from the container
#: itself -- ``docker inspect`` has to be joined on from outside.
IMAGE_TAG_VARS = ("MODEL_IMAGE_TAG", "QUANT_MODEL_TRAINER_IMAGE_TAG")


def file_digest(path: Path) -> Dict[str, Any]:
    """Size and sha256, so a later run can be told apart from this one."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"name": path.name, "bytes": size, "sha256": digest.hexdigest()}


def output_digests(out_dir: Path) -> List[Dict[str, Any]]:
    return [file_digest(p) for p in sorted(out_dir.glob("*")) if p.is_file()]


def git_revision(root: Optional[Path] = None) -> Optional[str]:
    """The checkout's HEAD, or None when running from the baked image."""
    base = str(root or Path(__file__).resolve().parents[2])
    try:
        out = subprocess.run(
            ["git", "-C", base, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value if out.returncode == 0 and value else None


def image_block() -> Dict[str, Any]:
    tag = next((os.getenv(name) for name in IMAGE_TAG_VARS if os.getenv(name)), None)
    return {
        "image_tag": tag,
        "git_revision": git_revision(),
        "note": (
            "image_tag comes from MODEL_IMAGE_TAG; git_revision is None inside the built image "
            "because it carries no .git. Both also live on the image as OCI labels "
            "(org.opencontainers.image.revision) -- read them with docker inspect."
        ),
    }


def protocol_block(backfill_purge: int) -> Dict[str, Any]:
    """The frozen constants a run is read against, so a reader need not trust the prose."""
    return {
        "label_horizon_days": LABEL_HORIZON_DAYS,
        "max_purge_days": MAX_PURGE_DAYS,
        "backfill_purge_days": backfill_purge,
        "execution_price": DEFAULT_EXECUTION_PRICE,
        "label_neutralize": "demean",
        "slippage_bps": DEFAULT_SLIPPAGE_BPS,
        "cost_per_side_bps": round(COST_PER_SIDE * 10_000, 2),
        "round_trip_bps": round(ROUND_TRIP_COST_RATE * 10_000, 2),
        "control_seed": CONTROL_SEED,
        "control_t_max": CONTROL_T_MAX,
        "self_computed_score": SELF_COMPUTED_SCORE,
    }


def requested_args(args: Any) -> Dict[str, Any]:
    """The resolved CLI namespace, minus defaults that carry no information."""
    out: Dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if value is None or value is False or value == "":
            continue
        if isinstance(value, Path):
            value = str(value)
        out[key] = value
    return out


def build_provenance(
    *,
    args: Any,
    index_code: str,
    window: Sequence[str],
    signals: Sequence[str],
    source: Optional[Dict[str, Any]] = None,
    model_source: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Everything needed to re-read or re-run this report, in one file."""
    data = payload or {}
    forward = data.get("forward_returns") or {}
    scores_a = data.get("scores_a") or {}
    scores_b = data.get("scores_b") or {}
    now = utcnow()
    block: Dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "generated_at_cst": to_cst(now).isoformat(),
        "argv": list(sys.argv),
        "requested": requested_args(args),
        "image": image_block(),
        "protocol": protocol_block(int(getattr(args, "backfill_purge", MAX_PURGE_DAYS))),
        "window": {
            "start": str(window[0])[:8] if window else None,
            "end": str(window[-1])[:8] if window else None,
            "trading_days": len(window),
            "rebalance_dates": [str(d)[:8] for d in signals],
            "rebalance_count": len(signals),
        },
        "data": {
            "source": source,
            "scored_days": {
                "track_a": len(scores_a),
                "track_b": len(scores_b),
                "forward_returns": len(forward),
            },
            "scored_names": {
                "track_a": sum(len(v or []) for v in scores_a.values()),
                "track_b": sum(len(v or []) for v in scores_b.values()),
            },
        },
        "model": model_source,
        "out_dir": str(out_dir) if out_dir else None,
    }
    if out_dir and out_dir.is_dir():
        block["outputs"] = output_digests(out_dir)
    return block


def write_provenance(out_dir: Path, block: Dict[str, Any]) -> Path:
    """Write ``metadata.json`` last, so its digest list covers the other files."""
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(block, indent=2, default=str) + "\n", encoding="utf-8")
    return path
