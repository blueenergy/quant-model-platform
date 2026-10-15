import argparse
import json
from pathlib import Path

from quant_model_platform.demo.provenance import (
    build_provenance,
    file_digest,
    output_digests,
    protocol_block,
    requested_args,
    write_provenance,
)
from quant_model_platform.protocol import LABEL_HORIZON_DAYS, MAX_PURGE_DAYS


def _args(**overrides):
    base = {
        "top_n": 10,
        "rebalance_days": 20,
        "years": 0,
        "backfill_purge": MAX_PURGE_DAYS,
        "index_code": "hs300",
        "out_dir": None,
        "model_id": "",
        "from_mongo": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _report(args=None, **kwargs):
    payload = {
        "trading_days": ["20250101", "20250121"],
        "scores_a": {"20250101": [{"ts_code": "AAA", "score": 1.0}]},
        "scores_b": {"20250101": [{"ts_code": "AAA", "score": 1.0}, {"ts_code": "BBB", "score": 0.5}]},
        "forward_returns": {"20250101": {"AAA": 0.01, "BBB": -0.01}},
        "model_source": {"kind": "job", "source_job_id": "job-1"},
    }
    payload.update(kwargs.pop("payload", {}))
    return build_provenance(
        args=args or _args(),
        index_code="hs300",
        window=["20250101", "20250121"],
        signals=["20250101"],
        payload=payload,
        model_source=payload.get("model_source"),
        **kwargs,
    )


def test_protocol_constants_are_recorded_not_restated_in_prose():
    block = protocol_block(MAX_PURGE_DAYS)
    assert block["label_horizon_days"] == LABEL_HORIZON_DAYS
    assert block["max_purge_days"] == MAX_PURGE_DAYS
    assert block["backfill_purge_days"] == MAX_PURGE_DAYS
    assert block["label_neutralize"] == "demean"


def test_window_and_counts_come_from_the_run():
    block = _report()
    assert block["window"]["start"] == "20250101"
    assert block["window"]["end"] == "20250121"
    assert block["window"]["rebalance_count"] == 1
    assert block["data"]["scored_days"]["track_b"] == 1
    assert block["data"]["scored_names"]["track_b"] == 2
    assert block["model"]["source_job_id"] == "job-1"


def test_an_overridden_purge_is_recorded_as_the_run_used_it():
    """The sensitivity runs are told apart by this field alone."""
    assert _report(args=_args(backfill_purge=0))["protocol"]["backfill_purge_days"] == 0


def test_requested_args_drops_empty_defaults_only():
    kept = requested_args(_args())
    assert kept["top_n"] == 10
    assert kept["backfill_purge"] == MAX_PURGE_DAYS
    assert "model_id" not in kept  # empty string
    assert "out_dir" not in kept  # None


def test_digest_changes_when_the_file_does(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    first = file_digest(path)
    path.write_text('{"a": 2}', encoding="utf-8")
    second = file_digest(path)
    assert first["sha256"] != second["sha256"]
    assert first["bytes"] > 0
    assert second["name"] == "summary.json"


def test_provenance_covers_the_other_outputs_and_not_itself(tmp_path: Path):
    (tmp_path / "summary.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "equity_curve.csv").write_text("period,nav\n0,1\n", encoding="utf-8")
    block = _report(out_dir=tmp_path)
    write_provenance(tmp_path, block)

    written = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in written["outputs"]}
    assert names == {"summary.json", "equity_curve.csv"}
    on_disk = file_digest(tmp_path / "summary.json")["sha256"]
    assert {e["name"]: e["sha256"] for e in written["outputs"]}["summary.json"] == on_disk


def test_outputs_are_listed_in_a_stable_order(tmp_path: Path):
    for name in ("b.txt", "a.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    assert [entry["name"] for entry in output_digests(tmp_path)] == ["a.txt", "b.txt", "c.txt"]
