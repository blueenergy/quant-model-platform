import json
from pathlib import Path

from quant_model_platform.demo.portfolio import compare_tracks, one_way_turnover
from quant_model_platform.tools.demo_hs300_20d_top20 import main


def test_turnover_counts_replaced_names():
    assert one_way_turnover([], ["A", "B"]) == 1.0
    assert one_way_turnover(["A", "B"], ["A", "C"]) == 0.5


def test_model_beats_baseline_when_forwards_say_so():
    dates = ["20250101", "20250121"]
    scores_a = {
        "20250101": [{"ts_code": "AAA", "score": 2}, {"ts_code": "BBB", "score": 1}],
        "20250121": [{"ts_code": "AAA", "score": 2}, {"ts_code": "BBB", "score": 1}],
    }
    scores_b = {
        "20250101": [{"ts_code": "CCC", "score": 2}, {"ts_code": "DDD", "score": 1}],
        "20250121": [{"ts_code": "CCC", "score": 2}, {"ts_code": "DDD", "score": 1}],
    }
    forwards = {
        "20250101": {"AAA": 0.0, "BBB": 0.0, "CCC": 0.10, "DDD": 0.10},
        "20250121": {"AAA": 0.0, "BBB": 0.0, "CCC": 0.10, "DDD": 0.10},
    }
    report = compare_tracks(dates, scores_a, scores_b, forwards, top_n=2, hold_days=20)
    assert report["track_b"]["total_return"] > report["track_a"]["total_return"]
    assert report["comparison"]["avg_overlap_ratio"] == 0.0
    assert report["track_a"]["max_drawdown"] <= 0.0


def test_cli_writes_summary(tmp_path: Path):
    bundle = {
        "trading_days": ["20250101", "20250121"],
        "scores_a": {
            "20250101": [{"ts_code": "AAA", "score": 1.0}],
            "20250121": [{"ts_code": "AAA", "score": 1.0}],
        },
        "scores_b": {
            "20250101": [{"ts_code": "BBB", "score": 1.0}],
            "20250121": [{"ts_code": "BBB", "score": 1.0}],
        },
        "forward_returns": {
            "20250101": {"AAA": 0.02, "BBB": 0.05},
            "20250121": {"AAA": -0.01, "BBB": 0.03},
        },
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    out = tmp_path / "out"
    code = main(["--bundle", str(path), "--top-n", "1", "--rebalance-days", "1", "--out-dir", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "equity"
    assert summary["rebalance_count"] == 2
    assert "total_return" in summary["track_a"]
    assert (out / "equity_curve.csv").is_file()
