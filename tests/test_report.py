"""End-to-end report generation from a synthetic summary.csv (constructed inside the test)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

from inferno.experiments.report import build_report

from .test_aggregate import _frame


def _write_run(tmp_path, shift):
    df = _frame(shift)
    df["e2e_p50"] = df["e2e_p99"] / 2
    df["e2e_p95"] = df["e2e_p99"] * 0.8
    df["ttft_p99"] = 0.3
    df["goodput_req_per_s"] = 3.0
    df["tokens_per_joule"] = 10.0
    df["avg_power_w"] = 40.0
    df.to_csv(tmp_path / "summary.csv", index=False)
    (tmp_path / "hardware.json").write_text(json.dumps({"gpus": [{"name": "TestGPU", "driver_version": "1"}],
                                                        "packages": {"vllm": "x"}, "git_commit": "abc"}))
    return tmp_path


def test_report_contains_tables_figures_and_h1_verdict(tmp_path):
    out = build_report(_write_run(tmp_path, 0.7))
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "H1: adaptive vs best fixed" in text and "adaptive better" in text
    assert "TestGPU" in text and "±" in text
    assert (out / "figures" / "pareto_steady_light.png").exists()
    assert (out / "figures" / "p99_by_pattern_dark.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert {r["pattern"] for r in summary["hypothesis_h1"]} == {"steady", "bursty"}


def test_report_states_null_result_plainly(tmp_path):
    out = build_report(_write_run(tmp_path, 1.0))
    assert "no significant difference" in (out / "report.md").read_text(encoding="utf-8")
