"""Kernel-run aggregation: median of run medians, spread, and correctness AND-ed across runs."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("agg", ROOT / "scripts" / "aggregate_kernel_runs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(us_a, us_b, correct_b=True):
    def k(name, us, ok=True):
        return {"kernel": name, "us_median": us, "us_p10": us * 0.95, "us_p90": us * 1.1,
                "gbps": 1000.0 / us * 10, "pct_peak": 10.0, "max_rel_err": 1e-4, "correct": ok}
    return {"device": "T", "peak_dram_gbps": 256.0, "l2_mb": 32.0, "group_size": 128,
            "shapes": [{"name": "s (N=1,K=1)", "N": 1, "K": 1, "int4_bytes": 1, "fp16_bytes": 4,
                        "kernels": [k("a", us_a), k("b", us_b, correct_b)]}]}


def test_median_of_run_medians_and_spread(tmp_path):
    import json
    for i, (a, b) in enumerate([(10.0, 20.0), (12.0, 20.0), (11.0, 20.0)]):
        (tmp_path / f"run_{i}.json").write_text(json.dumps(_run(a, b)))
    out = _mod().aggregate(sorted(tmp_path.glob("run_*.json")))
    a = out["shapes"][0]["kernels"][0]
    assert a["us_median"] == 11.0 and a["run_min_us"] == 10.0 and a["run_max_us"] == 12.0
    assert a["run_spread_pct"] == pytest.approx(100 * 2.0 / 11.0)
    assert a["n_runs"] == 3 and out["n_runs"] == 3
    b = out["shapes"][0]["kernels"][1]
    assert b["run_spread_pct"] == 0.0


def test_incorrect_in_any_run_marks_incorrect(tmp_path):
    import json
    (tmp_path / "run_1.json").write_text(json.dumps(_run(10.0, 20.0, True)))
    (tmp_path / "run_2.json").write_text(json.dumps(_run(10.0, 20.0, False)))
    out = _mod().aggregate(sorted(tmp_path.glob("run_*.json")))
    assert out["shapes"][0]["kernels"][1]["correct"] is False


def test_runs_within_2pct_separates_one_outlier_from_general_instability(tmp_path):
    import json
    # four runs agree, one outlier: the spread is large but 4 of 5 lie within 2% of the median
    for i, a in enumerate([10.0, 10.05, 9.98, 10.1, 15.0]):
        (tmp_path / f"run_{i}.json").write_text(json.dumps(_run(a, 20.0)))
    out = _mod().aggregate(sorted(tmp_path.glob("run_*.json")))
    k = out["shapes"][0]["kernels"][0]
    assert k["run_spread_pct"] > 40 and k["runs_within_2pct"] == 4
