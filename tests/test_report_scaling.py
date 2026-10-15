"""Scaling analysis on a synthetic scaling.csv constructed inside the test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from inferno.experiments.report_scaling import analyze, build


def _df(mpi_penalty=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n, scale in ((1, 1.0), (2, 1.6)):
        for rep in range(6):
            shared = rng.normal(0, 20)
            for router in ("http", "mpi"):
                thr = 2000 * scale * (1 - (mpi_penalty if router == "mpi" else 0)) + shared + rng.normal(0, 2)
                row = {"router": router, "replicas": n, "rep": rep, "throughput_tok_per_s": thr,
                       "e2e_p99": 10 / scale + rng.normal(0, 0.1), "n_errors": 0}
                if router == "mpi":
                    row.update(mpi_dispatch_us_p50=250.0, mpi_dispatch_us_p99=900.0,
                               mpi_result_us_p50=40.0, mpi_result_us_p99=200.0)
                rows.append(row)
    return pd.DataFrame(rows)


def test_efficiency_and_pairing():
    res = analyze(_df())
    eff = {(c["router"], c["replicas"]): c["efficiency_vs_1x"] for c in res["cells"]}
    assert eff[("http", 1)] == pytest.approx(1.0)
    assert eff[("http", 2)] == pytest.approx(0.8, abs=0.02)         # 1.6x throughput from 2x replicas
    thr = [r for r in res["mpi_vs_http"] if r["metric"] == "throughput_tok_per_s"]
    assert all(abs(r["pct_diff_mpi_vs_http"]) < 1 for r in thr)      # no injected penalty -> no difference
    assert res["overhead"][0]["mpi_dispatch_us_p50"] == 250.0


def test_injected_mpi_penalty_is_detected():
    res = analyze(_df(mpi_penalty=0.10))
    thr = [r for r in res["mpi_vs_http"] if r["metric"] == "throughput_tok_per_s"]
    assert all(r["pct_diff_mpi_vs_http"] == pytest.approx(-10, abs=1.5) for r in thr)
    assert all(r["p_wilcoxon"] < 0.05 for r in thr)


def test_build_writes_report_and_figures(tmp_path):
    _df().to_csv(tmp_path / "scaling.csv", index=False)
    report = build(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "MPI vs HTTP router" in text and "Measured MPI hop overhead" in text
    assert (tmp_path / "scaling_light.png").exists() and (tmp_path / "scaling_summary.json").exists()
