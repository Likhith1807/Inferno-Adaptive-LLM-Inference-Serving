"""Contention analysis on synthetic data built inside the test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from inferno.experiments.report_contention import analyze, build


def _df(active_factor=2.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for pattern, base in (("rho0.3", 4.0), ("rho0.6", 6.0)):
        for rep in range(5):
            shared = rng.normal(0, 0.1)
            for cfg, f in (("single", 1.0), ("split-idle", 1.0), ("split-active", active_factor)):
                rows.append({"config": cfg, "pattern": pattern, "rep": rep, "n_errors": 0,
                             "e2e_p99": base * f + shared + rng.normal(0, 0.02),
                             "ttft_p50": 0.04, "tokens_per_joule": 10.0 / f ** 0.2,
                             "throughput_tok_per_s": 1000 / f ** 0.1})
    return pd.DataFrame(rows)


def test_ratios_recover_the_injected_effect():
    res = analyze(_df(2.0))
    comps = {(c["pattern"], c["comparison"], c["metric"]): c for c in res["comparisons"]}
    active = comps[("rho0.3", "split-active vs split-idle", "e2e_p99")]
    idle = comps[("rho0.3", "split-idle vs single", "e2e_p99")]
    assert active["ratio_a_over_b"] == pytest.approx(2.0, rel=0.05) and active["p_ttest"] < 0.01
    assert idle["ratio_a_over_b"] == pytest.approx(1.0, abs=0.02)


def test_build_writes_report_and_both_figures(tmp_path):
    _df().to_csv(tmp_path / "summary.csv", index=False)
    text = build(tmp_path).read_text(encoding="utf-8")
    assert "Paired comparisons" in text and "split-active vs split-idle" in text
    assert (tmp_path / "contention_light.png").exists() and (tmp_path / "contention_dark.png").exists()
