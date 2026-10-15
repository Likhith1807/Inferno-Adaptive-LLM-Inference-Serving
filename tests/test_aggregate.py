"""Aggregation + H1 comparison logic, on synthetic data constructed inside the test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from inferno.analysis.aggregate import (best_fixed, ci_table, compare_to_best_fixed, holm,
                                        to_markdown_table)


def _frame(adaptive_shift: float, reps: int = 8, seed: int = 0) -> pd.DataFrame:
    """Two fixed configs + adaptive; per-rep noise is shared across configs (paired design)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pattern, base in (("steady", 2.0), ("bursty", 5.0)):
        for rep in range(reps):
            shared = rng.normal(0, 0.15)     # same request stream => shared per-rep effect
            for config, p99, thr in (("fp16-bs8", base * 1.5, 900.0), ("int4-bs32", base, 1500.0),
                                     ("adaptive", base * adaptive_shift, 1500.0)):
                rows.append({"config": config, "pattern": pattern, "rep": rep, "n_errors": 0,
                             "e2e_p99": p99 + shared + rng.normal(0, 0.02),
                             "throughput_tok_per_s": thr + rng.normal(0, 5)})
    return pd.DataFrame(rows)


def test_best_fixed_picks_lowest_metric_and_ignores_errors():
    df = _frame(0.8)
    assert best_fixed(df, "steady") == "int4-bs32"
    df.loc[(df.config == "int4-bs32") & (df.rep == 0), "n_errors"] = 3   # disqualify
    assert best_fixed(df, "steady") == "fp16-bs8"


def test_clear_win_is_reported_as_win():
    out = compare_to_best_fixed(_frame(0.7))
    assert set(out["pattern"]) == {"steady", "bursty"}
    assert (out["pct_change"] < -20).all()
    assert out["verdict"].str.startswith("adaptive better").all()


def test_no_difference_is_reported_honestly():
    out = compare_to_best_fixed(_frame(1.0))
    assert out["verdict"].str.startswith("no significant difference").all()


def test_regression_is_reported_as_worse():
    out = compare_to_best_fixed(_frame(1.4))
    assert out["verdict"].str.startswith("adaptive WORSE").all()


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and adj[2] == 0.06 and adj[1] == 0.06
    assert all(0 <= a <= 1 for a in adj)


def test_ci_table_shape_and_markdown():
    tbl = ci_table(_frame(0.8), metrics=["e2e_p99"])
    assert {"mean", "ci_lo", "ci_hi", "n"} <= set(tbl.columns)
    assert (tbl["ci_lo"] < tbl["mean"]).all() and (tbl["mean"] < tbl["ci_hi"]).all()
    assert to_markdown_table(tbl.head(2)).count("\n") == 3
