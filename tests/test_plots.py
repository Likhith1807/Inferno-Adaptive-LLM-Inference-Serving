"""Plot smoke tests: every figure renders in both themes without raising."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from inferno.analysis import plots
from .test_aggregate import _frame


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_result_figures_render(tmp_path, theme):
    df = _frame(0.8)
    plots.plot_pareto(df, "steady", tmp_path / "a.png", theme=theme)
    plots.plot_p99_by_pattern(df, tmp_path / "b.png", theme=theme)
    plots.plot_scaling([{"router": "mpi", "replicas": n, "throughput": 900 * n * 0.9, "ci": 20}
                        for n in (1, 2)] + [{"router": "http", "replicas": n, "throughput": 900 * n * 0.95,
                                              "ci": 20} for n in (1, 2)], tmp_path / "c.png", theme=theme)
    for f in "abc":
        assert (tmp_path / f"{f}.png").stat().st_size > 5000


def test_kernel_figures_render(tmp_path):
    names = ["cublas_fp16", "unfused_naive", "int4_v0_naive", "int4_v1_warp", "int4_v2_vec", "int4_v3_pad", "int4_v4_alg", "int4_v5_reuse", "int4_v6_lane16"]
    shape = lambda n: {"name": f"layer{n} (N={n},K=1024)", "N": n, "K": 1024, "fp16_bytes": 2 * n * 1024, "int4_bytes": n * 1024 // 2 + 4 * n * 1024 // 128, "kernels": [
        {"kernel": k, "us_median": 10.0 + i, "pct_peak": 20.0 + 10 * i, "gbps": 50.0, "us_p10": 1, "us_p90": 2,
         "max_rel_err": 1e-4, "correct": True} for i, k in enumerate(names)]}
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"device": "test", "peak_dram_gbps": 256.0, "shapes": [shape(1), shape(2)]}))
    plots.plot_kernel_speedup(p, tmp_path / "s.png")
    plots.plot_kernel_bandwidth(p, tmp_path / "b.png")
    assert (tmp_path / "s.png").exists() and (tmp_path / "b.png").exists()


def test_sweep_figure_renders(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for q, k in (("fp16", 0.5), ("int8", 0.8), ("int4", 1.0)):
        for bs in (1, 8, 32, 64):
            for rep in range(3):
                rows.append({"config": f"{q}-bs{bs}", "pattern": "steady", "rep": rep, "n_requests": 100,
                             "n_ok": 100 if bs > 1 else 60, "throughput_tok_per_s": k * 100 * bs ** 0.5 + rng.normal(0, 3),
                             "tokens_per_joule": k * 5 + bs * 0.05, "e2e_p50": 10 / (k * bs ** 0.3)})
    plots.plot_sweep(pd.DataFrame(rows), "steady", tmp_path / "sweep.png")
    assert (tmp_path / "sweep.png").stat().st_size > 5000
