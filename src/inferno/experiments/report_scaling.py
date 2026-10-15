"""
Analysis for the replica-scaling experiment (H2): scaling efficiency, MPI vs HTTP routing, per-hop MPI overhead.

    python -m inferno.experiments.report_scaling results/scaling_<timestamp>

Outputs (next to scaling.csv): scaling_report.md, scaling.png (+ dark), scaling_summary.json.
Everything is computed from scaling.csv; MPI and HTTP runs are paired by repetition (same seeded stream).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..analysis import plots
from ..analysis.aggregate import to_markdown_table
from ..analysis.stats import compute_ci, paired_significance_test


def _ci(vals) -> tuple[float, float]:
    v = [x for x in vals if not pd.isna(x)]
    if len(v) >= 2:
        m, lo, hi = compute_ci(v)
        return m, (hi - lo) / 2
    return (v[0] if v else float("nan")), 0.0


def analyze(df: pd.DataFrame) -> dict:
    out = {"cells": [], "mpi_vs_http": [], "efficiency": [], "overhead": []}
    for (router, n), g in df.groupby(["router", "replicas"]):
        thr, thr_ci = _ci(g["throughput_tok_per_s"])
        p99, p99_ci = _ci(g["e2e_p99"])
        out["cells"].append({"router": router, "replicas": int(n), "n": len(g), "throughput": thr,
                             "throughput_ci": thr_ci, "p99": p99, "p99_ci": p99_ci,
                             "errors": int(g["n_errors"].sum())})
    base = {c["router"]: c["throughput"] for c in out["cells"] if c["replicas"] == 1}
    for c in out["cells"]:
        if c["router"] in base:
            c["efficiency_vs_1x"] = c["throughput"] / (base[c["router"]] * c["replicas"])
    for n in sorted(df["replicas"].unique()):
        a = df[(df["replicas"] == n) & (df["router"] == "mpi")].set_index("rep")
        b = df[(df["replicas"] == n) & (df["router"] == "http")].set_index("rep")
        reps = sorted(set(a.index) & set(b.index))
        if len(reps) < 3:
            continue
        for metric in ("throughput_tok_per_s", "e2e_p99"):
            t = paired_significance_test(b.loc[reps, metric].tolist(), a.loc[reps, metric].tolist())
            out["mpi_vs_http"].append({
                "replicas": int(n), "metric": metric, "n_pairs": len(reps),
                "mpi_mean": float(a.loc[reps, metric].mean()), "http_mean": float(b.loc[reps, metric].mean()),
                "pct_diff_mpi_vs_http": float(100 * t["mean_difference"] / b.loc[reps, metric].mean()),
                "p_ttest": t["t_test"]["p_value"], "p_wilcoxon": t["wilcoxon_test"]["p_value"]})
    mpi = df[df["router"] == "mpi"]
    for n, g in mpi.groupby("replicas"):
        row = {"replicas": int(n)}
        for col in ("mpi_dispatch_us_p50", "mpi_dispatch_us_p99", "mpi_result_us_p50", "mpi_result_us_p99"):
            if col in g:
                row[col] = float(g[col].mean())
        out["overhead"].append(row)
    return out


def build(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "scaling.csv")
    res = analyze(df)
    cells = pd.DataFrame(res["cells"]).sort_values(["replicas", "router"])
    lines = [f"# Replica scaling: {run_dir.name}", "",
             "MPI and HTTP routers replay identical seeded, saturating request streams (paired by repetition).", "",
             "## Aggregate throughput and p99 (mean, ± 95% CI half-width)", "",
             to_markdown_table(cells.assign(
                 throughput=lambda d: d.throughput.round(0), throughput_ci=lambda d: d.throughput_ci.round(0),
                 p99=lambda d: d.p99.round(2), p99_ci=lambda d: d.p99_ci.round(2)))]
    if res["mpi_vs_http"]:
        lines += ["", "## MPI vs HTTP router (paired by repetition)", "",
                  to_markdown_table(pd.DataFrame(res["mpi_vs_http"])), "",
                  "`pct_diff_mpi_vs_http` > 0 means the MPI router's value is larger (higher throughput is better; "
                  "higher p99 is worse). A difference is meaningful only if both tests agree and the effect exceeds the CI."]
    if res["overhead"]:
        lines += ["", "## Measured MPI hop overhead inside the real router (microseconds, mean over runs)", "",
                  to_markdown_table(pd.DataFrame(res["overhead"])), "",
                  "`dispatch` = dispatcher isend to worker seeing the job (includes the worker's 200 us poll interval); "
                  "`result` = worker isend to dispatcher seeing the result. Compare with request latencies of seconds."]
    (run_dir / "scaling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "scaling_summary.json").write_text(json.dumps(res, indent=2))
    fig_rows = [{"router": c["router"], "replicas": c["replicas"], "throughput": c["throughput"],
                 "ci": c["throughput_ci"]} for c in res["cells"]]
    for theme in ("light", "dark"):
        plots.plot_scaling(fig_rows, run_dir / f"scaling_{theme}.png", theme=theme)
    return run_dir / "scaling_report.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    print(f"Wrote {build(ap.parse_args().run_dir)}")


if __name__ == "__main__":
    main()
