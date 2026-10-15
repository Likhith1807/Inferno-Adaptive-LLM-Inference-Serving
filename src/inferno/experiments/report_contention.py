"""
Analysis for the GPU co-residency study (run_contention.py).

    python -m inferno.experiments.report_contention results/contention_<timestamp>

Answers, per load level, with paired tests (same seeded stream per repetition):
    split-idle   vs single        cost of halving the memory budget (no contention)
    split-active vs split-idle    cost of two processes actively sharing the GPU
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..analysis import plots
from ..analysis.aggregate import to_markdown_table
from ..analysis.stats import compute_ci, paired_significance_test

CONDITIONS = ["single", "split-idle", "split-active"]


def _paired(df: pd.DataFrame, pattern: str, a: str, b: str, metric: str) -> dict:
    sa = df[(df.pattern == pattern) & (df.config == a)].set_index("rep")[metric]
    sb = df[(df.pattern == pattern) & (df.config == b)].set_index("rep")[metric]
    reps = sorted(set(sa.index) & set(sb.index))
    t = paired_significance_test(sb.loc[reps].tolist(), sa.loc[reps].tolist())     # treatment = a, baseline = b
    return {"pattern": pattern, "comparison": f"{a} vs {b}", "metric": metric, "n_pairs": len(reps),
            "a_mean": float(sa.loc[reps].mean()), "b_mean": float(sb.loc[reps].mean()),
            "ratio_a_over_b": float(sa.loc[reps].mean() / sb.loc[reps].mean()),
            "p_ttest": t["t_test"]["p_value"], "p_wilcoxon": t["wilcoxon_test"]["p_value"]}


def analyze(df: pd.DataFrame) -> dict:
    cells = []
    for (pattern, config), g in df.groupby(["pattern", "config"]):
        row = {"pattern": pattern, "config": config, "n": len(g)}
        for m in ("e2e_p99", "ttft_p50", "throughput_tok_per_s", "tokens_per_joule"):
            vals = g[m].dropna().tolist()
            mean, lo, hi = compute_ci(vals) if len(vals) >= 2 else (vals[0], float("nan"), float("nan"))
            row[m], row[m + "_ci"] = mean, (hi - lo) / 2
        cells.append(row)
    comps = []
    for pattern in sorted(df.pattern.unique()):
        for a, b in (("split-idle", "single"), ("split-active", "split-idle")):
            for m in ("e2e_p99", "throughput_tok_per_s", "tokens_per_joule"):
                comps.append(_paired(df, pattern, a, b, m))
    return {"cells": cells, "comparisons": comps}


def build(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "summary.csv")
    res = analyze(df)
    cells = pd.DataFrame(res["cells"])
    cells["config"] = pd.Categorical(cells["config"], CONDITIONS, ordered=True)
    cells = cells.sort_values(["pattern", "config"])
    show = cells[["pattern", "config", "n", "e2e_p99", "e2e_p99_ci", "throughput_tok_per_s", "tokens_per_joule"]].round(2)
    comp = pd.DataFrame(res["comparisons"])
    lines = [f"# GPU co-residency study: {run_dir.name}", "",
             "Same total load, same seeded streams. `single` = one replica with the GPU to itself; `split-idle` = two "
             "resident replicas, all traffic to one; `split-active` = two replicas sharing the traffic.", "",
             "## Latency, throughput, efficiency (mean; CI half-width for p99)", "", to_markdown_table(show), "",
             "## Paired comparisons", "", to_markdown_table(comp.round(4)), "",
             "`ratio_a_over_b` above 1 means the first condition's value is larger (worse for p99, better for throughput)."]
    (run_dir / "contention_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "contention_summary.json").write_text(json.dumps(res, indent=2))
    for theme in ("light", "dark"):
        plots.plot_contention(res["cells"], run_dir / f"contention_{theme}.png", theme=theme)
    return run_dir / "contention_report.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    print(f"Wrote {build(ap.parse_args().run_dir)}")


if __name__ == "__main__":
    main()
