"""
Turn a finished run directory into the report artifacts: tables (Markdown + CSV), figures (light + dark), and
a machine-readable hypothesis summary. Everything is computed from summary.csv; nothing is hand-entered.

    python -m inferno.experiments.report results/<run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..analysis import plots
from ..analysis.aggregate import (ADAPTIVE_LABEL, ablation, best_fixed, ci_table, compare_to_best_fixed,
                                  load_summary, to_markdown_table)

_HEADLINE = ["e2e_p50", "e2e_p95", "e2e_p99", "ttft_p99", "throughput_tok_per_s", "goodput_req_per_s",
             "tokens_per_joule"]


def _fmt_ci(row) -> str:
    if pd.isna(row["ci_lo"]):
        return f"{row['mean']:.3g}"
    half = (row["ci_hi"] - row["ci_lo"]) / 2
    return f"{row['mean']:.3g} ± {half:.2g}"


def headline_table(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    tbl = ci_table(df[df["pattern"] == pattern], _HEADLINE)
    tbl["cell"] = tbl.apply(_fmt_ci, axis=1)
    wide = tbl.pivot(index="config", columns="metric", values="cell")
    n = tbl.groupby("config")["n"].max()
    wide = wide[[m for m in _HEADLINE if m in wide.columns]]
    wide.insert(0, "n", n)
    return wide.reset_index()


def build_report(run_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "report"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = load_summary(run_dir)
    hw_path = run_dir / "hardware.json"
    hw = json.loads(hw_path.read_text()) if hw_path.exists() else {}
    gpu = (hw.get("gpus") or [{}])[0]

    lines = [f"# Results: {run_dir.name}", "",
             f"Hardware: {gpu.get('name', 'unknown GPU')}, driver {gpu.get('driver_version', '?')}, "
             f"vLLM {hw.get('packages', {}).get('vllm', '?')}, commit {hw.get('git_commit', '?')}.",
             f"Repetitions per cell: {int(df.groupby(['config', 'pattern']).size().min())} "
             "(mean ± 95% CI, t-distribution).", ""]

    summary = {"run": run_dir.name, "hypothesis_h1": []}
    for pattern in sorted(df["pattern"].unique()):
        lines += [f"## Load pattern: {pattern}", "", to_markdown_table(headline_table(df, pattern)), ""]
        for theme in ("light", "dark"):
            plots.plot_pareto(df, pattern, out / "figures" / f"pareto_{pattern}_{theme}.png", theme=theme)
        bf = best_fixed(df, pattern)
        if bf:
            lines += [f"Strongest fixed baseline on p99 latency: **{bf}**.", ""]

    for theme in ("light", "dark"):
        plots.plot_p99_by_pattern(df, out / "figures" / f"p99_by_pattern_{theme}.png", theme=theme)

    is_sweep = ADAPTIVE_LABEL not in set(df["config"]) and df["config"].str.match(r"^(fp16|int8|int4)-bs\d+$").all()
    if is_sweep:
        for pattern in sorted(df["pattern"].unique()):
            for theme in ("light", "dark"):
                plots.plot_sweep(df, pattern, out / "figures" / f"sweep_{pattern}_{theme}.png", theme=theme)

    if ADAPTIVE_LABEL in set(df["config"]):
        cmp = compare_to_best_fixed(df)
        cmp.to_csv(out / "h1_adaptive_vs_best_fixed.csv", index=False)
        summary["hypothesis_h1"] = json.loads(cmp.to_json(orient="records"))
        show = cmp[["pattern", "best_fixed", "n_pairs", "treatment_mean", "fixed_mean", "pct_change",
                    "p_holm", "cohens_dz", "throughput_ratio", "verdict"]]
        lines += ["## H1: adaptive vs best fixed (paired by repetition, p99 latency)", "",
                  to_markdown_table(show), "",
                  "`pct_change` < 0 means adaptive has lower p99. p-values are Wilcoxon signed-rank, Holm-corrected "
                  "across load patterns; a result counts as significant only if the paired t-test agrees. "
                  "`throughput_ratio` is adaptive / best-fixed mean throughput.", ""]

        for pattern in sorted(df["pattern"].unique()):
            ab = ablation(df, pattern)
            ab.to_csv(out / f"ablation_{pattern}.csv", index=False)

    if "avg_power_w" in df.columns:
        eff = ci_table(df, ["tokens_per_joule", "avg_power_w"])
        eff.to_csv(out / "efficiency.csv", index=False)

    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    ci_table(df).to_csv(out / "all_metrics_ci.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    print(f"Wrote {build_report(args.run_dir, args.out) / 'report.md'}")


if __name__ == "__main__":
    main()
