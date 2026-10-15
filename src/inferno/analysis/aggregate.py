"""
Turn raw run output (summary.csv) into the numbers the report cites.

Design rules (docs/methodology.md):
  * every headline number is mean +/- 95% CI over repetitions, never a single run;
  * adaptive-vs-best-fixed is a *paired* comparison: repetition k of every config
    replays the identical seeded request stream, so runs pair by (pattern, rep);
  * "best fixed" is chosen per load pattern on the *same metric being tested*,
    which is the strongest baseline available and makes the comparison hard to
    win by accident. It is chosen only among configs with zero errors;
  * p-values across the load patterns are Holm-corrected;
  * a null or negative result is reported as such (see verdict()).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from .stats import compute_ci, paired_significance_test

ADAPTIVE_LABEL = "adaptive"
KEY_METRICS = [
    "e2e_p50", "e2e_p95", "e2e_p99", "ttft_p99", "tpot_p99",
    "throughput_tok_per_s", "goodput_req_per_s", "slo_attainment",
    "avg_power_w", "tokens_per_joule", "peak_mem_mib",
]


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load a summary.csv (or a run directory containing one)."""
    p = Path(path)
    if p.is_dir():
        p = p / "summary.csv"
    return pd.read_csv(p)


def ci_table(df: pd.DataFrame, metrics: list[str] | None = None, confidence: float = 0.95) -> pd.DataFrame:
    """Mean and CI bounds per (config, pattern) for each metric. Long format."""
    metrics = [m for m in (metrics or KEY_METRICS) if m in df.columns]
    rows = []
    for (config, pattern), g in df.groupby(["config", "pattern"], sort=False):
        for m in metrics:
            vals = g[m].dropna().to_numpy(dtype=float)
            if len(vals) >= 2:
                mean, lo, hi = compute_ci(vals.tolist(), confidence)
            elif len(vals) == 1:
                mean, lo, hi = float(vals[0]), float("nan"), float("nan")
            else:
                continue
            rows.append({"config": config, "pattern": pattern, "metric": m, "n": len(vals),
                         "mean": mean, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def best_fixed(df: pd.DataFrame, pattern: str, metric: str = "e2e_p99",
               lower_is_better: bool = True) -> str | None:
    """Strongest fixed config for `pattern` on `metric`, among configs with no errors."""
    sub = df[(df["pattern"] == pattern) & (df["config"] != ADAPTIVE_LABEL)]
    if sub.empty:
        return None
    clean = sub.groupby("config")["n_errors"].sum()
    eligible = clean[clean == 0].index
    sub = sub[sub["config"].isin(eligible)]
    if sub.empty:
        return None
    means = sub.groupby("config")[metric].mean()
    return str(means.idxmin() if lower_is_better else means.idxmax())


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone step-down)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def _paired_vectors(df: pd.DataFrame, pattern: str, a: str, b: str, metric: str):
    sa = df[(df["pattern"] == pattern) & (df["config"] == a)].set_index("rep")[metric]
    sb = df[(df["pattern"] == pattern) & (df["config"] == b)].set_index("rep")[metric]
    reps = sorted(set(sa.index) & set(sb.index))
    return sa.loc[reps].to_numpy(float), sb.loc[reps].to_numpy(float), reps


def cohens_dz(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else float("nan")


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare_to_best_fixed(df: pd.DataFrame, treatment: str = ADAPTIVE_LABEL,
                          metric: str = "e2e_p99", throughput_metric: str = "throughput_tok_per_s",
                          noninferiority: float = 0.98) -> pd.DataFrame:
    """
    H1 test, per load pattern: treatment vs the best fixed config on `metric`.

    Columns: pattern, best_fixed, n_pairs, treatment_mean, fixed_mean, pct_change (negative = treatment
    lower), ci_lo/ci_hi (bootstrap 95% CI of the mean paired difference), cohens_dz, p_ttest, p_wilcoxon,
    p_holm (Holm-corrected Wilcoxon), throughput_ratio (treatment / fixed) and whether throughput is
    non-inferior (>= `noninferiority` x fixed), and a plain-language verdict.
    """
    rows = []
    for pattern in sorted(df["pattern"].unique()):
        base = best_fixed(df, pattern, metric)
        if base is None or treatment not in set(df[df["pattern"] == pattern]["config"]):
            continue
        t, f, reps = _paired_vectors(df, pattern, treatment, base, metric)
        if len(reps) < 3:
            continue
        diff = t - f
        test = paired_significance_test(f.tolist(), t.tolist())
        tp_t, tp_f, _ = _paired_vectors(df, pattern, treatment, base, throughput_metric)
        lo, hi = bootstrap_ci(diff)
        rows.append({
            "pattern": pattern, "best_fixed": base, "n_pairs": len(reps),
            "treatment_mean": float(t.mean()), "fixed_mean": float(f.mean()),
            "pct_change": float(100 * diff.mean() / f.mean()),
            "diff_ci_lo": lo, "diff_ci_hi": hi, "cohens_dz": cohens_dz(diff),
            "p_ttest": test["t_test"]["p_value"], "p_wilcoxon": test["wilcoxon_test"]["p_value"],
            "throughput_ratio": float(tp_t.mean() / tp_f.mean()),
            "throughput_noninferior": bool(tp_t.mean() >= noninferiority * tp_f.mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_wilcoxon"].tolist())
    out["verdict"] = [verdict(r) for r in out.to_dict("records")]
    return out


def verdict(row: dict, alpha: float = 0.05) -> str:
    """Plain-language conclusion. States null and negative results explicitly."""
    sig = row["p_holm"] < alpha and row["p_ttest"] < alpha
    better = row["pct_change"] < 0
    if sig and better and row["throughput_noninferior"]:
        return f"adaptive better: p99 {row['pct_change']:.1f}% (Holm p={row['p_holm']:.3g}), throughput non-inferior"
    if sig and better:
        return f"adaptive lower p99 ({row['pct_change']:.1f}%) but throughput regressed to {row['throughput_ratio']:.2f}x"
    if sig and not better:
        return f"adaptive WORSE: p99 +{row['pct_change']:.1f}% (Holm p={row['p_holm']:.3g})"
    return f"no significant difference ({row['pct_change']:+.1f}%, Holm p={row['p_holm']:.3g}); a tuned fixed config matches adaptive here"


def ablation(df: pd.DataFrame, pattern: str, metric: str = "e2e_p99") -> pd.DataFrame:
    """
    Decompose gains relative to the naive reference (fp16, smallest batch): what does
    quantization alone, batching alone, both, and adaptivity add? Expects fixed
    configs named '<quant>-bs<N>'. Returns pct change vs reference per config.
    """
    sub = df[df["pattern"] == pattern]
    means = sub.groupby("config")[metric].mean()
    ref_label = "fp16-bs1" if "fp16-bs1" in means.index else means.index[0]
    ref = means[ref_label]
    return pd.DataFrame({"config": means.index, "mean": means.values,
                         "pct_vs_reference": 100 * (means.values - ref) / ref,
                         "reference": ref_label})


def to_markdown_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """Dependency-free GitHub markdown table."""
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
