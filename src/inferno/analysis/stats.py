"""
Statistical utilities for the experimental campaign.

Fully implemented — this is what turns raw run outputs into the
defensible numbers docs/methodology.md requires (mean +/- 95% CI,
paired significance testing). No GPU required to test this module.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def compute_ci(samples: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Returns (mean, lower_bound, upper_bound) for the given confidence level,
    using a t-distribution (appropriate for small sample sizes like n=5-10).
    """
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 2:
        raise ValueError("Need at least 2 samples to compute a confidence interval")

    mean = float(np.mean(arr))
    sem = stats.sem(arr)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, mean - margin, mean + margin


def paired_significance_test(
    baseline: list[float], treatment: list[float], alpha: float = 0.05
) -> dict:
    """
    Paired comparison between a fixed baseline configuration and the
    adaptive policy (or any two matched conditions), used to validate H1-H3.

    Runs both a paired t-test and a Wilcoxon signed-rank test (the
    latter is distribution-free and more defensible with small n);
    reports both so the writeup can note if they disagree.
    """
    baseline_arr = np.asarray(baseline, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)
    if len(baseline_arr) != len(treatment_arr):
        raise ValueError("Paired test requires equal-length, matched samples")

    t_stat, t_pvalue = stats.ttest_rel(treatment_arr, baseline_arr)
    try:
        w_stat, w_pvalue = stats.wilcoxon(treatment_arr, baseline_arr)
    except ValueError:
        # Wilcoxon fails if all differences are zero — degenerate but valid case
        w_stat, w_pvalue = float("nan"), 1.0

    mean_diff = float(np.mean(treatment_arr - baseline_arr))

    return {
        "mean_difference": mean_diff,
        "t_test": {"statistic": float(t_stat), "p_value": float(t_pvalue)},
        "wilcoxon_test": {"statistic": float(w_stat), "p_value": float(w_pvalue)},
        "significant_at_alpha": bool(t_pvalue < alpha and w_pvalue < alpha),
        "alpha": alpha,
    }
