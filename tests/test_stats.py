"""Unit tests for the significance-testing utilities — fully runnable, no GPU needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inferno.analysis.stats import compute_ci, paired_significance_test


def test_compute_ci_basic():
    samples = [1.0, 1.2, 0.9, 1.1, 1.05, 0.95, 1.15, 1.0]
    mean, lower, upper = compute_ci(samples, confidence=0.95)

    assert lower < mean < upper


def test_paired_significance_detects_real_difference():
    baseline = [2.0, 2.1, 1.9, 2.05, 2.0, 1.95, 2.1, 2.0]
    treatment = [1.0, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.0]  # clearly lower latency

    result = paired_significance_test(baseline, treatment)
    assert result["mean_difference"] < 0
    assert result["significant_at_alpha"] is True


def test_paired_significance_no_difference():
    baseline = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    treatment = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    result = paired_significance_test(baseline, treatment)
    assert result["mean_difference"] == 0
