"""Calibration math: capacity from an overloaded run, and rate derivation. Pure logic, no GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import yaml

from inferno.experiments.calibrate import capacity_rps, derive_rates, mean_output_tokens
from inferno.metrics import RequestResult


def _r(i, end, tokens, ok=True):
    return RequestResult(i, 0.0, 0.0, 0.1, end, 10, tokens, ok)


def test_capacity_counts_only_completions_inside_window():
    results = [_r(0, 5.0, 100), _r(1, 12.0, 200), _r(2, 20.0, 200), _r(3, 45.0, 999), _r(4, 25.0, 50, ok=False)]
    cap = capacity_rps(results, (10.0, 30.0), mean_out_tokens=100.0)
    assert cap["tokens_per_s"] == pytest.approx(400 / 20.0)          # requests 1 and 2 only
    assert cap["requests_per_s"] == pytest.approx(0.2)
    assert cap["completed_in_window"] == 2


def test_derived_rates_are_fractions_of_capacity_and_bursts_exceed_it():
    r = derive_rates(cap_steady_rps=10.0, cap_mixed_rps=4.0, rho=0.6)
    assert r["steady"]["rate_requests_per_sec"] == pytest.approx(6.0)
    assert r["mixed_length"]["rate_requests_per_sec"] == pytest.approx(2.4)
    assert r["bursty"]["base_rate_requests_per_sec"] < 10.0 < r["bursty"]["burst_rate_requests_per_sec"]


def test_mean_output_tokens_reflects_the_length_mixture():
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "workload_config.yaml"))
    steady = mean_output_tokens(cfg, "steady")
    mixed = mean_output_tokens(cfg, "mixed_length")
    assert 150 < steady < 400
    assert mixed != pytest.approx(steady, rel=0.05)                  # mixture has a different mean
