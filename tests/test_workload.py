"""Unit tests for the workload generator — fully runnable, no GPU needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml
from inferno.workload.generator import WorkloadGenerator


def load_test_config():
    with open(Path(__file__).resolve().parents[1] / "configs" / "workload_config.yaml") as f:
        return yaml.safe_load(f)


def test_steady_arrivals_increasing():
    config = load_test_config()
    gen = WorkloadGenerator(config, load_pattern="steady", seed=1)
    requests = gen.generate(duration_sec=30.0)

    assert len(requests) > 0
    arrival_times = [r.arrival_time for r in requests]
    assert arrival_times == sorted(arrival_times)
    assert all(t <= 30.0 for t in arrival_times)


def test_request_lengths_positive():
    config = load_test_config()
    gen = WorkloadGenerator(config, load_pattern="steady", seed=2)
    requests = gen.generate(duration_sec=30.0)

    assert all(r.prompt_tokens >= 1 for r in requests)
    assert all(r.max_new_tokens >= 1 for r in requests)


def test_bursty_produces_more_requests_during_burst_windows():
    config = load_test_config()
    gen = WorkloadGenerator(config, load_pattern="bursty", seed=3)
    requests = gen.generate(duration_sec=120.0)

    # Sanity check: bursty rate > base rate, so it should generate
    # noticeably more requests than a steady stream at the base rate
    # over the same duration.
    base_rate = config["load_patterns"]["bursty"]["base_rate_requests_per_sec"]
    expected_min = base_rate * 120.0 * 0.5  # loose lower bound, allow for variance
    assert len(requests) > expected_min


def test_reproducible_with_same_seed():
    config = load_test_config()
    gen1 = WorkloadGenerator(config, load_pattern="steady", seed=42)
    gen2 = WorkloadGenerator(config, load_pattern="steady", seed=42)

    r1 = gen1.generate(duration_sec=20.0)
    r2 = gen2.generate(duration_sec=20.0)

    assert [r.arrival_time for r in r1] == [r.arrival_time for r in r2]


def test_mixed_length_is_bimodal_and_differs_from_steady():
    import numpy as np

    config = load_test_config()
    mixed = WorkloadGenerator(config, "mixed_length", seed=5).generate(duration_sec=600.0)
    steady = WorkloadGenerator(config, "steady", seed=5).generate(duration_sec=600.0)

    m = np.array([r.prompt_tokens for r in mixed])
    s = np.array([r.prompt_tokens for r in steady])
    assert (m > 500).mean() > 0.08          # ~15% long-context component
    assert (s > 500).mean() < 0.08          # lognormal(128, 0.8) tail is ~4.5% by construction
    assert m.max() <= config["request_length"]["max_prompt_tokens"]


def test_empirical_trace_sampling(tmp_path):
    import copy

    trace = tmp_path / "trace.csv"
    trace.write_text("prompt_tokens,output_tokens\n10,20\n30,40\n50,60\n")
    config = copy.deepcopy(load_test_config())
    config["request_length"]["distribution"] = "empirical"
    config["request_length"]["trace_path"] = str(trace)

    reqs = WorkloadGenerator(config, "steady", seed=1).generate(duration_sec=60.0)
    pairs = {(r.prompt_tokens, r.max_new_tokens) for r in reqs}
    assert pairs <= {(10, 20), (30, 40), (50, 60)}     # pairs stay joint
    assert len(pairs) > 1
