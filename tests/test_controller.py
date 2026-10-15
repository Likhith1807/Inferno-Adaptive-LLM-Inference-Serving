"""Unit tests for the adaptive controller — fully runnable, no GPU needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inferno.controller.policy import AdaptivePolicy
from inferno.controller.metrics_monitor import MetricsMonitor


def test_cold_start_returns_conservative_default():
    monitor = MetricsMonitor()
    policy = AdaptivePolicy()
    decision = policy.decide(monitor)

    assert decision.reason == "cold_start"
    assert decision.batch_size == policy.batch_sizes[0]
    assert decision.quantization == policy.quant_levels[0]


def test_escalates_under_overload():
    monitor = MetricsMonitor()
    policy = AdaptivePolicy(p99_latency_threshold_sec=1.0, queue_depth_threshold=5)

    # Feed it overloaded observations
    for _ in range(20):
        monitor.record(latency_sec=3.0, queue_depth=10)

    decision = policy.decide(monitor)
    assert "overloaded" in decision.reason
    assert decision.batch_size >= policy.batch_sizes[0]


def test_deescalates_under_light_load():
    monitor = MetricsMonitor()
    policy = AdaptivePolicy(p99_latency_threshold_sec=2.0, queue_depth_threshold=20)

    # First push it up under overload...
    for _ in range(20):
        monitor.record(latency_sec=5.0, queue_depth=30)
    policy.decide(monitor)

    # ...then feed light-load observations and confirm it steps back down
    monitor2 = MetricsMonitor()
    for _ in range(20):
        monitor2.record(latency_sec=0.1, queue_depth=1)
    decision = policy.decide(monitor2)

    assert "under threshold" in decision.reason
