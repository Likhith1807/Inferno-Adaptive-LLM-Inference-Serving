"""Adaptive tier policy: escalation, hysteresis, spill, and the fixed baselines. Pure logic, no GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inferno.controller.routing_policies import (AdaptiveConfig, AdaptiveTierPolicy, LeastLoadedPolicy,
                                                 PinnedPolicy, PoolState)


def state(inflight, capacity=(16, 64), ttft=None):
    return PoolState(inflight=list(inflight), capacity=list(capacity), ttft_p95=ttft)


def test_pinned_ignores_state():
    p = PinnedPolicy(1)
    assert p.route(10, state([16, 0])) == 1 and p.route(2000, state([0, 64])) == 1


def test_least_loaded_uses_relative_utilization():
    assert LeastLoadedPolicy().route(10, state([8, 8])) == 1        # 50% vs 12.5%
    assert LeastLoadedPolicy().route(10, state([1, 32])) == 0       # 6% vs 50%


def test_calm_level0_routes_everything_to_tier0():
    p = AdaptiveTierPolicy()
    assert p.level == 0
    assert p.route(50, state([2, 0])) == 0 and p.route(2000, state([2, 0])) == 0


def test_escalation_sends_long_prompts_to_bulk_tier():
    p = AdaptiveTierPolicy()
    p.tick(state([14, 0]))                                          # util0 = 0.875 >= 0.70
    assert p.level == 1
    assert p.route(900, state([2, 0])) == 1                         # prefill-heavy -> bulk tier
    assert p.route(100, state([2, 0])) == 0                         # short stays on the decode tier


def test_second_escalation_uses_all_replicas_and_caps_at_level2():
    p = AdaptiveTierPolicy()
    p.tick(state([14, 0])); p.tick(state([15, 10])); p.tick(state([16, 40]))
    assert p.level == 2
    assert p.route(100, state([12, 3])) == 1                         # least loaded relative to capacity


def test_deescalation_requires_consecutive_calm_ticks():
    p = AdaptiveTierPolicy(cfg=AdaptiveConfig(calm_ticks_required=3))
    p.tick(state([14, 0]))
    assert p.level == 1
    p.tick(state([1, 0])); p.tick(state([1, 0]))
    assert p.level == 1                                              # only 2 calm ticks so far
    p.tick(state([10, 12]))                                          # moderately busy: not calm, resets counter
    p.tick(state([1, 0])); p.tick(state([1, 0]))
    assert p.level == 1
    p.tick(state([1, 0]))
    assert p.level == 0


def test_ttft_slo_breach_escalates_even_when_utilization_is_low():
    p = AdaptiveTierPolicy(cfg=AdaptiveConfig(ttft_slo_sec=0.5))
    p.tick(state([2, 0], ttft=0.9))
    assert p.level == 1


def test_saturated_tier_spills_to_least_loaded():
    p = AdaptiveTierPolicy()                                         # level 0 wants tier 0
    assert p.route(50, state([16, 5])) == 1                          # tier 0 at 100% -> spill
