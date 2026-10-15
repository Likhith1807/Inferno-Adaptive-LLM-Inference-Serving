"""
Routing policies over a pool of pre-loaded, heterogeneous replicas.

vLLM cannot change batch size or quantization on a live server (that needs a reload, measured by the
harness). So "adaptivity" here means choosing *which pre-loaded configuration serves each request*, driven
by live load and request length. Tier 0 is the low-latency decode tier (small batch cap, small prefill
chunks); the last tier is the throughput/prefill tier (large batch cap, large prefill chunks).

All baselines run in the SAME pool (same GPU memory residency, same contention), so the comparison
"adaptive vs best fixed" cannot be won by giving the baseline a different hardware situation.

Policies are pure Python with no I/O: they see a PoolState snapshot and return a replica index. That keeps
them unit-testable and their decisions interpretable (every decision has a reason string).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PoolState:
    inflight: list[int]                  # requests currently in flight per replica
    capacity: list[int]                  # per-replica max concurrent sequences (vLLM max_num_seqs)
    ttft_p95: float | None = None        # recent completed-request TTFT p95 (seconds), pool-wide
    e2e_p99: float | None = None
    now: float = 0.0

    @property
    def n(self) -> int:
        return len(self.inflight)

    def utilization(self, i: int) -> float:
        return self.inflight[i] / max(1, self.capacity[i])


class Policy:
    name = "policy"

    def route(self, prompt_tokens: int, state: PoolState) -> int:
        raise NotImplementedError

    def tick(self, state: PoolState) -> dict:
        """Called periodically by the control loop. Returns a log record (may be empty)."""
        return {}


class PinnedPolicy(Policy):
    """Fixed configuration: every request goes to one replica. The replicas of the pool remain loaded."""

    def __init__(self, index: int, name: str | None = None):
        self.index = index
        self.name = name or f"fixed-{index}"

    def route(self, prompt_tokens: int, state: PoolState) -> int:
        return self.index


class LeastLoadedPolicy(Policy):
    """Static load balancer: fewest in-flight relative to capacity. Non-adaptive multi-replica baseline."""
    name = "static-lb"

    def route(self, prompt_tokens: int, state: PoolState) -> int:
        return min(range(state.n), key=lambda i: (state.utilization(i), i))


@dataclass
class AdaptiveConfig:
    long_prompt_tokens: int = 512        # requests at/above this count as prefill-heavy
    escalate_util: float = 0.70          # tier-0 utilization that triggers escalation
    deescalate_util: float = 0.25        # pool utilization below which we consider calming down
    ttft_slo_sec: float = 0.5
    calm_ticks_required: int = 5         # hysteresis: consecutive calm ticks before de-escalating
    spill_util: float = 0.90             # hard safety: a saturated tier spills to the least loaded one


@dataclass
class AdaptiveTierPolicy(Policy):
    """
    Three-level interpretable controller (level is the only mutable state):

      level 0  calm       every request -> tier 0 (low-latency decode tier).
      level 1  pressured  prefill-heavy requests -> last tier (isolates long prefills from tier-0 decode);
                          the rest stay on tier 0.
      level 2  overloaded least-loaded across all tiers (use every replica).

    Escalate one level when tier-0 utilization or recent TTFT p95 breaches its threshold; de-escalate one
    level only after `calm_ticks_required` consecutive calm ticks (hysteresis prevents flapping).
    A hard spill rule sends a request to the least-loaded tier if its chosen tier is saturated.
    """
    cfg: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    name: str = "adaptive"
    level: int = 0
    _calm: int = 0

    def route(self, prompt_tokens: int, state: PoolState) -> int:
        last = state.n - 1
        if state.n == 1:
            return 0
        if self.level == 0:
            choice = 0
        elif self.level == 1:
            choice = last if prompt_tokens >= self.cfg.long_prompt_tokens else 0
        else:
            choice = min(range(state.n), key=lambda i: (state.utilization(i), i))
        if state.utilization(choice) >= self.cfg.spill_util:
            choice = min(range(state.n), key=lambda i: (state.utilization(i), i))
        return choice

    def tick(self, state: PoolState) -> dict:
        c = self.cfg
        pressured = (state.utilization(0) >= c.escalate_util
                     or (state.ttft_p95 is not None and state.ttft_p95 > c.ttft_slo_sec))
        pool_util = sum(state.inflight) / max(1, sum(state.capacity))
        calm = (pool_util <= c.deescalate_util
                and (state.ttft_p95 is None or state.ttft_p95 <= 0.5 * c.ttft_slo_sec))
        before, reason = self.level, "hold"
        if pressured and self.level < 2:
            self.level += 1
            self._calm = 0
            reason = f"escalate: util0={state.utilization(0):.2f} ttft_p95={state.ttft_p95}"
        elif calm and self.level > 0:
            self._calm += 1
            if self._calm >= c.calm_ticks_required:
                self.level -= 1
                self._calm = 0
                reason = f"de-escalate: pool_util={pool_util:.2f}"
        elif not calm:
            self._calm = 0
        return {"t": state.now, "level_before": before, "level": self.level, "reason": reason,
                "util0": state.utilization(0), "pool_util": pool_util, "ttft_p95": state.ttft_p95,
                "e2e_p99": state.e2e_p99, "inflight": list(state.inflight)}
