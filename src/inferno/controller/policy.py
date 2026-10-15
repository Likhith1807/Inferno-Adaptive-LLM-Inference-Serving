"""
Adaptive controller: rule-based policy first (Week 7).

Deliberately NOT reinforcement learning — a threshold-based policy is
enough to test H1, is fully interpretable (important for the viva),
and avoids the scope trap of building an RL training loop on top of
an already-multi-part system. Only add a learned policy afterward,
and only if there's time left (see docs/roadmap.md "non-negotiables").
"""
from __future__ import annotations

from dataclasses import dataclass

from .metrics_monitor import MetricsMonitor


@dataclass
class PolicyDecision:
    batch_size: int
    quantization: str      # "fp16" | "int8" | "int4"
    target_replica: int | None = None  # None = let the dispatcher choose
    reason: str = ""


class AdaptivePolicy:
    """
    Threshold-based adaptive policy.

    Escalates (larger batch / more aggressive quantization) when queue
    depth or p99 latency exceeds a threshold; de-escalates when load
    is low, to protect per-request latency.
    """

    def __init__(
        self,
        p99_latency_threshold_sec: float = 2.0,
        queue_depth_threshold: int = 20,
        batch_sizes: tuple[int, ...] = (1, 8, 16, 32, 64),
        quant_levels: tuple[str, ...] = ("fp16", "int8", "int4"),
    ):
        self.p99_threshold = p99_latency_threshold_sec
        self.queue_threshold = queue_depth_threshold
        self.batch_sizes = batch_sizes
        self.quant_levels = quant_levels

        self._batch_idx = 0
        self._quant_idx = 0

    def decide(self, monitor: MetricsMonitor) -> PolicyDecision:
        snapshot = monitor.snapshot()
        p99 = snapshot["p99_latency"]
        queue_depth = snapshot["mean_queue_depth"]

        if p99 is None or queue_depth is None:
            # Not enough data yet — stay at the conservative default.
            return PolicyDecision(
                batch_size=self.batch_sizes[0],
                quantization=self.quant_levels[0],
                reason="cold_start",
            )

        overloaded = p99 > self.p99_threshold or queue_depth > self.queue_threshold

        if overloaded:
            self._batch_idx = min(self._batch_idx + 1, len(self.batch_sizes) - 1)
            self._quant_idx = min(self._quant_idx + 1, len(self.quant_levels) - 1)
            reason = f"overloaded (p99={p99:.2f}s, queue={queue_depth:.1f}) -> escalate"
        else:
            self._batch_idx = max(self._batch_idx - 1, 0)
            self._quant_idx = max(self._quant_idx - 1, 0)
            reason = f"under threshold (p99={p99:.2f}s, queue={queue_depth:.1f}) -> de-escalate"

        return PolicyDecision(
            batch_size=self.batch_sizes[self._batch_idx],
            quantization=self.quant_levels[self._quant_idx],
            reason=reason,
        )
