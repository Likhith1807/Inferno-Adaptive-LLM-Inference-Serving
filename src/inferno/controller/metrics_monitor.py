"""
Rolling-window metrics monitor.

Fully implemented — feeds the adaptive controller with live percentile
estimates. No GPU required; this can be unit-tested standalone.
"""
from __future__ import annotations

from collections import deque
import numpy as np


class MetricsMonitor:
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._queue_depths: deque[int] = deque(maxlen=window_size)

    def record(self, latency_sec: float, queue_depth: int) -> None:
        self._latencies.append(latency_sec)
        self._queue_depths.append(queue_depth)

    def percentile_latency(self, p: float) -> float | None:
        if not self._latencies:
            return None
        return float(np.percentile(list(self._latencies), p))

    def mean_queue_depth(self) -> float | None:
        if not self._queue_depths:
            return None
        return float(np.mean(list(self._queue_depths)))

    def snapshot(self) -> dict:
        return {
            "p50_latency": self.percentile_latency(50),
            "p95_latency": self.percentile_latency(95),
            "p99_latency": self.percentile_latency(99),
            "mean_queue_depth": self.mean_queue_depth(),
            "n_samples": len(self._latencies),
        }
