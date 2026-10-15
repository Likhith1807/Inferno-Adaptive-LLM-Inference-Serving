"""
Workload generator: produces a realistic stream of inference requests.

Fully implemented — no GPU required. This is the piece that should be
working and tested first (Weeks 1-2), since the rest of the pipeline
consumes its output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .traces import empirical_sampler, load_length_trace


@dataclass
class Request:
    request_id: int
    arrival_time: float          # seconds, relative to stream start
    prompt_tokens: int
    max_new_tokens: int


class WorkloadGenerator:
    """
    Generates a request stream using a Poisson arrival process and a
    configurable request-length distribution.

    Load patterns supported: "steady", "bursty", "mixed_length".
    See configs/workload_config.yaml for the parameters each pattern uses.
    """

    def __init__(self, config: dict, load_pattern: str, seed: int | None = None):
        self.config = config
        self.load_pattern = load_pattern
        self.rng = np.random.default_rng(seed if seed is not None else config.get("random_seed", 42))
        self._trace: np.ndarray | None = None
        if config["request_length"].get("distribution") == "empirical":
            self._trace = load_length_trace(Path(config["request_length"]["trace_path"]))

    def _arrival_times(self, duration_sec: float) -> np.ndarray:
        pattern_cfg = self.config["load_patterns"][self.load_pattern]

        if self.load_pattern == "bursty":
            return self._bursty_arrivals(duration_sec, pattern_cfg)

        rate = pattern_cfg["rate_requests_per_sec"]
        # Poisson process: inter-arrival times are exponential(1/rate)
        times = []
        t = 0.0
        while t < duration_sec:
            t += self.rng.exponential(1.0 / rate)
            if t < duration_sec:
                times.append(t)
        return np.array(times)

    def _bursty_arrivals(self, duration_sec: float, cfg: dict) -> np.ndarray:
        base_rate = cfg["base_rate_requests_per_sec"]
        burst_rate = cfg["burst_rate_requests_per_sec"]
        burst_duration = cfg["burst_duration_sec"]
        burst_interval = cfg["burst_interval_sec"]

        times = []
        t = 0.0
        while t < duration_sec:
            in_burst = (t % burst_interval) < burst_duration
            rate = burst_rate if in_burst else base_rate
            t += self.rng.exponential(1.0 / rate)
            if t < duration_sec:
                times.append(t)
        return np.array(times)

    def _lognormal(self, n: int, mean: float, sigma: float) -> np.ndarray:
        return np.clip(self.rng.lognormal(mean=np.log(mean), sigma=sigma, size=n).astype(int), 1, None)

    def _sample_lengths(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        length_cfg = self.config["request_length"]
        pattern_cfg = self.config["load_patterns"][self.load_pattern]
        mixture = pattern_cfg.get("length_mixture")

        if mixture:
            weights = np.array([c["weight"] for c in mixture], dtype=float)
            comp = self.rng.choice(len(mixture), size=n, p=weights / weights.sum())
            prompt = np.empty(n, dtype=int)
            output = np.empty(n, dtype=int)
            for k, c in enumerate(mixture):
                m = comp == k
                cnt = int(m.sum())
                prompt[m] = self._lognormal(cnt, c["prompt_tokens_mean"], c["prompt_tokens_sigma"])
                output[m] = self._lognormal(cnt, c["max_new_tokens_mean"], c["max_new_tokens_sigma"])
        elif self._trace is not None:
            pairs = empirical_sampler(self._trace, self.rng, n)
            prompt, output = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        else:
            prompt = self._lognormal(n, length_cfg["prompt_tokens_mean"], length_cfg["prompt_tokens_sigma"])
            output = self._lognormal(n, length_cfg["max_new_tokens_mean"], length_cfg["max_new_tokens_sigma"])

        prompt = np.clip(prompt, 1, length_cfg.get("max_prompt_tokens", 1 << 30))
        output = np.clip(output, 1, length_cfg.get("max_new_tokens_cap", 1 << 30))
        return prompt, output

    def generate(self, duration_sec: float) -> list[Request]:
        """Generate a full request stream for the given duration."""
        arrival_times = self._arrival_times(duration_sec)
        prompt_tokens, max_new_tokens = self._sample_lengths(len(arrival_times))

        requests = [
            Request(
                request_id=i,
                arrival_time=float(arrival_times[i]),
                prompt_tokens=int(prompt_tokens[i]),
                max_new_tokens=int(max_new_tokens[i]),
            )
            for i in range(len(arrival_times))
        ]
        return requests
