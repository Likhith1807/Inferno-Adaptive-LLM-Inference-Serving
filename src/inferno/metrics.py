"""
Per-request records and run-level summaries.

Latency is measured from the request's *scheduled* arrival time (open-loop), so
queueing delay caused by an overloaded server is counted, not hidden.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RequestResult:
    request_id: int
    scheduled_time: float            # seconds since run start
    start_time: float                # when the client actually sent it
    first_token_time: float | None   # None if no token arrived
    end_time: float | None
    prompt_tokens: int               # server-reported when available
    output_tokens: int
    ok: bool
    error: str | None = None
    replica: int = -1                       # which replica served it (-1 = single server)
    route_overhead_us: float | None = None  # router-added latency before the replica saw the request

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.scheduled_time

    @property
    def e2e(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.scheduled_time

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first."""
        if self.first_token_time is None or self.end_time is None or self.output_tokens < 2:
            return None
        return (self.end_time - self.first_token_time) / (self.output_tokens - 1)

    def to_row(self) -> dict:
        row = asdict(self)
        row.update(ttft=self.ttft, e2e=self.e2e, tpot=self.tpot)
        return row


def _pcts(values: list[float], prefix: str) -> dict:
    out = {}
    for p in (50, 95, 99):
        out[f"{prefix}_p{p}"] = float(np.percentile(values, p)) if values else float("nan")
    return out


def summarize(
    results: list[RequestResult],
    wall_time_sec: float,
    slo: dict | None = None,
    warmup_sec: float = 0.0,
) -> dict:
    """
    Aggregate a run into one row of headline metrics.

    Requests scheduled before `warmup_sec` are excluded. `wall_time_sec` is the measured arrival window
    (run duration minus warmup); rates use max(window, time until the last completion) so drain time counts.
    """
    kept = [r for r in results if r.scheduled_time >= warmup_sec]
    ok = [r for r in kept if r.ok]
    ttft = [r.ttft for r in ok if r.ttft is not None]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    e2e = [r.e2e for r in ok if r.e2e is not None]
    out_tokens = sum(r.output_tokens for r in ok)

    # Throughput is measured over the busy span, not just the arrival window: if the server is still draining a
    # backlog after arrivals stop, that time counts. Otherwise every policy that eventually finishes would report
    # the offered load and throughput could not distinguish them.
    last_end = max((r.end_time for r in ok if r.end_time is not None), default=0.0)
    span = max(wall_time_sec, last_end - warmup_sec)

    summary = {
        "n_requests": len(kept),
        "n_ok": len(ok),
        "n_errors": len(kept) - len(ok),
        "output_tokens": out_tokens,
        "offered_tok_per_s": out_tokens / wall_time_sec if wall_time_sec > 0 else float("nan"),
        "throughput_tok_per_s": out_tokens / span if span > 0 else float("nan"),
        "busy_span_sec": span,
        "request_rate_per_s": len(ok) / span if span > 0 else float("nan"),
    }
    summary.update(_pcts(ttft, "ttft"))
    summary.update(_pcts(tpot, "tpot"))
    summary.update(_pcts(e2e, "e2e"))

    if slo:
        met = [
            r for r in ok
            if r.ttft is not None and r.ttft <= slo["ttft_sec"]
            and (r.tpot is None or r.tpot <= slo["tpot_sec"])
        ]
        summary["slo_attainment"] = len(met) / len(kept) if kept else float("nan")
        summary["goodput_req_per_s"] = len(met) / span if span > 0 else float("nan")
    return summary
