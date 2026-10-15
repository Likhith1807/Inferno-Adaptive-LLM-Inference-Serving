"""
Closed-loop routing over a live replica pool.

A single asyncio loop replays the open-loop workload; each request is routed by a Policy that sees an
up-to-the-moment PoolState, and a separate control tick (default 1 s) lets the policy update its level from
recent latency. Every tick is logged, so the controller's behaviour over time can be plotted (the "watch it
adapt" demo) and audited.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

import httpx
import numpy as np

from ..metrics import RequestResult
from ..serving.client import send_request
from ..workload.generator import Request
from .routing_policies import Policy, PoolState


class RollingStats:
    """Recent completed-request TTFT / e2e (count-bounded window)."""

    def __init__(self, window: int = 64):
        self.ttft: deque[float] = deque(maxlen=window)
        self.e2e: deque[float] = deque(maxlen=window)

    def record(self, res: RequestResult) -> None:
        if res.ok and res.ttft is not None and res.e2e is not None:
            self.ttft.append(res.ttft)
            self.e2e.append(res.e2e)

    def ttft_p95(self) -> float | None:
        return float(np.percentile(self.ttft, 95)) if len(self.ttft) >= 8 else None

    def e2e_p99(self) -> float | None:
        return float(np.percentile(self.e2e, 99)) if len(self.e2e) >= 8 else None


async def run_policy_workload(
    requests: list[Request],
    base_urls: list[str],
    models: list[str],
    capacities: list[int],
    policy: Policy,
    tick_sec: float = 1.0,
    request_timeout_sec: float = 300.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[RequestResult], list[dict]]:
    """Returns (per-request results, controller tick log). Blocks until all requests complete."""
    urls = [u.rstrip("/") + "/v1/completions" for u in base_urls]
    inflight = [0] * len(urls)
    stats = RollingStats()
    decisions: list[dict] = []
    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=256)

    def snapshot(t0: float) -> PoolState:
        return PoolState(inflight=list(inflight), capacity=list(capacities), ttft_p95=stats.ttft_p95(),
                         e2e_p99=stats.e2e_p99(), now=time.monotonic() - t0)

    async def one(client, idx, req, t0):
        try:
            res = await send_request(client, urls[idx], models[idx], req, t0)
        finally:
            inflight[idx] -= 1
        res.replica = idx
        stats.record(res)
        return res

    async def control_loop(t0, stop: asyncio.Event):
        while not stop.is_set():
            rec = policy.tick(snapshot(t0))
            if rec:
                decisions.append(rec)
            try:
                await asyncio.wait_for(stop.wait(), timeout=tick_sec)
            except asyncio.TimeoutError:
                pass

    async with httpx.AsyncClient(timeout=request_timeout_sec, limits=limits, transport=transport) as client:
        t0 = time.monotonic()
        stop = asyncio.Event()
        ctl = asyncio.create_task(control_loop(t0, stop))
        tasks = []
        for req in sorted(requests, key=lambda r: r.arrival_time):
            while (delay := req.arrival_time - (time.monotonic() - t0)) > 0:
                await asyncio.sleep(delay)
            idx = policy.route(req.prompt_tokens, snapshot(t0))
            inflight[idx] += 1
            tasks.append(asyncio.create_task(one(client, idx, req, t0)))
        results = list(await asyncio.gather(*tasks))
        stop.set()
        await ctl
    return results, decisions
