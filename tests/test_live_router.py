"""Closed-loop router against a slow mock server: the adaptive policy must escalate under a burst."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from inferno.controller.live import run_policy_workload
from inferno.controller.routing_policies import AdaptiveConfig, AdaptiveTierPolicy, LeastLoadedPolicy, PinnedPolicy
from inferno.workload.generator import Request


def slow_server(delay=0.25):
    hits = {"a": 0, "b": 0}

    async def handler(request):
        hits[request.url.host] += 1
        await asyncio.sleep(delay)
        chunk = "data: " + json.dumps({"choices": [{"text": " t"}]})
        return httpx.Response(200, content="\n".join([chunk, chunk, "data: [DONE]", ""]).encode())

    return httpx.MockTransport(handler), hits


def run(policy, n=40, delay=0.25, tick=0.05):
    transport, hits = slow_server(delay)
    reqs = [Request(i, 0.0, 800 if i % 4 == 0 else 50, 2) for i in range(n)]      # burst, 25% long prompts
    res, log = asyncio.run(run_policy_workload(reqs, ["http://a", "http://b"], ["m", "m"], [16, 64], policy,
                                               tick_sec=tick, transport=transport))
    return res, log, hits


def test_pinned_uses_only_its_replica():
    res, log, hits = run(PinnedPolicy(1))
    assert all(r.ok and r.replica == 1 for r in res) and hits["a"] == 0


def test_adaptive_escalates_under_burst_and_uses_both_tiers():
    policy = AdaptiveTierPolicy(cfg=AdaptiveConfig(escalate_util=0.5, long_prompt_tokens=512))
    res, log, hits = run(policy)
    assert all(r.ok for r in res) and len(res) == 40
    assert hits["a"] > 0 and hits["b"] > 0                        # spill and/or long-prompt isolation


def test_static_lb_spreads_a_burst():
    res, _, hits = run(LeastLoadedPolicy())
    assert hits["a"] > 0 and hits["b"] > 0
