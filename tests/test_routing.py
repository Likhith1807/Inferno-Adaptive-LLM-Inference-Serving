"""Pure routing logic + overhead accounting + HTTP router baseline (no MPI, no GPU needed)."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
import pytest

from inferno.mpi_router.routing import LeastOutstanding, overhead_us
from inferno.serving.client import run_workload_routed
from inferno.workload.generator import Request


def test_least_outstanding_balances_and_breaks_ties_low():
    r = LeastOutstanding(3)
    assert [r.pick() for _ in range(3)] == [0, 1, 2]
    r.finish(1)
    assert r.pick() == 1                       # replica 1 freed up, so it is now least loaded
    assert r.outstanding == [1, 1, 1]


def test_least_outstanding_prefers_the_idle_replica():
    r = LeastOutstanding(2)
    r.pick(); r.pick(); r.pick()               # loads [2, 1]
    assert r.outstanding == [2, 1]
    assert r.pick() == 1                       # replica 1 is less loaded
    r.finish(0); r.finish(0)                   # replica 0 drains completely
    assert r.pick() == 0


def test_finish_without_pick_raises():
    with pytest.raises(RuntimeError):
        LeastOutstanding(2).finish(0)


def test_overhead_split_in_microseconds():
    ov = overhead_us({"t_dispatch": 10.000000, "t_worker_recv": 10.000250,
                      "t_worker_send": 12.0, "t_dispatcher_recv": 12.000100})
    assert ov["mpi_dispatch_us"] == pytest.approx(250.0)
    assert ov["mpi_result_us"] == pytest.approx(100.0)


def test_http_router_baseline_balances_across_replicas():
    hits = {"a": 0, "b": 0}

    def handler(request):
        hits[request.url.host] += 1
        chunk = "data: " + json.dumps({"choices": [{"text": " t"}]})
        return httpx.Response(200, content="\n".join([chunk, chunk, "data: [DONE]", ""]).encode())

    reqs = [Request(i, 0.0, 4, 2) for i in range(20)]     # simultaneous burst
    res = asyncio.run(run_workload_routed(reqs, ["http://a", "http://b"], "m",
                                          transport=httpx.MockTransport(handler)))
    assert all(r.ok for r in res)
    assert {r.replica for r in res} == {0, 1}
    assert abs(hits["a"] - hits["b"]) <= 4
