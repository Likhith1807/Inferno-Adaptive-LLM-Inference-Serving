"""Open-loop client against an in-memory mock of an OpenAI-compatible streaming server."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from inferno.serving.client import make_prompt, run_workload
from inferno.workload.generator import Request


def _sse(n_tokens: int, prompt_tokens: int, include_usage=True) -> bytes:
    lines = []
    for i in range(n_tokens):
        lines.append("data: " + json.dumps({"choices": [{"text": f" t{i}"}]}))
        lines.append("")
    if include_usage:
        lines.append("data: " + json.dumps(
            {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_tokens}}))
        lines.append("")
    lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def _transport(handler):
    return httpx.MockTransport(handler)


def test_successful_stream_records_tokens_and_ttft():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["ignore_eos"] is True
        return httpx.Response(200, content=_sse(body["max_tokens"], 7))

    reqs = [Request(0, 0.0, 5, 4), Request(1, 0.05, 5, 6)]
    results = asyncio.run(run_workload(reqs, "http://mock", "m", transport=_transport(handler)))

    assert [r.ok for r in results] == [True, True]
    assert [r.output_tokens for r in results] == [4, 6]
    assert all(r.prompt_tokens == 7 for r in results)          # taken from server usage
    assert all(r.ttft is not None and r.ttft >= 0 for r in results)


def test_falls_back_to_chunk_count_without_usage():
    handler = lambda request: httpx.Response(200, content=_sse(5, 0, include_usage=False))
    (r,) = asyncio.run(run_workload([Request(0, 0.0, 5, 5)], "http://mock", "m", transport=_transport(handler)))
    assert r.ok and r.output_tokens == 5


def test_http_error_is_recorded_not_raised():
    handler = lambda request: httpx.Response(500, content=b"boom")
    (r,) = asyncio.run(run_workload([Request(0, 0.0, 5, 5)], "http://mock", "m", transport=_transport(handler)))
    assert not r.ok and "HTTPStatusError" in r.error


def test_arrivals_are_open_loop_paced():
    starts = []

    def handler(request):
        return httpx.Response(200, content=_sse(2, 3))

    reqs = [Request(i, i * 0.1, 3, 2) for i in range(4)]
    results = asyncio.run(run_workload(reqs, "http://mock", "m", transport=_transport(handler)))
    starts = [r.start_time for r in results]
    assert starts == sorted(starts)
    assert starts[-1] >= 0.29          # last request held until t=0.3


def test_make_prompt_length():
    assert len(make_prompt(50).split()) == 50
    assert len(make_prompt(0).split()) == 1


def test_drain_timeout_cancels_stragglers_and_records_failures():
    async def handler(request):
        body = json.loads(request.content)
        if body["max_tokens"] > 100:            # a "slow" request that will not finish within the drain window
            await asyncio.sleep(30)
        return httpx.Response(200, content=_sse(body["max_tokens"], 5))

    reqs = [Request(0, 0.0, 5, 4), Request(1, 0.0, 5, 500), Request(2, 0.05, 5, 4)]
    results = asyncio.run(run_workload(reqs, "http://mock", "m", transport=_transport(handler),
                                       drain_timeout_sec=0.5))
    by_id = {r.request_id: r for r in results}
    assert by_id[0].ok and by_id[2].ok
    assert not by_id[1].ok and by_id[1].error == "drain_timeout" and by_id[1].output_tokens == 0
