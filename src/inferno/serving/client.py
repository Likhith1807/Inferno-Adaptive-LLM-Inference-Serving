"""
Open-loop load client for any OpenAI-compatible /v1/completions endpoint (vLLM, TGI, ...).

Open-loop means requests are sent at their scheduled arrival times regardless of
how many are still in flight, which is how real traffic behaves. A closed-loop
client (wait for each response) would hide queueing and flatter the server.

Output length is pinned with `ignore_eos` + `max_tokens`, so every configuration
generates the same number of tokens per request and results are comparable.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx

from ..metrics import RequestResult
from ..workload.generator import Request

_WORDS = (
    "system data model batch queue latency memory kernel thread cache token "
    "network compute vector matrix device stream buffer request response cluster "
    "schedule policy metric signal window sample layer weight tensor block page"
).split()


def make_prompt(n_tokens: int) -> str:
    """Deterministic filler prompt of roughly n_tokens tokens (~1 token per word).
    Exact counts are recovered from the server's usage report."""
    return " ".join(_WORDS[i % len(_WORDS)] for i in range(max(1, n_tokens)))


async def send_request(
    client: httpx.AsyncClient, url: str, model: str, req: Request, t0: float
) -> RequestResult:
    scheduled = req.arrival_time
    start = time.monotonic() - t0
    first_token = None
    n_chunks = 0
    prompt_tokens = req.prompt_tokens
    completion_tokens = None
    payload = {
        "model": model,
        "prompt": make_prompt(req.prompt_tokens),
        "max_tokens": req.max_new_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                    completion_tokens = chunk["usage"].get("completion_tokens")
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if first_token is None:
                        first_token = time.monotonic() - t0
                    n_chunks += 1
        end = time.monotonic() - t0
        return RequestResult(
            request_id=req.request_id, scheduled_time=scheduled, start_time=start,
            first_token_time=first_token, end_time=end, prompt_tokens=prompt_tokens,
            output_tokens=completion_tokens if completion_tokens is not None else n_chunks,
            ok=first_token is not None,
            error=None if first_token is not None else "no tokens received",
        )
    except Exception as exc:  # network error, HTTP error, malformed stream
        return RequestResult(
            request_id=req.request_id, scheduled_time=scheduled, start_time=start,
            first_token_time=first_token, end_time=None, prompt_tokens=prompt_tokens,
            output_tokens=0, ok=False, error=f"{type(exc).__name__}: {exc}",
        )


async def collect_with_drain(pairs, t0: float, drain_timeout_sec: float | None) -> list[RequestResult]:
    """
    Await all request tasks; if `drain_timeout_sec` is set, give unfinished requests at most that long after the
    last arrival, then cancel them and record each as a failure ("drain_timeout"). Bounds the run time of
    configurations whose capacity is far below the offered load, while keeping the loss visible in n_errors.
    """
    tasks = [t for _, t in pairs]
    if drain_timeout_sec is None:
        return list(await asyncio.gather(*tasks))
    done, pending = await asyncio.wait(tasks, timeout=drain_timeout_sec)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    out = []
    for req, t in pairs:
        if t in pending:
            out.append(RequestResult(request_id=req.request_id, scheduled_time=req.arrival_time,
                                     start_time=req.arrival_time, first_token_time=None, end_time=None,
                                     prompt_tokens=req.prompt_tokens, output_tokens=0, ok=False,
                                     error="drain_timeout"))
        else:
            out.append(t.result())
    return out


async def run_workload(
    requests: list[Request],
    base_url: str,
    model: str,
    request_timeout_sec: float = 300.0,
    transport: httpx.AsyncBaseTransport | None = None,
    drain_timeout_sec: float | None = None,
) -> list[RequestResult]:
    """Replay `requests` open-loop against the server; returns one result per request.

    With `drain_timeout_sec`, requests still unfinished that long after the last arrival are cancelled and
    recorded as failures."""
    url = base_url.rstrip("/") + "/v1/completions"
    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=256)
    async with httpx.AsyncClient(
        timeout=request_timeout_sec, limits=limits, transport=transport
    ) as client:
        t0 = time.monotonic()
        tasks = []
        for req in sorted(requests, key=lambda r: r.arrival_time):
            # asyncio.sleep can wake slightly early relative to time.monotonic()
            # (coarse timer resolution on Windows), which would send a request
            # before its scheduled time and yield a negative TTFT. Loop until due.
            while (delay := req.arrival_time - (time.monotonic() - t0)) > 0:
                await asyncio.sleep(delay)
            tasks.append((req, asyncio.create_task(send_request(client, url, model, req, t0))))
        return await collect_with_drain(tasks, t0, drain_timeout_sec)


async def run_workload_routed(
    requests: list[Request],
    base_urls: list[str],
    model: str,
    request_timeout_sec: float = 300.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[RequestResult]:
    """
    HTTP least-outstanding-requests router baseline: same open-loop replay, but each
    request goes directly to the replica with the fewest in-flight requests. No MPI,
    no extra hop. This is what MPI routing is compared against in H2.
    """
    urls = [u.rstrip("/") + "/v1/completions" for u in base_urls]
    inflight = [0] * len(urls)
    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=256)

    async def routed(client, idx, req, t0):
        try:
            res = await send_request(client, urls[idx], model, req, t0)
        finally:
            inflight[idx] -= 1
        res.replica = idx
        return res

    async with httpx.AsyncClient(timeout=request_timeout_sec, limits=limits, transport=transport) as client:
        t0 = time.monotonic()
        tasks = []
        for req in sorted(requests, key=lambda r: r.arrival_time):
            while (delay := req.arrival_time - (time.monotonic() - t0)) > 0:
                await asyncio.sleep(delay)
            idx = min(range(len(urls)), key=lambda i: inflight[i])
            inflight[idx] += 1
            tasks.append(asyncio.create_task(routed(client, idx, req, t0)))
        return list(await asyncio.gather(*tasks))
