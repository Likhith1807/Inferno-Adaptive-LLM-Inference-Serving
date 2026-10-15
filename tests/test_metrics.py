"""Metrics math: TTFT / TPOT / e2e, warmup exclusion, SLO goodput."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math

from inferno.metrics import RequestResult, summarize


def _r(i, sched, first, end, out=11, ok=True):
    return RequestResult(i, sched, sched, first, end, 10, out, ok)


def test_request_derived_latencies():
    r = _r(0, sched=1.0, first=1.5, end=2.5, out=11)
    assert math.isclose(r.ttft, 0.5)
    assert math.isclose(r.e2e, 1.5)
    assert math.isclose(r.tpot, 0.1)   # (2.5-1.5)/(11-1)


def test_tpot_undefined_for_single_token():
    assert _r(0, 0.0, 0.2, 0.2, out=1).tpot is None


def test_summary_excludes_warmup_and_errors():
    results = [
        _r(0, 0.0, 9.0, 9.5),                    # inside warmup: excluded
        _r(1, 10.0, 10.2, 11.2),
        _r(2, 11.0, 11.4, 12.4),
        _r(3, 12.0, None, None, out=0, ok=False),
    ]
    s = summarize(results, wall_time_sec=10.0, warmup_sec=5.0)
    assert s["n_requests"] == 3 and s["n_ok"] == 2 and s["n_errors"] == 1
    assert math.isclose(s["throughput_tok_per_s"], 22 / 10.0)   # 22 tokens over max(10 s window, drain to t=12 -> 7 s)


def test_goodput_counts_only_requests_meeting_slo():
    slo = {"ttft_sec": 0.3, "tpot_sec": 0.2}
    results = [
        _r(0, 0.0, 0.2, 1.2),    # ttft 0.2 ok, tpot 0.1 ok  -> meets SLO
        _r(1, 0.0, 0.5, 1.5),    # ttft 0.5 > 0.3            -> misses
        _r(2, 0.0, 0.1, 3.1),    # tpot 0.3 > 0.2            -> misses
    ]
    s = summarize(results, wall_time_sec=1.0, slo=slo)
    assert math.isclose(s["slo_attainment"], 1 / 3)
    assert math.isclose(s["goodput_req_per_s"], 1 / 3.1)      # busy span = 3.1 s (last completion), not the 1 s window


def test_empty_run_does_not_crash():
    s = summarize([], wall_time_sec=10.0, slo={"ttft_sec": 1, "tpot_sec": 1})
    assert s["n_requests"] == 0 and math.isnan(s["e2e_p99"])


def test_throughput_includes_drain_time_after_arrivals_stop():
    # Everything arrives inside a 10 s window but the last request only finishes at t=30: a policy that drains
    # slowly must not report the offered load.
    results = [_r(i, float(i), float(i) + 0.1, float(i) + 1.0, out=100) for i in range(10)]
    results.append(_r(10, 9.0, 9.5, 30.0, out=100))
    s = summarize(results, wall_time_sec=10.0, warmup_sec=0.0)
    assert math.isclose(s["offered_tok_per_s"], 1100 / 10.0)
    assert math.isclose(s["throughput_tok_per_s"], 1100 / 30.0)
    assert s["busy_span_sec"] == 30.0
