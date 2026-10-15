"""Dashboard time-series builder: re-slices recorded request files; verified on files written by the test."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from inferno.demo_data import build_timeseries


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _row(i, sched, e2e, tokens=10, ok=True):
    return {"request_id": i, "scheduled_time": sched, "start_time": sched, "first_token_time": sched + 0.1,
            "end_time": sched + e2e, "prompt_tokens": 50, "output_tokens": tokens, "ok": ok, "e2e": e2e,
            "ttft": 0.1, "tpot": 0.01, "error": None, "replica": 0, "route_overhead_us": None}


def test_timeseries_from_recorded_files(tmp_path):
    req = tmp_path / "requests"
    req.mkdir()
    rows = [_row(i, float(i), 1.0 + (i % 3)) for i in range(20)]
    _write(req / "adaptive__bursty__rep0.jsonl", rows)
    _write(req / "fixed-fast__bursty__rep0.jsonl", [_row(i, float(i), 3.0) for i in range(20)])
    _write(req / "adaptive__bursty__rep0.decisions.jsonl", [
        {"t": 2.0, "level": 1, "reason": "escalate: x"}, {"t": 5.0, "level": 1, "reason": "hold"},
        {"t": 12.0, "level": 0, "reason": "de-escalate: y"}])

    ts = build_timeseries(tmp_path, "bursty")
    assert set(ts["policies"]) == {"adaptive", "fixed-fast"}
    n = len(ts["t"])
    assert all(len(s["p99_e2e"]) == n for s in ts["policies"].values())
    assert ts["policies"]["fixed-fast"]["p99_e2e"][15] == pytest.approx(3.0)      # constant latency 3 s
    assert ts["policies"]["fixed-fast"]["p99_e2e"][0] is None                      # too few completions yet
    assert ts["arrivals"][10] == pytest.approx(1.0)                                # one request per second
    assert ts["controller"]["level"][3] == 1 and ts["controller"]["level"][13] == 0
    assert [e["t"] for e in ts["controller"]["events"]] == [2.0, 12.0]             # 'hold' ticks omitted


def test_missing_data_raises(tmp_path):
    (tmp_path / "requests").mkdir()
    with pytest.raises(FileNotFoundError):
        build_timeseries(tmp_path, "steady")
