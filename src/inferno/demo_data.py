"""
Build dashboard time series from a finished campaign's per-request files. Pure function of recorded data:
nothing here simulates or invents measurements; it only re-slices what was measured onto a 1 s grid.

Inputs (written by experiments/run_adaptive.py):
    <run>/requests/<policy>__<pattern>__rep<k>.jsonl            one RequestResult row per request
    <run>/requests/<policy>__<pattern>__rep<k>.decisions.jsonl  controller ticks (adaptive only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

P99_WINDOW_SEC = 10.0
RATE_WINDOW_SEC = 5.0


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _series_for_policy(rows: list[dict], grid: np.ndarray) -> dict:
    ok = [r for r in rows if r["ok"] and r["end_time"] is not None]
    end = np.array([r["end_time"] for r in ok])
    e2e = np.array([r["e2e"] for r in ok])
    toks = np.array([r["output_tokens"] for r in ok], dtype=float)
    start = np.array([r["scheduled_time"] for r in rows])
    fin = np.array([r["end_time"] if r["end_time"] is not None else np.inf for r in rows])

    p99, tput, inflight = [], [], []
    for t in grid:
        w = (end > t - P99_WINDOW_SEC) & (end <= t)
        p99.append(float(np.percentile(e2e[w], 99)) if w.sum() >= 3 else None)
        r = (end > t - RATE_WINDOW_SEC) & (end <= t)
        tput.append(float(toks[r].sum() / RATE_WINDOW_SEC))
        inflight.append(int(((start <= t) & (fin > t)).sum()))
    return {"p99_e2e": p99, "throughput_tok_per_s": tput, "inflight": inflight}


def build_timeseries(run_dir: str | Path, pattern: str, rep: int = 0) -> dict:
    """Return {"t": [...], "arrivals": [...], "policies": {name: {...}}, "controller": {...}}."""
    req_dir = Path(run_dir) / "requests"
    per_policy: dict[str, list[dict]] = {}
    for f in sorted(req_dir.glob(f"*__{pattern}__rep{rep}.jsonl")):
        per_policy[f.name.split("__")[0]] = _load_jsonl(f)
    if not per_policy:
        raise FileNotFoundError(f"no request files for pattern={pattern} rep={rep} in {req_dir}")

    horizon = max(r["end_time"] for rows in per_policy.values() for r in rows if r["end_time"] is not None)
    grid = np.arange(0, np.ceil(horizon) + 1, 1.0)

    any_rows = next(iter(per_policy.values()))
    arr = np.array(sorted(r["scheduled_time"] for r in any_rows))    # identical stream for every policy
    arrivals = [float(((arr > t - RATE_WINDOW_SEC) & (arr <= t)).sum() / RATE_WINDOW_SEC) for t in grid]

    out = {"pattern": pattern, "rep": rep, "t": grid.tolist(), "arrivals": arrivals,
           "policies": {name: _series_for_policy(rows, grid) for name, rows in per_policy.items()},
           "controller": None}
    dec_path = req_dir / f"adaptive__{pattern}__rep{rep}.decisions.jsonl"
    if dec_path.exists():
        ticks = _load_jsonl(dec_path)
        level, i, cur = [], 0, 0
        for t in grid:
            while i < len(ticks) and ticks[i]["t"] <= t:
                cur = ticks[i]["level"]
                i += 1
            level.append(cur)
        out["controller"] = {"level": level, "events": [
            {"t": d["t"], "level": d["level"], "reason": d["reason"]} for d in ticks if d["reason"] != "hold"]}
    return out
