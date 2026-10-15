"""
Calibrate offered load to *this machine's measured capacity*, instead of guessing request rates.

A benchmark whose "steady" load already saturates the GPU (queueing, 0% SLO attainment) says nothing about
policy quality. So: saturate one reference replica, measure its sustainable throughput for each length profile,
convert that to requests/s, and derive every load pattern as a fraction of capacity:

    steady        rho * capacity                                   (default rho = 0.60)
    bursty        base 0.35 * capacity, bursts at 1.5 * capacity   (bursts exceed capacity by design)
    mixed_length  rho * capacity of the *mixed* length profile

Writes configs/workload_config.local.yaml (auto-used by every experiment when present) and
results/calibration.json (the raw measurements).

    python -m inferno.experiments.calibrate --config configs/experiment_config.yaml
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from ..metrics import RequestResult
from ..serving.client import run_workload
from ..serving.config import pool_from_yaml
from ..serving.pool import ReplicaPool
from ..workload.generator import Request, WorkloadGenerator
from .runner import DEFAULT_MODEL_CFG, DEFAULT_WORKLOAD_CFG, load_yaml

LOCAL_WORKLOAD_CFG = "configs/workload_config.local.yaml"
DEFAULT_POOL_CFG = "configs/pool_config.yaml"


def mean_output_tokens(wl_cfg: dict, pattern: str, n: int = 20000, seed: int = 0) -> float:
    """Expected output tokens per request under a pattern's length profile (after caps)."""
    gen = WorkloadGenerator(wl_cfg, pattern, seed=seed)
    _, out = gen._sample_lengths(n)
    return float(np.mean(out))


def capacity_rps(results: list[RequestResult], window: tuple[float, float], mean_out_tokens: float) -> dict:
    """Sustainable capacity from an overloaded run: tokens completed inside [t0, t1] per second."""
    t0, t1 = window
    done = [r for r in results if r.ok and r.end_time is not None and t0 <= r.end_time <= t1]
    tok_per_s = sum(r.output_tokens for r in done) / (t1 - t0)
    return {"tokens_per_s": tok_per_s, "requests_per_s": tok_per_s / mean_out_tokens,
            "completed_in_window": len(done)}


def derive_rates(cap_steady_rps: float, cap_mixed_rps: float, rho: float = 0.60,
                 base_frac: float = 0.35, burst_frac: float = 1.5) -> dict:
    """Pure function: request rates for each load pattern from measured capacities."""
    return {
        "steady": {"rate_requests_per_sec": round(rho * cap_steady_rps, 3)},
        "bursty": {"base_rate_requests_per_sec": round(base_frac * cap_steady_rps, 3),
                   "burst_rate_requests_per_sec": round(burst_frac * cap_steady_rps, 3)},
        "mixed_length": {"rate_requests_per_sec": round(rho * cap_mixed_rps, 3)},
    }


def _saturate(base_url, model, wl_cfg, pattern, seconds, warm, overload_rate) -> dict:
    cfg = copy.deepcopy(wl_cfg)
    cfg["load_patterns"][pattern]["rate_requests_per_sec"] = overload_rate
    reqs = WorkloadGenerator(cfg, pattern, seed=123).generate(seconds)
    results = asyncio.run(run_workload(reqs, base_url, model, request_timeout_sec=seconds * 4))
    m = mean_output_tokens(wl_cfg, pattern)
    cap = capacity_rps(results, (warm, seconds), m)
    cap["mean_output_tokens"] = m
    return cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment_config.yaml (for output_dir)")
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CFG)
    ap.add_argument("--workload-config", default=DEFAULT_WORKLOAD_CFG)
    ap.add_argument("--pool-config", default=DEFAULT_POOL_CFG)
    ap.add_argument("--rho", type=float, default=0.60, help="steady utilization target")
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--warm", type=float, default=15.0)
    ap.add_argument("--overload-rate", type=float, default=30.0, help="req/s, far above capacity")
    args = ap.parse_args()

    model_cfg, wl_cfg, pool_cfg = (load_yaml(p) for p in (args.model_config, args.workload_config, args.pool_config))
    # Reference replica = the highest-capacity tier of the pool, given the whole GPU to itself.
    ref = pool_from_yaml(model_cfg, pool_cfg)[-1]
    ref.gpu_memory_utilization = 0.70
    ref.port = 8300
    print(f"calibrating on {ref.label} ({ref.model_name}), bs={ref.batch_size}")

    measured = {}
    with ReplicaPool([ref]) as pool:
        asyncio.run(run_workload([Request(i, i * 0.05, 64, 32) for i in range(8)], pool.urls[0], pool.models[0]))
        for pattern in ("steady", "mixed_length"):
            measured[pattern] = _saturate(pool.urls[0], pool.models[0], wl_cfg, pattern, args.seconds, args.warm,
                                          args.overload_rate)
            print(f"  {pattern:13s} capacity {measured[pattern]['tokens_per_s']:.0f} tok/s = "
                  f"{measured[pattern]['requests_per_s']:.2f} req/s")

    rates = derive_rates(measured["steady"]["requests_per_s"], measured["mixed_length"]["requests_per_s"], args.rho)
    local = copy.deepcopy(wl_cfg)
    for pattern, params in rates.items():
        local["load_patterns"][pattern].update(params)
    local["calibration"] = {"reference": ref.label, "rho": args.rho, "measured": measured,
                            "note": "rates derived from measured capacity by experiments/calibrate.py"}
    Path(LOCAL_WORKLOAD_CFG).write_text(yaml.safe_dump(local, sort_keys=False))
    Path("results").mkdir(exist_ok=True)
    Path("results/calibration.json").write_text(json.dumps({"measured": measured, "rates": rates,
                                                            "reference": ref.label}, indent=2))
    print(f"\nwrote {LOCAL_WORKLOAD_CFG}\nrates: {json.dumps(rates)}")


if __name__ == "__main__":
    main()
