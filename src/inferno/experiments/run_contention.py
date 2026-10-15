"""
Mechanism study: what does putting a second replica on the SAME GPU cost?

The campaign compares routing policies over co-resident replicas. If policies that spread traffic across both
replicas do worse than one replica alone, this experiment says why, by holding total load fixed and varying only
how the GPU is shared:

    single        1 replica, the GPU's memory to itself (gpu_memory_utilization 0.70)
    split-idle    2 replicas (0.36 each), ALL traffic to replica 0; replica 1 sits idle but resident
    split-active  2 replicas (0.36 each), traffic balanced across both (least-loaded)

Comparisons:
    split-idle vs single    cost of halving the memory budget (smaller KV cache), no contention
    split-active vs split-idle    cost of two processes actively time-slicing the GPU

Load is a fraction rho of the measured single-replica capacity (results/calibration.json), so it is meaningful on
any GPU. Output uses the same summary.csv schema as the campaign, with pattern = "rho<value>", so the report and
statistics tools apply unchanged (paired by repetition seed).

    python -m inferno.experiments.run_contention --config configs/experiment_config.yaml
"""
import argparse
import asyncio
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np

from ..controller.live import run_policy_workload
from ..controller.routing_policies import LeastLoadedPolicy, PinnedPolicy
from ..hardware import capture_hardware
from ..metrics import summarize
from ..power import PowerSampler
from ..serving.config import pool_from_yaml
from ..serving.pool import ReplicaPool
from ..workload.generator import WorkloadGenerator
from .runner import DEFAULT_MODEL_CFG, DEFAULT_WORKLOAD_CFG, load_yaml, new_run_dir

DEFAULT_POOL_CFG = "configs/pool_config.yaml"


def _tier_cfg(model_cfg, pool_cfg, port, util, name):
    cfg = pool_from_yaml(model_cfg, pool_cfg)[-1]          # the "bulk" engine settings for every replica
    cfg.port, cfg.gpu_memory_utilization, cfg.tier_name = port, util, name
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CFG)
    ap.add_argument("--workload-config", default=DEFAULT_WORKLOAD_CFG)
    ap.add_argument("--pool-config", default=DEFAULT_POOL_CFG)
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.3, 0.6])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--duration", type=float, default=75.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp_cfg, model_cfg, wl_cfg, pool_cfg = (load_yaml(p) for p in (
        args.config, args.model_config, args.workload_config, args.pool_config))
    cal = json.loads(Path(args.calibration).read_text())
    cap_rps = cal["measured"]["steady"]["requests_per_s"]
    wl_cfg["duration_sec"] = args.duration
    wl_cfg["warmup_sec"] = min(wl_cfg["warmup_sec"], args.duration / 6)
    duration, warmup, slo = args.duration, wl_cfg["warmup_sec"], wl_cfg["slo"]

    runs = len(args.rhos) * args.reps * 3
    print(f"capacity {cap_rps:.2f} req/s; rhos {args.rhos}; {runs} runs, ~{runs * (duration + 25) / 3600:.1f} h")
    if args.dry_run:
        return

    run_dir = new_run_dir(exp_cfg.get("output_dir", "results/"), "contention")
    (run_dir / "hardware.json").write_text(json.dumps(capture_hardware(), indent=2))
    rng = np.random.default_rng(5)
    rows, csv_path = [], run_dir / "summary.csv"

    def run_condition(name, pool, policy_factory):
        for rho in args.rhos:
            cfg = copy.deepcopy(wl_cfg)
            cfg["load_patterns"]["steady"]["rate_requests_per_sec"] = rho * cap_rps
            for rep in range(args.reps):
                seed = wl_cfg.get("random_seed", 42) + rep
                reqs = WorkloadGenerator(cfg, "steady", seed=seed).generate(duration)
                t0 = time.monotonic()
                with PowerSampler() as power:
                    results, _ = asyncio.run(run_policy_workload(
                        reqs, pool.urls, pool.models, pool.capacities, policy_factory(), tick_sec=1.0))
                wall = time.monotonic() - t0
                stats = summarize(results, wall_time_sec=duration - warmup, slo=slo, warmup_sec=warmup)
                stats.update(power.summary(output_tokens=stats["output_tokens"]))
                row = {"config": name, "pattern": f"rho{rho}", "rep": rep, "seed": seed, "rho": rho,
                       "run_wall_sec": wall, **stats}
                rows.append(row)
                with open(csv_path, "w", newline="") as f:
                    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in row, k))
                    w = csv.DictWriter(f, fieldnames=keys, restval="")
                    w.writeheader()
                    w.writerows(rows)
                print(f"  {name:13s} rho={rho} rep{rep}: p99={row['e2e_p99']:.2f}s "
                      f"thr={row['throughput_tok_per_s']:.0f} tok/s ttft_p50={row['ttft_p50']:.2f}s", flush=True)

    print("\n== single replica ==", flush=True)
    with ReplicaPool([_tier_cfg(model_cfg, pool_cfg, 8400, 0.70, "single")], log_dir=run_dir / "logs") as pool:
        _warm(pool)
        run_condition("single", pool, lambda: PinnedPolicy(0))

    print("\n== two replicas, split memory ==", flush=True)
    two = [_tier_cfg(model_cfg, pool_cfg, 8410 + i, 0.36, f"r{i}") for i in range(2)]
    with ReplicaPool(two, log_dir=run_dir / "logs") as pool:
        _warm(pool)
        order = list(rng.permutation(["split-idle", "split-active"]))
        for name in order:
            run_condition(name, pool, (lambda: PinnedPolicy(0)) if name == "split-idle" else (lambda: LeastLoadedPolicy()))
    print(f"\nWrote {csv_path}")


def _warm(pool: ReplicaPool) -> None:
    from ..serving.client import run_workload
    from ..workload.generator import Request

    for url, model in zip(pool.urls, pool.models):
        asyncio.run(run_workload([Request(i, i * 0.05, 64, 32) for i in range(8)], url, model))


if __name__ == "__main__":
    main()
