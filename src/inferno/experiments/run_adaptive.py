"""
Phases 5 and 7: adaptive routing vs every fixed baseline, in one shared replica pool.

Policies compared (all with the same pre-loaded pool, same seeded request streams per repetition):
    fixed-fast    all traffic to tier 0            (low-latency decode tier)
    fixed-bulk    all traffic to the last tier     (throughput / long-prefill tier)
    static-lb     least-loaded across tiers        (non-adaptive multi-replica baseline)
    adaptive      AdaptiveTierPolicy               (load- and length-aware, hysteresis)

Run order inside each (pattern, repetition) is shuffled with a recorded seed, so slow drift (thermal
throttling of a laptop GPU, page cache) cannot systematically favour one policy.

    python -m inferno.experiments.run_adaptive --config configs/experiment_config.yaml
    python -m inferno.experiments.run_adaptive --config configs/experiment_config.yaml --dry-run
"""
import argparse
import asyncio
import csv
import json
import time

import numpy as np

from ..controller.live import run_policy_workload
from ..controller.routing_policies import (AdaptiveConfig, AdaptiveTierPolicy, LeastLoadedPolicy,
                                           PinnedPolicy)
from ..hardware import capture_hardware
from ..metrics import summarize
from ..power import PowerSampler
from ..serving.config import pool_from_yaml
from ..serving.pool import ReplicaPool
from ..workload.generator import WorkloadGenerator
from .runner import DEFAULT_MODEL_CFG, DEFAULT_WORKLOAD_CFG, load_yaml, new_run_dir

DEFAULT_POOL_CFG = "configs/pool_config.yaml"


def make_policies(pool_cfg: dict, n_tiers: int) -> list:
    a = pool_cfg.get("adaptive", {})
    cfg = AdaptiveConfig(**{k: v for k, v in a.items() if k != "tick_sec"})
    return [PinnedPolicy(0, "fixed-fast"), PinnedPolicy(n_tiers - 1, "fixed-bulk"), LeastLoadedPolicy(),
            AdaptiveTierPolicy(cfg=cfg)]


def run_campaign(args_ns) -> None:
    exp_cfg, model_cfg, wl_cfg, pool_cfg = (load_yaml(p) for p in (
        args_ns.config, args_ns.model_config, args_ns.workload_config, args_ns.pool_config))
    if args_ns.duration:
        wl_cfg["duration_sec"] = args_ns.duration
        wl_cfg["warmup_sec"] = min(wl_cfg["warmup_sec"], args_ns.duration / 4)
    reps = args_ns.reps or exp_cfg["repetitions"]
    patterns = args_ns.patterns or exp_cfg["load_patterns_to_test"]
    tiers = pool_from_yaml(model_cfg, pool_cfg)
    policies = make_policies(pool_cfg, len(tiers))
    duration, warmup, slo = wl_cfg["duration_sec"], wl_cfg["warmup_sec"], wl_cfg["slo"]
    tick_sec = pool_cfg.get("adaptive", {}).get("tick_sec", 1.0)

    runs = len(patterns) * reps * len(policies)
    print(f"{len(tiers)} tiers, {len(policies)} policies x {len(patterns)} patterns x {reps} reps = {runs} runs, "
          f"~{runs * (duration + 15) / 3600:.1f} h")
    if args_ns.dry_run:
        return

    run_dir = new_run_dir(exp_cfg.get("output_dir", "results/"), args_ns.name)
    (run_dir / "hardware.json").write_text(json.dumps(capture_hardware(), indent=2))
    (run_dir / "config_snapshot.json").write_text(json.dumps(
        {"workload": wl_cfg, "pool": pool_cfg, "tiers": [c.__dict__ for c in tiers], "reps": reps,
         "patterns": patterns, "order_seed": args_ns.order_seed}, indent=2, default=str))
    order_rng = np.random.default_rng(args_ns.order_seed)
    writer = csv_file = None

    with ReplicaPool(tiers, log_dir=run_dir / "logs") as pool:
        print(f"pool ready (startup {['%.0fs' % s for s in pool.startup_sec]})", flush=True)
        _warm(pool)
        for pattern in patterns:
            for rep in range(reps):
                seed = wl_cfg.get("random_seed", 42) + rep
                requests = WorkloadGenerator(wl_cfg, pattern, seed=seed).generate(duration)
                for pol_idx in order_rng.permutation(len(policies)):
                    policy = make_policies(pool_cfg, len(tiers))[pol_idx]     # fresh state each run
                    t_start = time.monotonic()
                    with PowerSampler() as power:
                        results, decisions = asyncio.run(run_policy_workload(
                            requests, pool.urls, pool.models, pool.capacities, policy, tick_sec=tick_sec))
                    wall = time.monotonic() - t_start
                    stats = summarize(results, wall_time_sec=duration - warmup, slo=slo, warmup_sec=warmup)
                    stats.update(power.summary(output_tokens=stats["output_tokens"]))
                    tier_share = np.bincount([r.replica for r in results], minlength=len(tiers)) / max(1, len(results))
                    row = {"config": policy.name, "pattern": pattern, "rep": rep, "seed": seed,
                           "run_wall_sec": wall, **{f"share_tier{i}": float(s) for i, s in enumerate(tier_share)},
                           **stats}
                    if writer is None:
                        csv_file = open(run_dir / "summary.csv", "w", newline="")
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_file.flush()
                    stem = f"{policy.name}__{pattern}__rep{rep}"
                    with open(run_dir / "requests" / f"{stem}.jsonl", "w") as f:
                        f.writelines(json.dumps(r.to_row()) + "\n" for r in results)
                    if decisions:
                        with open(run_dir / "requests" / f"{stem}.decisions.jsonl", "w") as f:
                            f.writelines(json.dumps(d) + "\n" for d in decisions)
                    print(f"  {pattern:12s} rep{rep} {policy.name:11s} p99={row['e2e_p99']:.2f}s "
                          f"thr={row['throughput_tok_per_s']:.0f} tok/s errors={row['n_errors']}", flush=True)
    if csv_file:
        csv_file.close()
    print(f"\nWrote {run_dir / 'summary.csv'}")


def _warm(pool: ReplicaPool) -> None:
    from ..serving.client import run_workload
    from ..workload.generator import Request

    for url, model in zip(pool.urls, pool.models):
        asyncio.run(run_workload([Request(i, i * 0.05, 64, 32) for i in range(8)], url, model))


def build_parser(default_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="experiment_config.yaml")
    p.add_argument("--model-config", default=DEFAULT_MODEL_CFG)
    p.add_argument("--workload-config", default=DEFAULT_WORKLOAD_CFG)
    p.add_argument("--pool-config", default=DEFAULT_POOL_CFG)
    p.add_argument("--reps", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--patterns", nargs="+")
    p.add_argument("--order-seed", type=int, default=7)
    p.add_argument("--name", default=default_name)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    run_campaign(build_parser("adaptive").parse_args())


if __name__ == "__main__":
    main()
