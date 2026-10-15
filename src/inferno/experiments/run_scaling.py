"""
Phase 4 / H2: replica scaling and coordination overhead, MPI router vs HTTP router.

For each replica count N: start N identical vLLM replicas (sharing the one GPU), then for each repetition
replay the same seeded, saturating request stream through (a) an HTTP least-outstanding router and (b) the
MPI dispatcher (mpirun -n N+1). Router order is shuffled per repetition. Reports aggregate throughput, p99,
and for MPI the directly measured per-hop overhead.

Caveat that belongs in the report: on one GPU the replicas contend for the same memory bandwidth, so
aggregate throughput is not expected to scale linearly; the informative comparison is MPI vs HTTP at equal N,
and the growth of coordination overhead with N (see also mpi_router/microbench.py for a GPU-free view).

    python -m inferno.experiments.run_scaling --config configs/experiment_config.yaml
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..hardware import capture_hardware
from ..metrics import RequestResult, summarize
from ..power import PowerSampler
from ..serving.client import run_workload_routed
from ..serving.config import configs_from_yaml
from ..serving.pool import ReplicaPool
from ..workload.generator import WorkloadGenerator
from .runner import DEFAULT_MODEL_CFG, DEFAULT_WORKLOAD_CFG, load_yaml, new_run_dir


def _result_from_row(row: dict) -> RequestResult:
    fields = RequestResult.__dataclass_fields__
    return RequestResult(**{k: v for k, v in row.items() if k in fields})


def run_mpi_router(requests, ports, model, workdir) -> tuple[list[RequestResult], list[dict]]:
    (workdir).mkdir(parents=True, exist_ok=True)
    req_file = workdir / "requests.json"
    req_file.write_text(json.dumps([asdict(r) for r in requests]))
    mpirun = shutil.which("mpirun") or "mpirun"
    cmd = [mpirun, "--map-by", ":OVERSUBSCRIBE", "-n", str(len(ports) + 1), sys.executable, "-m", "inferno.mpi_router.run_router",
           "--requests", str(req_file), "--ports", ",".join(map(str, ports)), "--model", model,
           "--out", str(workdir)]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    results = [_result_from_row(json.loads(l)) for l in (workdir / "results.jsonl").read_text().splitlines()]
    overhead = [json.loads(l) for l in (workdir / "overhead.jsonl").read_text().splitlines()]
    return results, overhead


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CFG)
    ap.add_argument("--workload-config", default=DEFAULT_WORKLOAD_CFG)
    ap.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--quantization", default="int4")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rate-multiplier", type=float, default=3.0, help="scale steady rate to saturate")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--base-port", type=int, default=8200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp_cfg, model_cfg, wl_cfg = (load_yaml(p) for p in (args.config, args.model_config, args.workload_config))
    if args.duration:
        wl_cfg["duration_sec"] = args.duration
        wl_cfg["warmup_sec"] = min(wl_cfg["warmup_sec"], args.duration / 4)
    wl_cfg["load_patterns"]["steady"]["rate_requests_per_sec"] *= args.rate_multiplier
    reps = args.reps or exp_cfg["repetitions"]
    duration, warmup, slo = wl_cfg["duration_sec"], wl_cfg["warmup_sec"], wl_cfg["slo"]
    print(f"replicas {args.replicas} x routers [http, mpi] x {reps} reps, "
          f"rate {wl_cfg['load_patterns']['steady']['rate_requests_per_sec']}/s")
    if args.dry_run:
        return

    run_dir = new_run_dir(exp_cfg.get("output_dir", "results/"), "scaling")
    (run_dir / "hardware.json").write_text(json.dumps(capture_hardware(), indent=2))
    rng = np.random.default_rng(11)
    rows: list[dict] = []

    for n in args.replicas:
        cfgs = []
        for i in range(n):
            c = configs_from_yaml(model_cfg, args.quantization, args.batch_size)
            c.port = args.base_port + i
            c.gpu_memory_utilization = round(0.72 / n, 3)
            c.tier_name = f"r{i}"
            cfgs.append(c)
        with ReplicaPool(cfgs, log_dir=run_dir / "logs") as pool:
            print(f"\n== {n} replica(s) ready ==", flush=True)
            for rep in range(reps):
                seed = wl_cfg.get("random_seed", 42) + rep
                requests = WorkloadGenerator(wl_cfg, "steady", seed=seed).generate(duration)
                for router in rng.permutation(["http", "mpi"]):
                    overhead = []
                    t0 = time.monotonic()
                    with PowerSampler() as power:
                        if router == "http":
                            results = asyncio.run(run_workload_routed(requests, pool.urls, pool.models[0]))
                        else:
                            results, overhead = run_mpi_router(
                                requests, [c.port for c in cfgs], pool.models[0], run_dir / "mpi" / f"n{n}_rep{rep}")
                    wall = time.monotonic() - t0
                    stats = summarize(results, wall_time_sec=duration - warmup, slo=slo, warmup_sec=warmup)
                    stats.update(power.summary(output_tokens=stats["output_tokens"]))
                    row = {"router": router, "replicas": n, "rep": rep, "seed": seed, "run_wall_sec": wall, **stats}
                    if overhead:
                        for hop in ("mpi_dispatch_us", "mpi_result_us"):
                            v = np.array([o[hop] for o in overhead])
                            row[f"{hop}_p50"], row[f"{hop}_p99"] = float(np.percentile(v, 50)), float(np.percentile(v, 99))
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(run_dir / "scaling.csv", index=False)   # rewrite: columns may grow
                    print(f"  n={n} rep{rep} {router:4s} thr={row['throughput_tok_per_s']:.0f} tok/s "
                          f"p99={row['e2e_p99']:.2f}s errors={row['n_errors']}", flush=True)
    print(f"\nWrote {run_dir / 'scaling.csv'}")


if __name__ == "__main__":
    main()
