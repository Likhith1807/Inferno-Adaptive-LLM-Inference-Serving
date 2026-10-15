"""
Shared experiment runner used by run_baseline / run_sweep.

For every (serving config) cell it starts vLLM once, then for each load pattern
and repetition replays an open-loop workload, samples GPU power, and appends one
summary row to summary.csv. The workload seed depends only on the repetition
index, so every configuration sees the *same* request stream in repetition k.
That pairing is what makes the later paired significance tests valid.

Output layout:
    results/<run_id>/hardware.json
    results/<run_id>/config_snapshot.json
    results/<run_id>/summary.csv
    results/<run_id>/requests/<label>__<pattern>__rep<k>.jsonl
    results/<run_id>/logs/vllm_<label>.log
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from ..hardware import capture_hardware
from ..metrics import summarize
from ..power import PowerSampler
from ..serving.client import run_workload
from ..serving.config import ServingConfig, configs_from_yaml
from ..serving.harness import InferenceHarness
from ..workload.generator import WorkloadGenerator

DEFAULT_MODEL_CFG = "configs/model_config.yaml"
_BASE_WORKLOAD_CFG = "configs/workload_config.yaml"
_LOCAL_WORKLOAD_CFG = "configs/workload_config.local.yaml"     # written by experiments/calibrate.py
# Prefer the machine-calibrated workload (rates derived from measured capacity) when it exists.
DEFAULT_WORKLOAD_CFG = _LOCAL_WORKLOAD_CFG if Path(_LOCAL_WORKLOAD_CFG).exists() else _BASE_WORKLOAD_CFG


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def new_run_dir(out_root: str | Path, name: str) -> Path:
    run_dir = Path(out_root) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (run_dir / "requests").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    return run_dir


def plan_summary(cells: list[ServingConfig], patterns: list[str], reps: int, wl_cfg: dict) -> str:
    per_run = wl_cfg["duration_sec"] + 15   # duration + drain/overhead allowance
    runs = len(cells) * len(patterns) * reps
    startup = len(cells) * 90                # rough vLLM cold-start allowance
    total_h = (runs * per_run + startup) / 3600
    lines = [f"{len(cells)} configs x {len(patterns)} patterns x {reps} reps = {runs} runs",
             f"estimated wall time: ~{total_h:.1f} h (duration {wl_cfg['duration_sec']}s/run)"]
    lines += [f"  - {c.label}  ({c.model_name})" for c in cells]
    return "\n".join(lines)


def run_cells(
    cells: list[ServingConfig],
    patterns: list[str],
    reps: int,
    wl_cfg: dict,
    run_dir: Path,
    extra_meta: dict | None = None,
) -> Path:
    (run_dir / "hardware.json").write_text(json.dumps(capture_hardware(), indent=2))
    (run_dir / "config_snapshot.json").write_text(
        json.dumps({"workload": wl_cfg, "cells": [c.__dict__ for c in cells],
                    "patterns": patterns, "reps": reps, **(extra_meta or {})}, indent=2, default=str)
    )
    summary_path = run_dir / "summary.csv"
    writer = None
    csv_file = None
    duration, warmup, slo = wl_cfg["duration_sec"], wl_cfg["warmup_sec"], wl_cfg["slo"]

    for cell in cells:
        print(f"\n=== {cell.label}: starting server ({cell.model_name}) ===", flush=True)
        with InferenceHarness(cell, log_dir=run_dir / "logs") as harness:
            cold_start = harness.last_reload_sec
            print(f"    ready in {cold_start:.1f}s", flush=True)
            _warm_up(cell)
            for pattern in patterns:
                for rep in range(reps):
                    seed = wl_cfg.get("random_seed", 42) + rep
                    requests = WorkloadGenerator(wl_cfg, pattern, seed=seed).generate(duration)
                    t_start = time.monotonic()
                    with PowerSampler() as power:
                        results = asyncio.run(run_workload(requests, cell.base_url, cell.model_name,
                                                         drain_timeout_sec=wl_cfg.get("drain_timeout_sec")))
                    wall = time.monotonic() - t_start

                    stats = summarize(results, wall_time_sec=duration - warmup, slo=slo, warmup_sec=warmup)
                    stats.update(power.summary(output_tokens=stats["output_tokens"]))
                    row = {"config": cell.label, "quantization": cell.quantization,
                           "batch_size": cell.batch_size, "pattern": pattern, "rep": rep,
                           "seed": seed, "cold_start_sec": cold_start, "run_wall_sec": wall, **stats}
                    if writer is None:
                        csv_file = open(summary_path, "w", newline="")
                        writer = csv.DictWriter(csv_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_file.flush()

                    req_path = run_dir / "requests" / f"{cell.label}__{pattern}__rep{rep}.jsonl"
                    with open(req_path, "w") as rf:
                        for r in results:
                            rf.write(json.dumps(r.to_row()) + "\n")
                    print(f"    {pattern:13s} rep{rep}: p99 e2e={row['e2e_p99']:.2f}s "
                          f"thr={row['throughput_tok_per_s']:.0f} tok/s "
                          f"errors={row['n_errors']}", flush=True)
    if csv_file:
        csv_file.close()
    return summary_path


def _warm_up(cell: ServingConfig, n: int = 8) -> None:
    """Send a few requests so CUDA graphs / kernels are compiled before measurement."""
    from ..workload.generator import Request

    reqs = [Request(i, i * 0.05, 64, 32) for i in range(n)]
    asyncio.run(run_workload(reqs, cell.base_url, cell.model_name))


def cells_from_fixed(exp_cfg: dict, model_cfg: dict) -> list[ServingConfig]:
    return [configs_from_yaml(model_cfg, c["quantization"], c["batch_size"])
            for c in exp_cfg["configs_to_test"]["fixed"]]


def cells_from_grid(model_cfg: dict) -> list[ServingConfig]:
    return [configs_from_yaml(model_cfg, q, b)
            for q in model_cfg["quantization"]["levels"] for b in model_cfg["batch_sizes"]]
