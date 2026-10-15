"""
Phase 1: fixed-config baselines.

For each fixed (quantization, batch_size) in experiment_config.yaml, serve the model with vLLM
and replay each load pattern `repetitions` times, recording latency / throughput / power.

    python -m inferno.experiments.run_baseline --config configs/experiment_config.yaml
    python -m inferno.experiments.run_baseline --config configs/experiment_config.yaml --dry-run
"""
import argparse

from .runner import (DEFAULT_MODEL_CFG, DEFAULT_WORKLOAD_CFG, cells_from_fixed, load_yaml,
                     new_run_dir, plan_summary, run_cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiment_config.yaml")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--workload-config", default=DEFAULT_WORKLOAD_CFG)
    parser.add_argument("--reps", type=int, help="override repetitions")
    parser.add_argument("--duration", type=float, help="override run duration (sec), for smoke tests")
    parser.add_argument("--patterns", nargs="+", help="override load patterns")
    parser.add_argument("--only", nargs="+", help="run only these config labels, e.g. int4-bs32")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    exp_cfg, model_cfg, wl_cfg = (load_yaml(p) for p in (args.config, args.model_config, args.workload_config))
    if args.duration:
        wl_cfg["duration_sec"] = args.duration
        wl_cfg["warmup_sec"] = min(wl_cfg["warmup_sec"], args.duration / 4)
    reps = args.reps or exp_cfg["repetitions"]
    patterns = args.patterns or exp_cfg["load_patterns_to_test"]
    cells = cells_from_fixed(exp_cfg, model_cfg)
    if args.only:
        cells = [c for c in cells if c.label in args.only]
        if not cells:
            raise SystemExit(f"no config matches {args.only}")

    print(plan_summary(cells, patterns, reps, wl_cfg))
    if args.dry_run:
        return
    run_dir = new_run_dir(exp_cfg.get("output_dir", "results/"), "baseline")
    path = run_cells(cells, patterns, reps, wl_cfg, run_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
