"""
Phase 2: full batch-size x quantization grid (every level in model_config.yaml).

    python -m inferno.experiments.run_sweep --config configs/experiment_config.yaml
"""
import argparse

from .runner import (DEFAULT_MODEL_CFG, DEFAULT_WORKLOAD_CFG, cells_from_grid, load_yaml,
                     new_run_dir, plan_summary, run_cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiment_config.yaml")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--workload-config", default=DEFAULT_WORKLOAD_CFG)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--patterns", nargs="+")
    parser.add_argument("--quantizations", nargs="+", help="subset of fp16 int8 int4")
    parser.add_argument("--batch-sizes", type=int, nargs="+", help="subset of the configured batch sizes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp_cfg, model_cfg, wl_cfg = (load_yaml(p) for p in (args.config, args.model_config, args.workload_config))
    if args.duration:
        wl_cfg["duration_sec"] = args.duration
        wl_cfg["warmup_sec"] = min(wl_cfg["warmup_sec"], args.duration / 4)
    reps = args.reps or exp_cfg["repetitions"]
    patterns = args.patterns or exp_cfg["load_patterns_to_test"]
    cells = cells_from_grid(model_cfg)
    if args.quantizations:
        cells = [c for c in cells if c.quantization in args.quantizations]
    if args.batch_sizes:
        cells = [c for c in cells if c.batch_size in args.batch_sizes]
    if not cells:
        raise SystemExit("no configs left after filtering")

    print(plan_summary(cells, patterns, reps, wl_cfg))
    if args.dry_run:
        return
    run_dir = new_run_dir(exp_cfg.get("output_dir", "results/"), "sweep")
    print(f"\nWrote {run_cells(cells, patterns, reps, wl_cfg, run_dir)}")


if __name__ == "__main__":
    main()
