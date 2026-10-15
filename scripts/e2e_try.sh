#!/usr/bin/env bash
# One end-to-end configuration, for development:  bash scripts/in_env.sh bash scripts/e2e_try.sh awq_inferno eager
set -uo pipefail
source "$(dirname "$0")/toolchain_env.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
CFG="${1:-awq_inferno}"; MODE="${2:-eager}"
python -m inferno.experiments.run_e2e_kernel --config "$CFG" --mode "$MODE" --reps 2 --concurrency 1,4 --out "results/e2e_dev/${CFG}_${MODE}.json" 2>&1 \
  | grep -vE "^(INFO|WARNING|\(EngineCore)|Processed prompts|Adding requests|^$" | tail -30
