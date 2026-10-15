#!/usr/bin/env bash
# End-to-end matrix: every configuration in a fresh process, CUDA-graph decode, 3 repetitions in a different
# (fixed, recorded) order each time to spread thermal drift across configurations.
#   bash scripts/in_env.sh bash scripts/run_e2e_matrix.sh [reps]
set -uo pipefail
source "$(dirname "$0")/toolchain_env.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
REPS="${1:-3}"
OUT=results/e2e_kernel
mkdir -p "$OUT"
# name:config:max_batch
ORDERS=(
  "fp16:fp16:0 awq_marlin:awq_marlin:0 awq_classic:awq_classic:0 inferno_b8:awq_inferno:8 inferno_b2:awq_inferno:2"
  "inferno_b2:awq_inferno:2 awq_classic:awq_classic:0 fp16:fp16:0 inferno_b8:awq_inferno:8 awq_marlin:awq_marlin:0"
  "awq_marlin:awq_marlin:0 inferno_b8:awq_inferno:8 inferno_b2:awq_inferno:2 fp16:fp16:0 awq_classic:awq_classic:0"
)
for r in $(seq 1 "$REPS"); do
  for item in ${ORDERS[$(( (r - 1) % 3 ))]}; do
    IFS=: read -r name cfg mb <<< "$item"
    args=(--config "$cfg" --mode graphs --reps 5 --concurrency 1,2,4,8 --out "$OUT/${name}_rep${r}.json")
    [ "$mb" != "0" ] && args+=(--max-batch "$mb")
    python -m inferno.experiments.run_e2e_kernel "${args[@]}" > "$OUT/${name}_rep${r}.log" 2>&1
    echo "rep $r $name: $(grep -E 'concurrency 1:' "$OUT/${name}_rep${r}.log" | tail -1)"
    sleep 15
  done
done
echo E2E_DONE
