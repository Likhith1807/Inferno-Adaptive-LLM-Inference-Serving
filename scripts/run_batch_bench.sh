#!/usr/bin/env bash
# Runs the small-batch kernel benchmark N times (default 3) and stores each invocation.
#   bash scripts/run_batch_bench.sh 3
set -euo pipefail
source "$(dirname "$0")/toolchain_env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${1:-3}"
mkdir -p "$ROOT/results/kernel_batch_runs"
cd "$ROOT/src/inferno/kernels"
make bench_int4_batch >/dev/null
for i in $(seq 1 "$RUNS"); do
  ./bench_int4_batch > "$ROOT/results/kernel_batch_runs/run_$i.json" 2> "$ROOT/results/kernel_batch_runs/run_$i.txt"
  echo "run $i done"
done
cat "$ROOT/results/kernel_batch_runs/run_1.txt"
