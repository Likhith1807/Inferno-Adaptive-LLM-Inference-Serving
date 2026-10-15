#!/usr/bin/env bash
# Run the kernel benchmark N times (fresh process each time) and aggregate with run-to-run spread.
#   bash scripts/in_env.sh bash scripts/repeat_kernel_bench.sh 5
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-5}"
make -C src/inferno/kernels bench_int4_gemv > /dev/null
mkdir -p results/kernel_bench_runs
rm -f results/kernel_bench_runs/run_*.json
for i in $(seq 1 "$N"); do
    echo "== kernel benchmark run $i/$N =="
    src/inferno/kernels/bench_int4_gemv > "results/kernel_bench_runs/run_$i.json" 2> "results/kernel_bench_runs/run_$i.txt"
    grep -c INCORRECT "results/kernel_bench_runs/run_$i.txt" | sed 's/^/  incorrect kernels: /' || true
    sleep 20      # let the laptop GPU settle between runs
done
python scripts/aggregate_kernel_runs.py results/kernel_bench_runs results/kernel_bench.json
