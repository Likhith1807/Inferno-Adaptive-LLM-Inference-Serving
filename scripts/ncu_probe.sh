#!/usr/bin/env bash
# Quick probe: can Nsight Compute read GPU performance counters on this machine?
set -uo pipefail
cd "$(dirname "$0")/../src/inferno/kernels"
./bench_int4_gemv --shape 2 --only int4_v4_alg --iters 1 --reps 1 > /dev/null 2> /tmp/bench_err.txt; echo "bench exit: $?"; tail -3 /tmp/bench_err.txt
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --kernel-name regex:gemv_int4 \
    --launch-count 1 ./bench_int4_gemv --shape 2 --only int4_v4_alg --iters 1 --reps 1 2>&1 \
  | grep -E "ERR_|==PROF==|==ERROR==|dram__|gpu__time|Warning|WARNING|error" | head -20
