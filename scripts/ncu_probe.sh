#!/usr/bin/env bash
# Quick probe: can Nsight Compute read GPU performance counters on this machine?
set -uo pipefail
cd "$(dirname "$0")/../src/inferno/kernels"
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum --launch-count 2 ./bench_int4_gemv 2>&1 \
  | grep -E "ERR_|==PROF==|==ERROR==|dram__|gpu__time|Warning|WARNING" | head -20
