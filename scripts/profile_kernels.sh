#!/usr/bin/env bash
# Nsight Compute profile of every int4 GEMV version on two shapes.
#   source scripts/toolchain_env.sh && make -C src/inferno/kernels profile
#
# Outputs (results/ncu/):
#   <kernel>_shape<i>.ncu-rep   full-set report (open in the Nsight Compute GUI)
#   <kernel>_shape<i>.csv       the metrics below, one row per kernel
# Then: python scripts/summarize_ncu.py  ->  results/ncu/summary.md
#
# NOTE (WSL2): GPU performance counters may be blocked ("ERR_NVGPUCTRPERM"). Enable "Allow access to the GPU
# performance counters to all users" in the NVIDIA Control Panel (Developer > Manage GPU Performance Counters)
# on Windows, then retry. If ncu still cannot access counters under WSL2, profile the same binary from Windows.
set -euo pipefail
BIN="${1:-./bench_int4_gemv}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/results/ncu"
mkdir -p "$OUT"

KERNELS="int4_v0_naive int4_v1_warp int4_v2_vec int4_v3_pad int4_v4_alg int4_v5_reuse"
SHAPES="2 5"     # qwen1.5B gate_up (largest Qwen layer), llama7B mlp_up
METRICS="gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed.sum,\
launch__registers_per_thread,\
launch__shared_mem_per_block_dynamic"

for shape in $SHAPES; do
  for k in $KERNELS; do
    echo "== profile $k (shape $shape) =="
    ncu --kernel-name "regex:gemv_int4" --launch-count 1 --set full \
        -o "$OUT/${k}_shape${shape}" -f \
        "$BIN" --shape "$shape" --only "$k" --iters 1 --reps 1 > /dev/null
    ncu --import "$OUT/${k}_shape${shape}.ncu-rep" --metrics "$METRICS" --csv \
        --page raw > "$OUT/${k}_shape${shape}.csv"
  done
done
echo "Reports in $OUT. Summarize with: python scripts/summarize_ncu.py"
