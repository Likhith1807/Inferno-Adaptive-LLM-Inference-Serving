#!/usr/bin/env bash
# Quick A/B helper: rebuild the kernel benchmark (optionally with extra nvcc flags) and print v3/v4 vs cuBLAS on
# selected shape indices (0 qkv, 1 o_proj, 2 gate_up, 3 down, 4 llama attn, 5 llama mlp_up).
#   bash scripts/in_env.sh bash scripts/bench_kernel_shapes.sh "2 3 5" "-DINFERNO_NO_STREAM_LOADS"
set -euo pipefail
cd "$(dirname "$0")/../src/inferno/kernels"
SHAPES="${1:-2 3 5}"
EXTRA="${2:-}"
rm -f bench_int4_gemv
make EXTRA="$EXTRA" bench_int4_gemv > /dev/null
for s in $SHAPES; do
    ./bench_int4_gemv --shape "$s" 2>&1 >/dev/null | grep -E "qwen|llama|cublas_fp16|int4_v3|int4_v4|INCORRECT"
done
