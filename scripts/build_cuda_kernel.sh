#!/usr/bin/env bash
# Build and run the standalone CUDA kernel benchmark (needs nvcc; see scripts/setup_toolchain.sh).
#   source scripts/toolchain_env.sh && bash scripts/build_cuda_kernel.sh
set -euo pipefail
cd "$(dirname "$0")/.."
make -C src/inferno/kernels bench_int4_gemv
echo "Built src/inferno/kernels/bench_int4_gemv. Run: make -C src/inferno/kernels run   (or: make kernel)"
