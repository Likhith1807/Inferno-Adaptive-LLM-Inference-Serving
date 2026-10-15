#!/usr/bin/env bash
# Rebuild every kernel artifact and re-run the kernel measurements from scratch (no cached numbers):
#   5 invocations of the batch-1 ladder, 3 of the small-batch sweep; then the shared library for vLLM.
#   bash scripts/in_env.sh bash scripts/rerun_kernel_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/toolchain_env.sh
make -C src/inferno/kernels libinferno_int4.so
bash scripts/repeat_kernel_bench.sh 5
bash scripts/run_batch_bench.sh 3 > /dev/null
echo ALLDONE
