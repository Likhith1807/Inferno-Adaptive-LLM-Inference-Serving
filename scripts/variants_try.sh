#!/usr/bin/env bash
# Exploratory: lane-count / unroll variants of the batch-1 kernel on every shape (one invocation).
set -uo pipefail
source "$(dirname "$0")/toolchain_env.sh"
cd "$(dirname "$0")/../src/inferno/kernels"
make bench_int4_gemv >/dev/null || exit 1
./bench_int4_gemv --variants --only v6 2>&1 >/dev/null | grep -E "^qwen|^llama|int4_v6|INCORRECT"
echo ---
./bench_int4_gemv --only v5 2>&1 >/dev/null | grep -E "^qwen|^llama|int4_v5"
