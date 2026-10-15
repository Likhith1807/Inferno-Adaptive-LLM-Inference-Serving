#!/usr/bin/env bash
# Development: each configuration once in graph mode (quick look; the real matrix is scripts/run_e2e_matrix.sh).
set -uo pipefail
for c in awq_inferno awq_classic fp16; do
  bash "$(dirname "$0")/e2e_try.sh" "$c" graphs
done
