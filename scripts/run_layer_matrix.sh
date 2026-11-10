#!/usr/bin/env bash
# Milestone 2 layer matrix: one process per shape (WSL2's GPU allocator fails after many allocations in one
# process), merged into one JSON, then the generated report.
#   wsl -d Ubuntu -- bash /mnt/d/inferno/scripts/in_env.sh bash scripts/run_layer_matrix.sh
set -euo pipefail
OUT=results/layer_matrix/layer_matrix.json
mkdir -p results/layer_matrix
rm -f "$OUT"
for s in qkv o_proj gate_up down; do
  python -m inferno.kernels.bench_layer_matrix --shapes "$s" --out "$OUT" "$@"
done
python scripts/report_layer_matrix.py "$OUT" docs/results/layer_matrix
