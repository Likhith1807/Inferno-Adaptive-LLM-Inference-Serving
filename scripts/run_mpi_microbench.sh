#!/usr/bin/env bash
# GPU-free MPI coordination benchmark: ping-pong latency vs message size, and sustained dispatch rate /
# round-trip latency vs number of workers. Writes results/mpi_microbench.json.
#   source scripts/toolchain_env.sh && bash scripts/run_mpi_microbench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
OUT=results/mpi_microbench.json
JOBS="${JOBS:-20000}"
PAYLOAD="${PAYLOAD:-512}"

echo "== ping-pong (2 ranks) =="
PP=$(mpirun --map-by :OVERSUBSCRIBE -n 2 python -m inferno.mpi_router.microbench pingpong)

echo "== dispatch scaling =="
DISPATCH=""
for W in 1 2 4 8 15; do
    ROW=$(mpirun --map-by :OVERSUBSCRIBE -n $((W + 1)) python -m inferno.mpi_router.microbench dispatch --jobs "$JOBS" --payload "$PAYLOAD")
    echo "  workers=$W  $ROW"
    DISPATCH="${DISPATCH:+$DISPATCH,}$ROW"
done

printf '{"pingpong": %s, "dispatch": [%s], "cores": %s}\n' "$PP" "$DISPATCH" "$(nproc)" > "$OUT"
echo "wrote $OUT"
