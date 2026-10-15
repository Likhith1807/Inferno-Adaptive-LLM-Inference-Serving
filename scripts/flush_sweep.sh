#!/usr/bin/env bash
# Does the rotation size matter? Time the int4 kernels on one shape with increasing weight-rotation footprints.
#   bash scripts/in_env.sh bash scripts/flush_sweep.sh
set -euo pipefail
source "$(dirname "$0")/toolchain_env.sh"
cd "$(dirname "$0")/../src/inferno/kernels"
make bench_int4_gemv >/dev/null
mkdir -p ../../../results/flush_sweep
for mb in 192 384 768 1536; do   # 768 = default since the sweep showed 192 MB leaks ~6% via L2
  for r in 1 2 3; do
    ./bench_int4_gemv --shape 5 --flush-mb $mb --only int4_v5 > ../../../results/flush_sweep/mb${mb}_r${r}.json 2>/dev/null
    ./bench_int4_gemv --shape 5 --flush-mb $mb --only cublas > ../../../results/flush_sweep/cub_mb${mb}_r${r}.json 2>/dev/null
  done
done
echo done
