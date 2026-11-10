#!/usr/bin/env bash
# Short GPU benchmark (~3-5 min) for independent reproduction. Records the hardware, builds the kernel library,
# times custom / Marlin / AWQ / fp16 on two Qwen2.5-1.5B layers at batch 1 and 16, and runs the Marlin
# batch-size correctness probe on one shape. Results land in results/quick/<timestamp>/.
#
#   make quick-bench
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="results/quick/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

{
  echo "date: $(date -Is)"
  echo "host: $(uname -srm)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv,noheader \
    | sed 's/^/gpu: /'
  python -c "import torch, vllm; print('torch:', torch.__version__, '| cuda:', torch.version.cuda, '| vllm:', vllm.__version__)"
  nvcc --version | tail -1 | sed 's/^/nvcc: /'
} | tee "$OUT/hardware.txt"

make -s -C src/inferno/kernels libinferno_int4.so

python -m inferno.kernels.bench_layer_matrix --shapes qkv --batches 1,16 --rotate-mb 256 --trials 3 --reps 5 \
  --out "$OUT/layer_matrix.json" 2>&1 | grep -vE "^(INFO|WARNING)"
python -m inferno.kernels.bench_layer_matrix --shapes gate_up --batches 1,16 --rotate-mb 256 --trials 3 --reps 5 \
  --out "$OUT/layer_matrix.json" 2>&1 | grep -vE "^(INFO|WARNING)"
python scripts/probe_marlin_flaky.py 2048 1536 2>&1 | grep -vE "^(INFO|WARNING)" | tee "$OUT/marlin_probe.txt"

echo
echo "Done: $OUT  (send hardware.txt, layer_matrix.json and marlin_probe.txt with your reproduction report)"
