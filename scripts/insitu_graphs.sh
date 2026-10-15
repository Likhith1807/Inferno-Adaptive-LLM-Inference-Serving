#!/usr/bin/env bash
# In-situ kernel timing of the three int4 paths under BOTH engine modes: one fresh process per (path, mode, repetition),
# 3 repetitions, then aggregated by median across processes (process-to-process variance is ~10%).
#   bash scripts/in_env.sh bash scripts/insitu_graphs.sh
set -uo pipefail
source "$(dirname "$0")/toolchain_env.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p results/insitu
rm -f results/insitu/*.json results/insitu/*.log
for rep in 1 2 3; do
  for mode in graphs eager; do
    for q in awq_marlin awq awq_inferno; do
      python -m inferno.kernels.bench_vllm_marlin --quantization "$q" --mode "$mode" --out "results/insitu/${q}_${mode}_rep${rep}.json" > "results/insitu/${q}_${mode}_rep${rep}.log" 2>&1
      echo "rep $rep $q $mode done"
      sleep 10
    done
  done
done
python scripts/aggregate_insitu.py results/insitu
echo INSITU_DONE
