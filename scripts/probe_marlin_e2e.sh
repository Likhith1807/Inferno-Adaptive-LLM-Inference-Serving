#!/usr/bin/env bash
# Real-checkpoint check of the Marlin batch-size fault: Marlin vs classic AWQ, CUDA graphs and eager.
set -uo pipefail
mkdir -p results/marlin_fault
for q in awq_marlin awq; do
  python scripts/probe_marlin_e2e.py --quantization "$q" 2>&1 | grep -E "^(awq|Traceback|.*Error)"
  python scripts/probe_marlin_e2e.py --quantization "$q" --eager 2>&1 | grep -E "^(awq|Traceback|.*Error)"
done
