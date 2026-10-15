#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/toolchain_env.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m inferno.experiments.report_e2e results/e2e_kernel docs/results/e2e_kernel
