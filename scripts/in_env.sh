#!/usr/bin/env bash
# Run any command inside the project environment (venv + native toolchain), from the repo root.
#   wsl -d Ubuntu -- bash /mnt/d/inferno/scripts/in_env.sh make -C src/inferno/kernels run
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/toolchain_env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
