#!/usr/bin/env bash
# One command from a fresh WSL/Linux shell to a working environment: Python stack, then native toolchain.
#   wsl -d Ubuntu -- bash /mnt/<drive>/path/to/inferno/scripts/bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_env.sh
echo ENV_DONE
bash scripts/setup_toolchain.sh
echo TOOLCHAIN_DONE
