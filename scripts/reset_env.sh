#!/usr/bin/env bash
# Rebuild the Python venv and the native toolchain from scratch (use after changing the pinned resolution date,
# or if an install was interrupted). Deletes only the project venv, the toolchain env, and the uv cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

VENV_DIR="${INFERNO_VENV:-$HOME/.venvs/inferno}"
TOOLCHAIN="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/${INFERNO_TOOLCHAIN_ENV:-inferno-hpc}"

rm -rf "$VENV_DIR" "$TOOLCHAIN"
command -v uv &> /dev/null && uv cache clean

bash scripts/setup_env.sh
echo "ENV_DONE"
bash scripts/setup_toolchain.sh
echo "TOOLCHAIN_DONE"
