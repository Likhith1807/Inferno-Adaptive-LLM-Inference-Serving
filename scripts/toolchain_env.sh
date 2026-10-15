#!/usr/bin/env bash
# Source me:  source scripts/toolchain_env.sh
# Puts the project venv and the no-root native toolchain (nvcc, gcc, mpirun, ncu) on PATH.
ENV_NAME="${INFERNO_TOOLCHAIN_ENV:-inferno-hpc}"
INFERNO_TOOLCHAIN_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/$ENV_NAME"
VENV_DIR="${INFERNO_VENV:-$HOME/.venvs/inferno}"

export PATH="$INFERNO_TOOLCHAIN_PREFIX/bin:$VENV_DIR/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="$INFERNO_TOOLCHAIN_PREFIX"
export CC="$INFERNO_TOOLCHAIN_PREFIX/bin/gcc"
export CXX="$INFERNO_TOOLCHAIN_PREFIX/bin/g++"
# WSL2 exposes the Windows driver's libcuda here; Triton and the CUDA runtime need to find it.
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$INFERNO_TOOLCHAIN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TRITON_LIBCUDA_PATH="/usr/lib/wsl/lib"
# Open MPI: allow running as a normal user, oversubscribe cores for scaling microbenchmarks.
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
