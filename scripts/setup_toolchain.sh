#!/usr/bin/env bash
# No-root native toolchain for WSL2/Linux: nvcc + cuBLAS headers, gcc/g++, Nsight Compute, OpenMPI.
# Uses micromamba so no sudo is needed. Idempotent. Afterwards:  source scripts/toolchain_env.sh
#
# What needs it:
#   Phase 3 kernel build/benchmark/profile  -> nvcc, cublas, ncu
#   Phase 4 MPI router                      -> mpirun, mpi4py
#   vLLM/Triton runtime stubs               -> a C compiler
set -euo pipefail

ENV_NAME="${INFERNO_TOOLCHAIN_ENV:-inferno-hpc}"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

# Validate, do not just locate: an interrupted earlier install can leave a 0-byte executable that "succeeds".
if [ -z "$(micromamba --version 2>/dev/null || true)" ]; then
    rm -f "$HOME/.local/bin/micromamba"
    echo "== Installing micromamba =="
    # Plain static binary: no bzip2/tar needed (a fresh WSL image has neither bzip2 nor conda).
    curl -fL -o "$HOME/.local/bin/micromamba"         https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
    chmod +x "$HOME/.local/bin/micromamba"
fi

if [ ! -d "$MAMBA_ROOT/envs/$ENV_NAME" ]; then
    echo "== Creating toolchain env '$ENV_NAME' (CUDA 12.4 toolchain, gcc 12, OpenMPI) =="
    create_env() {
        micromamba create -y -n "$ENV_NAME"             -c nvidia/label/cuda-12.4.1 -c conda-forge             cuda-nvcc cuda-cudart-dev cuda-cudart-static cuda-cccl libcublas-dev cuda-nsight-compute nsight-compute             "gxx_linux-64=12" "gcc_linux-64=12" openmpi make
    }
    # An interrupted earlier run can leave empty cache shards that break the solver; clean and retry once.
    create_env || { echo "== solver/cache error; cleaning the conda cache and retrying ==";                     micromamba clean -a -y; rm -rf "$HOME/.cache/conda" "$MAMBA_ROOT/envs/$ENV_NAME"; create_env; }
fi

PREFIX="$MAMBA_ROOT/envs/$ENV_NAME"
# nvcc and Triton look for plain gcc/g++/cc; conda ships them with a target prefix.
for pair in "gcc:gcc" "cc:gcc" "g++:g++" "c++:g++"; do
    name="${pair%%:*}"; real="${pair##*:}"
    [ -e "$PREFIX/bin/x86_64-conda-linux-gnu-$real" ] && ln -sf "$PREFIX/bin/x86_64-conda-linux-gnu-$real" "$PREFIX/bin/$name"
done

# The conda nsight-compute package unpacks to nsight-compute/<ver>/, but the ncu launcher looks for
# nsight-compute-<ver>/ next to bin/. Bridge with symlinks.
for d in "$PREFIX"/nsight-compute/*/; do
    [ -d "$d" ] && ln -sfn "${d%/}" "$PREFIX/nsight-compute-$(basename "$d")"
done

for tool in nvcc mpirun mpicc gcc; do
    [ -x "$PREFIX/bin/$tool" ] || { echo "ERROR: $tool missing from $PREFIX/bin - toolchain env is incomplete" >&2; exit 1; }
done
echo "== Toolchain ready in $PREFIX =="
"$PREFIX/bin/nvcc" --version | tail -2 || true
"$PREFIX/bin/mpirun" --version | head -1 || true

# mpi4py must be built against this OpenMPI and installed into the project venv.
VENV_DIR="${INFERNO_VENV:-$HOME/.venvs/inferno}"
if [ -x "$VENV_DIR/bin/python" ]; then
    echo "== Installing mpi4py into $VENV_DIR against the conda OpenMPI =="
    MPICC="$PREFIX/bin/mpicc" PATH="$PREFIX/bin:$PATH" uv pip install --python "$VENV_DIR/bin/python" \
        --no-binary mpi4py "mpi4py>=3.1.5"
fi
echo "Done. Use it with:  source scripts/toolchain_env.sh"
