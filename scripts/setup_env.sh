#!/usr/bin/env bash
# Inferno environment setup for Linux / WSL2 (Ubuntu). vLLM does not support native Windows.
# Uses uv with a pinned Python 3.12 because the distro's default Python may be too new for vLLM wheels.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Downloads are I/O-bound (parallel is fine); unpacking large wheels is the memory-heavy part (keep serial).
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"   # big CUDA wheels on a shared link exceed the 30 s default
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-8}" UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"

echo "== Inferno environment setup =="

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Keep the venv on the Linux filesystem: /mnt/<drive> is far slower for many small files in WSL.
VENV_DIR="${INFERNO_VENV:-$HOME/.venvs/inferno}"
# Reproducibility + driver compatibility: resolve only against packages published before this date.
# Unpinned resolution today pulls torch built for CUDA 13, which needs a >=580 driver; this date yields a
# CUDA 12.x stack that runs on 12.x drivers. Override with UV_EXCLUDE_NEWER=... if your driver is newer.
export UV_EXCLUDE_NEWER="${UV_EXCLUDE_NEWER:-2025-11-01T00:00:00Z}"
[ -x "$VENV_DIR/bin/python" ] || uv venv --python 3.12 "$VENV_DIR"   # idempotent: reuse an existing venv
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt
uv pip install -e .

echo "== Checking system build tools (Triton/vLLM compile small C stubs at runtime) =="
missing=()
for tool in gcc make; do command -v "$tool" &> /dev/null || missing+=("$tool"); done
if [ ${#missing[@]} -gt 0 ]; then
    echo "MISSING: ${missing[*]}  ->  run scripts/setup_toolchain.sh (no root) or: sudo apt install -y build-essential"
fi

echo "== Checking GPU availability =="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. In WSL2 install the NVIDIA driver on Windows (not inside WSL)."
fi

echo "== Checking CUDA toolkit (needed for Phase 3 kernel build) =="
if command -v nvcc &> /dev/null; then
    nvcc --version | tail -1
else
    echo "NOTE: nvcc not found. Run scripts/setup_toolchain.sh (no root needed) before Phase 3."
fi

echo "== Checking MPI (needed for Phase 4) =="
if command -v mpirun &> /dev/null; then
    mpirun --version | head -1
else
    echo "NOTE: mpirun not found. Run scripts/setup_toolchain.sh (no root needed) before Phase 4."
fi

python - <<'PY'
import sys, torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: PyTorch cannot see the GPU. Check that the Windows NVIDIA driver supports this CUDA build "
             "(nvidia-smi shows the maximum CUDA version), or set UV_EXCLUDE_NEWER to an older date.")
PY

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
