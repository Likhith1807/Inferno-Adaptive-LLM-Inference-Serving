#!/usr/bin/env python
"""
Environment doctor: verifies everything the project needs and says exactly what is missing.

    wsl -d Ubuntu -- bash /mnt/d/inferno/scripts/in_env.sh python scripts/check_env.py

Exit code 0 only if every *required* check passes (GPU visible to PyTorch, vLLM importable, nvcc, gcc, MPI).
"""
import importlib
import shutil
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    results = []

    def check(name, ok, detail, required=True):
        results.append((name, ok, detail, required))

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__} (built for CUDA {torch.version.cuda})"
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            detail += f"; {props.name}, {props.total_memory / 2**30:.1f} GiB, sm_{props.major}{props.minor}"
        check("PyTorch sees the GPU", cuda_ok, detail)
    except Exception as exc:
        check("PyTorch sees the GPU", False, f"import failed: {exc}")

    for mod in ("vllm", "httpx", "numpy", "scipy", "pandas", "matplotlib", "yaml", "mpi4py"):
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", True, getattr(m, "__version__", "ok"), required=(mod != "mpi4py"))
        except Exception as exc:
            check(f"import {mod}", False, str(exc).splitlines()[0][:80], required=(mod != "mpi4py"))

    nvcc = shutil.which("nvcc")
    check("nvcc", bool(nvcc), (run(["nvcc", "--version"]).splitlines() or ["missing"])[-1] if nvcc else "missing: run scripts/setup_toolchain.sh")
    check("gcc", bool(shutil.which("gcc")), run(["gcc", "--version"]).splitlines()[0] if shutil.which("gcc") else "missing")
    check("mpirun", bool(shutil.which("mpirun")), run(["mpirun", "--version"]).splitlines()[0] if shutil.which("mpirun") else "missing")
    check("ncu (Nsight Compute)", bool(shutil.which("ncu")), shutil.which("ncu") or "missing (needed only for kernel profiling)", required=False)
    check("nvidia-smi", bool(shutil.which("nvidia-smi")), run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]))

    width = max(len(r[0]) for r in results)
    bad = 0
    for name, ok, detail, required in results:
        mark = "ok  " if ok else ("FAIL" if required else "warn")
        bad += (not ok) and required
        print(f"[{mark}] {name:<{width}}  {detail}")
    print("\nAll required checks passed." if not bad else f"\n{bad} required check(s) failed.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
