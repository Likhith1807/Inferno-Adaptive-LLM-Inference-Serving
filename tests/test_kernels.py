"""
GPU test: run the compiled kernel benchmark on one small shape and require that every kernel matches the CPU
reference. Skipped automatically when there is no NVIDIA GPU or the benchmark has not been built
(`make -C src/inferno/kernels`). CI runs on CPU-only runners, so this is a local/GPU-box test.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[1] / "src" / "inferno" / "kernels" / "bench_int4_gemv"

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="the benchmark is built for Linux (WSL2 counts)"),
    pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="needs an NVIDIA GPU"),
    pytest.mark.skipif(not BIN.exists(), reason="build first: make -C src/inferno/kernels"),
]


def test_all_kernels_match_cpu_reference_on_a_small_shape():
    proc = subprocess.run([str(BIN), "--shape", "1", "--iters", "2", "--reps", "2"], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "INCORRECT" not in proc.stderr, proc.stderr
    for name in ("int4_v0_naive", "int4_v1_warp", "int4_v2_vec", "int4_v3_pad", "int4_v4_alg", "int4_v5_reuse"):
        assert name in proc.stderr
