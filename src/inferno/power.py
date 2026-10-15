"""
GPU power / memory / utilization sampler built on `nvidia-smi` streaming mode
(no extra Python dependency; works on Linux, WSL2 and Windows).

Use as a context manager around a measured run, then read .summary():

    with PowerSampler() as ps:
        run_workload(...)
    stats = ps.summary(output_tokens=n)   # includes energy_j and tokens_per_joule
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

import numpy as np

_QUERY = "power.draw,memory.used,utilization.gpu"


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _count_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout
        return max(1, sum(1 for l in out.splitlines() if l.startswith("GPU ")))
    except (OSError, subprocess.SubprocessError):
        return 1


class PowerSampler:
    def __init__(self, interval_ms: int = 200, gpu_indices: list[int] | int | None = 0):
        """gpu_indices: one index, a list, or None for every visible GPU. With several GPUs, power and
        memory are summed and utilization averaged per sampling tick."""
        self.interval_ms = interval_ms
        self.gpu_indices = None if gpu_indices is None else (
            [gpu_indices] if isinstance(gpu_indices, int) else list(gpu_indices))
        self._n_gpus = len(self.gpu_indices) if self.gpu_indices else _count_gpus()
        self._tick: list[tuple[float, float, float]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float, float, float]] = []  # (t, watts, mem_mib, util_pct)

    def __enter__(self) -> "PowerSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if not nvidia_smi_available():
            return
        self._proc = subprocess.Popen(
            ["nvidia-smi", *([f"--id={','.join(map(str, self.gpu_indices))}"] if self.gpu_indices else []),
             f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            self._ingest(line, time.monotonic())

    def _ingest(self, line: str, t: float) -> None:
        """Consume one nvidia-smi CSV line. nvidia-smi prints one line per GPU per tick, so a tick is
        complete after `_n_gpus` lines."""
        try:
            watts, mem, util = (float(x) for x in line.strip().split(","))
        except ValueError:   # "[N/A]" on GPUs that do not report power
            return
        self._tick.append((watts, mem, util))
        if len(self._tick) == self._n_gpus:
            w, m, u = zip(*self._tick)
            self.samples.append((t, sum(w), sum(m), sum(u) / len(u)))
            self._tick = []

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self, output_tokens: int | None = None) -> dict:
        if len(self.samples) < 2:
            return {"n_power_samples": len(self.samples)}
        t, w, mem, util = (np.array(c) for c in zip(*self.samples))
        energy = float(np.sum((w[1:] + w[:-1]) / 2 * np.diff(t)))   # trapezoid rule
        out = {
            "n_power_samples": len(t),
            "avg_power_w": float(w.mean()),
            "peak_power_w": float(w.max()),
            "energy_j": energy,
            "peak_mem_mib": float(mem.max()),
            "avg_gpu_util_pct": float(util.mean()),
        }
        if output_tokens is not None and energy > 0:
            out["tokens_per_joule"] = output_tokens / energy
        return out
