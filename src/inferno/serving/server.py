"""
Lifecycle manager for a vLLM OpenAI-compatible server subprocess.

Running vLLM as a server (instead of in-process `LLM(...)`) gives the benchmark
a real network boundary, streaming responses for TTFT/TPOT, and a place to
measure cold-start cost, which is the price of any reconfiguration that needs
a reload (e.g. switching quantization level).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import ServingConfig


class ServerStartError(RuntimeError):
    pass


def build_command(config: ServingConfig) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model_name,
        "--host", config.host,
        "--port", str(config.port),
        "--max-num-seqs", str(config.batch_size),
        "--max-model-len", str(config.max_model_len),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--dtype", "float16",   # AWQ requires fp16; keep all levels consistent
    ]
    if config.vllm_quantization:
        cmd += ["--quantization", config.vllm_quantization]
    return cmd + list(config.extra_args)


class VLLMServer:
    def __init__(self, config: ServingConfig, log_path: str | Path | None = None):
        self.config = config
        self.log_path = Path(log_path) if log_path else None
        self._proc: subprocess.Popen | None = None
        self._log = None
        self.startup_sec: float | None = None

    def start(self) -> float:
        """Launch the server and block until /health responds. Returns cold-start seconds."""
        if self._proc is not None:
            raise RuntimeError("server already started")
        stdout = subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "wb")
            stdout = self._log
        t0 = time.monotonic()
        env = os.environ.copy()
        if self.config.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_index)
        self._proc = subprocess.Popen(
            build_command(self.config), stdout=stdout, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,   # own process group so stop() reaps vLLM workers too
        )
        deadline = t0 + self.config.startup_timeout_sec
        health = self.config.base_url + "/health"
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerStartError(
                    f"vLLM exited with code {self._proc.returncode} during startup"
                    + (f"; see {self.log_path}" if self.log_path else "")
                )
            try:
                if httpx.get(health, timeout=2.0).status_code == 200:
                    self.startup_sec = time.monotonic() - t0
                    return self.startup_sec
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        self.stop()
        raise ServerStartError(f"vLLM not healthy after {self.config.startup_timeout_sec:.0f}s")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                self._proc.wait()
        self._proc = None
        if self._log:
            self._log.close()
            self._log = None

    def __enter__(self) -> "VLLMServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
