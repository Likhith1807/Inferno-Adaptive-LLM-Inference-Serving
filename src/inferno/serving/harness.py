"""
Inference serving harness: the one object the rest of the system talks to.

It owns a vLLM server process (serving/server.py) and exposes load / generate /
reconfigure. Load generation for benchmarks goes through serving/client.py
directly, because that path needs streaming and open-loop timing.

Reconfiguration (batch size, quantization) requires a server restart in vLLM,
so set_batch_size() / set_quantization() return the measured reload cost in
seconds. That cost is the "adaptive switching cost" the H1 discussion needs.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx

from .config import ServingConfig
from .server import VLLMServer


class InferenceHarness:
    def __init__(self, config: ServingConfig, log_dir: str | Path | None = None):
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else None
        self._server: VLLMServer | None = None
        self._last_latency_sec: float | None = None
        self.last_reload_sec: float | None = None

    def _log_path(self) -> Path | None:
        return self.log_dir / f"vllm_{self.config.label}.log" if self.log_dir else None

    def load_model(self) -> float:
        """Start the server for the current config. Returns cold-start seconds."""
        self._server = VLLMServer(self.config, self._log_path())
        self.last_reload_sec = self._server.start()
        return self.last_reload_sec

    def unload(self) -> None:
        if self._server:
            self._server.stop()
            self._server = None

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Blocking single-request generation (used by MPI replica workers)."""
        t0 = time.monotonic()
        resp = httpx.post(
            self.config.base_url + "/v1/completions",
            json={"model": self.config.model_name, "prompt": prompt,
                  "max_tokens": max_new_tokens, "temperature": 0.0},
            timeout=300.0,
        )
        resp.raise_for_status()
        self._last_latency_sec = time.monotonic() - t0
        return resp.json()["choices"][0]["text"]

    def reconfigure(self, **changes) -> float:
        """Apply config changes and restart the server. Returns reload seconds."""
        for key, value in changes.items():
            setattr(self.config, key, value)
        self.unload()
        return self.load_model()

    def set_batch_size(self, batch_size: int) -> float:
        return self.reconfigure(batch_size=batch_size)

    def set_quantization(self, quantization: str, model_name: str, vllm_quantization: str | None) -> float:
        return self.reconfigure(
            quantization=quantization, model_name=model_name, vllm_quantization=vllm_quantization
        )

    def get_last_latency(self) -> float | None:
        return self._last_latency_sec

    def __enter__(self) -> "InferenceHarness":
        self.load_model()
        return self

    def __exit__(self, *exc) -> None:
        self.unload()
