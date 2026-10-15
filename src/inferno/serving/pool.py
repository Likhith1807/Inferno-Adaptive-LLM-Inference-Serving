"""A set of concurrently resident vLLM replicas (heterogeneous configs, one port each)."""
from __future__ import annotations

from pathlib import Path

from .config import ServingConfig
from .server import VLLMServer


class ReplicaPool:
    def __init__(self, configs: list[ServingConfig], log_dir: str | Path | None = None):
        self.configs = configs
        self.log_dir = Path(log_dir) if log_dir else None
        self._servers: list[VLLMServer] = []
        self.startup_sec: list[float] = []

    def start(self) -> None:
        """Start replicas one at a time (vLLM sizes its memory pool from what is free at startup, so
        concurrent starts would race)."""
        try:
            for cfg in self.configs:
                log = self.log_dir / f"vllm_{cfg.label}.log" if self.log_dir else None
                srv = VLLMServer(cfg, log)
                self._servers.append(srv)
                self.startup_sec.append(srv.start())
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        for srv in reversed(self._servers):
            srv.stop()
        self._servers = []

    @property
    def urls(self) -> list[str]:
        return [c.base_url for c in self.configs]

    @property
    def models(self) -> list[str]:
        return [c.model_name for c in self.configs]

    @property
    def capacities(self) -> list[int]:
        return [c.batch_size for c in self.configs]

    def __enter__(self) -> "ReplicaPool":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
