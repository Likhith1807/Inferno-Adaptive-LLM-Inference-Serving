"""Capture the environment a benchmark ran in, so every result file is self-describing."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

_GPU_FIELDS = "name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit"
_PACKAGES = ("vllm", "torch", "transformers", "numpy", "httpx")


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_hardware() -> dict:
    gpu_raw = _run(["nvidia-smi", f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (gpu_raw or "").splitlines():
        vals = [v.strip() for v in line.split(",")]
        gpus.append(dict(zip(_GPU_FIELDS.split(","), vals)))

    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpus": gpus,
        "packages": versions,
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
    }
