"""Serving configuration dataclass. No external deps."""
from __future__ import annotations

from dataclasses import dataclass, field

QUANT_LEVELS = ("fp16", "int8", "int4")


@dataclass
class ServingConfig:
    model_name: str
    quantization: str = "fp16"       # "fp16" | "int8" | "int4"
    batch_size: int = 16             # maps to vLLM --max-num-seqs
    max_model_len: int = 2048
    device: str = "cuda"
    use_custom_kernel: bool = False  # toggles the hand-written CUDA dequant kernel (H3)
    vllm_quantization: str | None = None   # value for --quantization (None = unquantized)
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float = 0.70
    startup_timeout_sec: float = 600.0
    extra_args: list[str] = field(default_factory=list)
    tier_name: str = ""               # set for replica-pool members; overrides label
    gpu_index: int | None = None      # pin this replica to one GPU (CUDA_VISIBLE_DEVICES); None = default

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def label(self) -> str:
        return self.tier_name or f"{self.quantization}-bs{self.batch_size}"


def configs_from_yaml(model_cfg: dict, quantization: str, batch_size: int) -> ServingConfig:
    """Build a ServingConfig for one (quantization, batch_size) cell from model_config.yaml."""
    server = model_cfg.get("server", {})
    return ServingConfig(
        model_name=model_cfg["checkpoints"][quantization],
        quantization=quantization,
        batch_size=batch_size,
        max_model_len=model_cfg["model"]["max_model_len"],
        vllm_quantization=model_cfg.get("vllm_quantization", {}).get(quantization),
        host=server.get("host", "127.0.0.1"),
        port=server.get("port", 8000),
        gpu_memory_utilization=server.get("gpu_memory_utilization", 0.70),
        startup_timeout_sec=server.get("startup_timeout_sec", 600.0),
    )


def pool_from_yaml(model_cfg: dict, pool_cfg: dict) -> list[ServingConfig]:
    """Build one ServingConfig per pool tier, each on its own port with its own vLLM engine args."""
    out = []
    for i, tier in enumerate(pool_cfg["pool"]):
        cfg = configs_from_yaml(model_cfg, tier["quantization"], tier["batch_size"])
        cfg.port = pool_cfg.get("base_port", 8100) + i
        cfg.gpu_memory_utilization = tier.get("gpu_memory_utilization", 0.40)
        cfg.tier_name = tier["name"]
        cfg.gpu_index = tier.get("gpu_index")
        if "max_num_batched_tokens" in tier:
            cfg.extra_args += ["--max-num-batched-tokens", str(tier["max_num_batched_tokens"])]
        out.append(cfg)
    return out
