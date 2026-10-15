"""
Run the custom fused int4 kernel inside vLLM's AWQ linear layers.

vLLM keeps AWQ weights as AutoAWQ-format tensors (qweight [K, N/8] int32 with the nibbles of one word holding eight
OUTPUT channels in the interleaved order 0,2,4,6,1,3,5,7; qzeros the same; scales [K/G, N]). This module

  1. repacks those tensors once, at load time, into the kernel's layout (uint32 [N][K/8] along the reduction axis,
     scales and zeros [N][K/G]),
  2. registers the kernel as a torch custom op (``torch.ops.inferno.int4_gemv``) with a fake implementation, so
     torch.compile and CUDA-graph capture can see it,
  3. patches ``AWQLinearMethod`` so decode steps with at most ``max_batch`` tokens use the custom kernel and every
     other step (prefill, large batches) falls back to vLLM's stock path.

Nothing else in vLLM is changed. ``install()`` must be called before the model is built. ``verify_layer`` compares the
custom kernel against vLLM's own dequantize-then-matmul on the loaded weights, so correctness is checked on the real
checkpoint and not only on synthetic data.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

_LIB = None
_STATS = {"custom_calls": 0, "fallback_calls": 0}
MAX_DIRECT_BATCH = 16


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = os.environ.get("INFERNO_INT4_LIB") or str(Path(__file__).with_name("libinferno_int4.so"))
        _LIB = ctypes.CDLL(path)
        # (int B, W, S, Z, X, Y, int N, int K, int G, void* stream)
        _LIB.inferno_int4_gemv.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        _LIB.inferno_int4_gemv.restype = ctypes.c_int
    return _LIB


# ------------------------------------------------------------------ weight repacking
_AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]     # nibble i of an AutoAWQ word holds output channel 8*col + _AWQ_ORDER[i]


def _unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """[R, C] int32 (8 nibbles per word, AWQ interleave) -> [R, 8*C] int32 with values 0..15 in channel order."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF                 # [R, C, 8]; index i = nibble position
    order = torch.tensor(_AWQ_ORDER, device=packed.device)
    ch = torch.empty_like(nib)
    ch[..., order] = nib                                          # channel-ordered
    return ch.reshape(packed.shape[0], -1)


def repack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor):
    """AutoAWQ tensors -> (W uint32-as-int32 [N, K/8], S fp16 [N, K/G], Z fp16 [N, K/G])."""
    K = qweight.shape[0]
    q = _unpack_awq(qweight)                                      # [K, N]
    N = q.shape[1]
    z = _unpack_awq(qzeros)                                       # [K/G, N]
    qt = q.t().contiguous().view(N, K // 8, 8).to(torch.int64)    # nibble j of word w is k = 8w + j
    shifts = (torch.arange(8, device=q.device, dtype=torch.int64) * 4)
    words = (qt << shifts).sum(-1)                                # [N, K/8], 0 .. 2^32-1
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).contiguous()
    return words, scales.t().contiguous().to(torch.float16), z.t().contiguous().to(torch.float16)


# ------------------------------------------------------------------ custom op
@torch.library.custom_op("inferno::int4_gemv", mutates_args=())
def int4_gemv(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, z: torch.Tensor, group: int) -> torch.Tensor:
    B, K = x.shape
    N = w.shape[0]
    y = torch.empty((B, N), dtype=torch.float16, device=x.device)
    rc = _load_lib().inferno_int4_gemv(B, w.data_ptr(), s.data_ptr(), z.data_ptr(), x.data_ptr(), y.data_ptr(), N, K, group,
                                       torch.cuda.current_stream(x.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"inferno_int4_gemv failed with code {rc} (B={B}, N={N}, K={K}, G={group})")
    return y


@int4_gemv.register_fake
def _(x, w, s, z, group):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float16)


# ------------------------------------------------------------------ vLLM patch
def install(max_batch: int = 8, verbose: bool = True) -> None:
    """Patch vLLM's AWQ linear method. Call before constructing the LLM."""
    from vllm.model_executor.layers.quantization.awq import AWQLinearMethod

    if getattr(AWQLinearMethod, "_inferno_patched", False):
        AWQLinearMethod._inferno_max_batch = max_batch
        return
    orig_post, orig_apply = AWQLinearMethod.process_weights_after_loading, AWQLinearMethod.apply
    AWQLinearMethod._inferno_max_batch = max_batch

    def post(self, layer):
        orig_post(self, layer)
        gs = self.quant_config.group_size
        w, s, z = repack_awq(layer.qweight.data, layer.qzeros.data, layer.scales.data)
        layer.inferno_w, layer.inferno_s, layer.inferno_z, layer.inferno_group = w, s, z, gs

    def apply(self, layer, x, bias=None):
        n = x.shape[:-1].numel()
        if n <= self._inferno_max_batch and x.dtype == torch.float16 and hasattr(layer, "inferno_w"):
            _STATS["custom_calls"] += 1
            y = torch.ops.inferno.int4_gemv(x.reshape(n, x.shape[-1]).contiguous(), layer.inferno_w, layer.inferno_s,
                                            layer.inferno_z, layer.inferno_group)
            if bias is not None:
                y = y + bias
            return y.reshape(x.shape[:-1] + (y.shape[-1],))
        _STATS["fallback_calls"] += 1
        return orig_apply(self, layer, x, bias)

    AWQLinearMethod.process_weights_after_loading = post
    AWQLinearMethod.apply = apply
    AWQLinearMethod._inferno_patched = True
    if verbose:
        print(f"[inferno] AWQLinearMethod patched: custom kernel for <= {max_batch} tokens")


def stats() -> dict:
    return dict(_STATS)


def verify_layer(layer, x_tokens: int = 1, seed: int = 0) -> float:
    """Max relative error of the custom kernel vs vLLM's dequantize + matmul on one loaded layer."""
    from vllm import _custom_ops as ops

    g = torch.Generator(device=layer.qweight.device).manual_seed(seed)
    K = layer.qweight.shape[0]
    x = torch.randn((x_tokens, K), generator=g, device=layer.qweight.device, dtype=torch.float16)
    ref = torch.matmul(x, ops.awq_dequantize(layer.qweight, layer.scales, layer.qzeros, 0, 0, 0)).float()
    out = torch.ops.inferno.int4_gemv(x, layer.inferno_w, layer.inferno_s, layer.inferno_z, layer.inferno_group).float()
    return ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
