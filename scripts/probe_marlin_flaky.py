"""Isolate intermittent Marlin errors on the Qwen2.5-1.5B down shape (N=1536, K=8960).

Variants, 40 calls each, same inputs every call:
  plain      back-to-back calls, shared workspace (what vLLM does)
  dirty      allocate + free NaN-filled buffers between calls (stresses uninitialized-memory reads)
  fresh_ws   a newly zeroed workspace for every call (tests the cross-block lock/semaphore hypothesis)
  sync       torch.cuda.synchronize() between calls

    python scripts/probe_marlin_flaky.py [N K]
"""
import os
import sys

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
from vllm.scalar_type import scalar_types

GROUP = 128
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
N, K = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1536, 8960)


def rnd_words(r, c):
    return torch.randint(-2**31, 2**31 - 1, (r, c), generator=g, device=dev, dtype=torch.int64).to(torch.int32)


qweight, qzeros = rnd_words(K, N // 8), rnd_words(K // GROUP, N // 8)
scales = (torch.rand((K // GROUP, N), generator=g, device=dev) * 0.02 + 0.001).half()
ref_w = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0).float()
qw = ops.awq_marlin_repack(qweight, size_k=K, size_n=N, num_bits=4)
sc = mu.marlin_permute_scales(scales, size_k=K, size_n=N, group_size=GROUP)
zp = mu.awq_to_marlin_zero_points(qzeros, size_k=K // GROUP, size_n=N, num_bits=4)
e = mu.marlin_make_empty_g_idx(dev)
ws_shared = mu.marlin_make_workspace_new(dev)
print(f"N={N} K={K} alloc={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')} gpu={torch.cuda.get_device_name(0)} "
      f"SMs={torch.cuda.get_device_properties(0).multi_processor_count} workspace={tuple(ws_shared.shape)}")

for B in [1, 8, 16, 32, 48, 64, 128]:
    x = torch.randn((B, K), generator=g, device=dev, dtype=torch.float16)
    ref = x.float() @ ref_w
    for variant in ["plain", "dirty", "fresh_ws", "sync"]:
        bad, worst, junk = 0, 0.0, []
        for _ in range(40):
            ws = mu.marlin_make_workspace_new(dev) if variant == "fresh_ws" else ws_shared
            if variant == "dirty":
                junk.append(torch.full((1 << 20,), float("nan"), device=dev, dtype=torch.float16))
                del junk[:-4]
            y = ops.gptq_marlin_gemm(x, None, qw, None, sc, None, zp, e, e, ws, scalar_types.uint4, size_m=B, size_n=N,
                                     size_k=K, use_atomic_add=False, use_fp32_reduce=True)
            if variant == "sync":
                torch.cuda.synchronize()
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            bad += not (err < 1e-2)
            worst = max(worst, err)
        print(f"B={B:4d} {variant:8s} bad={bad:2d}/40 worst_rel_err={worst:.2e}", flush=True)
