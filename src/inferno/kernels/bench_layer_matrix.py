"""
Milestone 2: which int4 implementation wins on the real Qwen2.5-1.5B layer shapes, at each batch size?

For every merged linear layer vLLM runs during decode (qkv, o_proj, gate_up, down) and batch sizes 1..64 this times
four implementations on the SAME AutoAWQ-format weights:

    custom      the fused lane-split int4 kernel (libinferno_int4.so), after a one-time repack to its layout;
                instantiated for B in {1,2,4,8,16}; larger batches fall back to the built-in path (cell = n/a)
    marlin      vLLM's gptq_marlin_gemm exactly as apply_awq_marlin_linear calls it (after awq_marlin_repack)
    awq         vLLM's classic AWQ path exactly as AWQLinearMethod.apply runs it (awq_gemm below 256 tokens)
    fp16        torch.matmul on the dequantized fp16 weight (cuBLAS), as an un-quantized reference

Method (same for every cell):
  * weights are rotated over enough independent copies that the working set is >= --rotate-mb (default 768 MB,
    24x the 32 MB L2), so each call streams its weight from DRAM like a real decode step over 28 layers;
  * one CUDA graph holds one call per copy, so Python and launch overhead are excluded for every implementation;
  * the graph is replayed --reps times per trial, --trials trials; the cell reports the median trial and its
    min/max, in microseconds per call;
  * correctness: custom and marlin outputs are compared with awq_dequantize + matmul (relative max error).

One-time conversion costs (repack AutoAWQ -> custom layout, AutoAWQ -> Marlin layout) are timed separately: the
deployed path pays them once at model load, never per token.

    python -m inferno.kernels.bench_layer_matrix --out results/layer_matrix/layer_matrix.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

# Hundreds of small weight copies: pooled segments avoid a per-allocation limit hit under WSL2.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

SHAPES = {  # name: (N, K) for Qwen2.5-1.5B (hidden 1536, 12 q heads + 2 kv heads of 128, intermediate 8960)
    "qkv": (2048, 1536),
    "o_proj": (1536, 1536),
    "gate_up": (17920, 1536),
    "down": (1536, 8960),
}
BATCHES = [1, 2, 4, 8, 16, 32, 64]
CUSTOM_BATCHES = {1, 2, 4, 8, 16}
GROUP = 128


def make_awq(N: int, K: int, g: torch.Generator, dev) -> dict:
    """Random AutoAWQ-format layer: qweight [K, N/8] int32, qzeros [K/G, N/8] int32, scales [K/G, N] fp16."""
    def rnd_words(r, c):
        return torch.randint(-2**31, 2**31 - 1, (r, c), generator=g, device=dev, dtype=torch.int64).to(torch.int32)
    return {"qweight": rnd_words(K, N // 8), "qzeros": rnd_words(K // GROUP, N // 8),
            "scales": (torch.rand((K // GROUP, N), generator=g, device=dev) * 0.02 + 0.001).half()}


class Impl:
    """prepare(awq) -> state (one-time conversion); call(x, state) -> y."""

    def __init__(self, name, prepare, call, batches=None):
        self.name, self.prepare, self.call, self.batches = name, prepare, call, batches


def build_impls(N: int, K: int):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.scalar_type import scalar_types

    from inferno.kernels import vllm_int4

    dev = torch.device("cuda")
    workspace = mu.marlin_make_workspace_new(dev)
    empty = mu.marlin_make_empty_g_idx(dev)

    def prep_custom(a):
        w, s, z = vllm_int4.repack_awq(a["qweight"], a["qzeros"], a["scales"])
        return (w, s, z)

    def call_custom(x, st):
        return torch.ops.inferno.int4_gemv(x, st[0], st[1], st[2], GROUP)

    def prep_marlin(a):
        return (ops.awq_marlin_repack(a["qweight"], size_k=K, size_n=N, num_bits=4),
                mu.marlin_permute_scales(a["scales"], size_k=K, size_n=N, group_size=GROUP),
                mu.awq_to_marlin_zero_points(a["qzeros"], size_k=K // GROUP, size_n=N, num_bits=4))

    def call_marlin(x, st):
        return mu.apply_awq_marlin_linear(x, st[0], st[1], st[2], empty, empty, workspace, scalar_types.uint4,
                                          output_size_per_partition=N, input_size_per_partition=K)

    def prep_awq(a):
        return (a["qweight"], a["scales"], a["qzeros"])

    def call_awq(x, st):
        if x.shape[0] >= 256:
            return torch.matmul(x, ops.awq_dequantize(st[0], st[1], st[2], 0, 0, 0))
        return ops.awq_gemm(x, st[0], st[1], st[2], 8)

    def prep_fp16(a):
        return (ops.awq_dequantize(a["qweight"], a["scales"], a["qzeros"], 0, 0, 0).t().contiguous(),)  # [N, K]

    def call_fp16(x, st):
        return torch.nn.functional.linear(x, st[0])

    return [Impl("custom", prep_custom, call_custom, CUSTOM_BATCHES), Impl("marlin", prep_marlin, call_marlin),
            Impl("awq", prep_awq, call_awq), Impl("fp16", prep_fp16, call_fp16)]


def state_bytes(st) -> int:
    return sum(t.numel() * t.element_size() for t in st)


def time_cell(impl: Impl, states: list, x: torch.Tensor, reps: int, trials: int) -> dict:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                      # warm up (autotune / lazy init) outside capture
        for st in states[:3]:
            impl.call(x, st)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for st in states:
            impl.call(x, st)
    graph.replay()
    torch.cuda.synchronize()
    per_call = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        torch.cuda.synchronize()
        per_call.append(a.elapsed_time(b) * 1000.0 / (reps * len(states)))
    del graph
    return {"us_median": statistics.median(per_call), "us_min": min(per_call), "us_max": max(per_call),
            "trials": len(per_call), "copies": len(states)}


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return ((y.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/layer_matrix/layer_matrix.json")
    ap.add_argument("--rotate-mb", type=float, default=768.0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--max-copies", type=int, default=1024)
    args = ap.parse_args()

    from vllm import _custom_ops as ops
    torch.manual_seed(0)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    batches = [int(b) for b in args.batches.split(",")]
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "group": GROUP,
              "rotate_mb": args.rotate_mb, "reps": args.reps, "trials": args.trials, "batches": batches,
              "method": "CUDA graph of one call per weight copy, rotated over >= rotate_mb; us per call",
              "shapes": {}}
    import vllm
    result["vllm"] = vllm.__version__
    if os.path.exists(args.out):        # merge: each shape may run in its own process (WSL2 allocator limits)
        with open(args.out) as f:
            prev = json.load(f)
        if prev.get("batches") == batches and prev.get("rotate_mb") == args.rotate_mb:
            result["shapes"].update(prev.get("shapes", {}))

    for name in args.shapes.split(","):
        N, K = SHAPES[name]
        impls = build_impls(N, K)
        base = make_awq(N, K, g, dev)
        ref_w = ops.awq_dequantize(base["qweight"], base["scales"], base["qzeros"], 0, 0, 0).float()   # [K, N]
        shape_res = {"N": N, "K": K, "conversion_ms": {}, "max_rel_err": {}, "cells": {}}
        for impl in impls:
            # one-time conversion cost (median of 5, synchronized)
            ts = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                st0 = impl.prepare(base)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            shape_res["conversion_ms"][impl.name] = statistics.median(ts)
            # correctness on the base copy
            errs = []
            for B in (1, max(b for b in batches if impl.batches is None or b in impl.batches)):
                x = torch.randn((B, K), generator=g, device=dev, dtype=torch.float16)
                errs.append(rel_err(impl.call(x, st0), x.float() @ ref_w))
            shape_res["max_rel_err"][impl.name] = max(errs)
            # rotated copies
            nbytes = state_bytes(st0)
            copies = int(min(args.max_copies, max(4, -(-args.rotate_mb * 2**20 // nbytes))))
            states = [st0] + [impl.prepare(make_awq(N, K, g, dev)) for _ in range(copies - 1)]
            torch.cuda.synchronize()
            for B in batches:
                key = f"{impl.name}/B{B}"
                if impl.batches is not None and B not in impl.batches:
                    shape_res["cells"][key] = None
                    continue
                x = torch.randn((B, K), generator=g, device=dev, dtype=torch.float16)
                shape_res["cells"][key] = time_cell(impl, states, x, args.reps, args.trials)
                c = shape_res["cells"][key]
                print(f"{name:8s} {impl.name:7s} B={B:3d}  {c['us_median']:8.1f} us  "
                      f"[{c['us_min']:.1f}, {c['us_max']:.1f}]  copies={copies}", flush=True)
            del states, st0
            torch.cuda.empty_cache()
        result["shapes"][name] = shape_res
        print(f"{name}: conversion ms {shape_res['conversion_ms']}  max rel err {shape_res['max_rel_err']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
