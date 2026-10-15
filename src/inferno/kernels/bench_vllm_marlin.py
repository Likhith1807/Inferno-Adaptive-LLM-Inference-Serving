"""
Measure the quantized GEMM kernels vLLM *actually runs* during batch-1 decode of the Qwen2.5-1.5B AWQ model,
in situ, with torch.profiler. This is the fair "library" comparison for the custom kernel: same model, same layer
shapes, real serving path, weights streamed from DRAM (28 distinct layers exceed L2), no synthetic setup that could
favour either side.

Two library paths are measurable:
    --quantization awq_marlin   Marlin kernels (vLLM's fast path on Ampere+; what serving should use)
    --quantization awq          the original AutoAWQ-derived GEMM kernel plus its split-K reduce kernel

Method: run the engine in-process (V1 multiprocessing disabled), eager mode (so kernels are individually visible),
profile several decode steps, keep the target kernels, drop the first step (prefill), and group the remaining calls
by their position within a layer (qkv, o_proj, gate_up, down: the order vLLM executes them). If the call count does
not divide evenly into 4 kernels per layer, the script says so and reports only the pooled median instead of
guessing a role mapping. For the AWQ path each GEMM's time includes the reduce kernel that immediately follows it.

    python -m inferno.kernels.bench_vllm_marlin --quantization awq_marlin --out results/vllm_awq_marlin_decode.json
    python -m inferno.kernels.bench_vllm_marlin --quantization awq_inferno --mode graphs   # custom kernel, CUDA graphs
"""
import argparse
import json
import os
import statistics

ROLES = ["qkv", "o_proj", "gate_up", "down"]
KERNEL_SUBSTR = {"awq_marlin": "arlin", "awq": "gemm_forward_4bit", "awq_inferno": "gemv_int4_lane"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    ap.add_argument("--quantization", default="awq_marlin", choices=sorted(KERNEL_SUBSTR))
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=24, help="decode tokens to profile")
    ap.add_argument("--mode", default="eager", choices=["eager", "graphs"],
                    help="eager: kernels launched one by one from Python (GPU idles between launches, clocks may drop); "
                         "graphs: CUDA-graph decode, the GPU stays busy like it does in real serving")
    args = ap.parse_args()
    out_path = args.out or f"results/vllm_{args.quantization}_decode{'_graphs' if args.mode == 'graphs' else ''}.json"

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # keep the engine core in this process
    import torch
    from torch.profiler import ProfilerActivity, profile
    if args.quantization == "awq_inferno":
        from inferno.kernels import vllm_int4
        vllm_int4.install(max_batch=8)
    from vllm import LLM, SamplingParams

    kw = dict(model=args.model, quantization="awq" if args.quantization == "awq_inferno" else args.quantization, dtype="float16",
              max_model_len=512, gpu_memory_utilization=0.5, max_num_seqs=16, seed=0)
    if args.mode == "eager":
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    llm = LLM(**kw)
    n_layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
    sp = SamplingParams(max_tokens=args.steps, temperature=0.0, ignore_eos=True)
    prompt = ["system data model batch queue latency memory kernel " * 2]
    llm.generate(prompt, sp)                       # warmup / compile / autotune

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate(prompt, sp)
        torch.cuda.synchronize()

    def dur_us(e):
        v = getattr(e, "device_time", None)
        return v if v is not None else e.cuda_time

    all_events = list(prof.events())
    cuda_events = [e for e in all_events if getattr(getattr(e, "device_type", None), "name", "") == "CUDA"
                   and dur_us(e) > 0]
    cuda_events.sort(key=lambda e: e.time_range.start)
    by_name = {}
    for e in cuda_events:
        n, t = by_name.get(e.name, (0, 0.0))
        by_name[e.name] = (n + 1, t + dur_us(e))
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:12]

    substr = KERNEL_SUBSTR[args.quantization]
    kernel_times = []                              # (duration us) per target kernel, in execution order
    for i, e in enumerate(cuda_events):
        if substr not in e.name:
            continue
        t = dur_us(e)
        if args.quantization == "awq":             # split-K: add the reduce kernel that follows this GEMM
            nxt = cuda_events[i + 1] if i + 1 < len(cuda_events) else None
            if nxt is not None and "reduce_kernel" in nxt.name:
                t += dur_us(nxt)
        kernel_times.append(t)

    per_step = n_layers * len(ROLES)
    result = {"model": args.model, "quantization": args.quantization, "mode": args.mode, "layers": n_layers,
              "target_kernel_substring": substr, "n_target_kernels": len(kernel_times),
              "expected_per_step": per_step, "gpu": torch.cuda.get_device_name(0),
              "profiler_events_total": len(all_events), "profiler_cuda_kernel_events": len(cuda_events),
              "top_kernels_by_total_us": [{"name": n[:120], "calls": c, "total_us": t} for n, (c, t) in top]}

    decode = kernel_times[per_step:]               # drop the prefill step
    if decode and len(decode) % per_step == 0:
        by_role = {r: [] for r in ROLES}
        for i, t in enumerate(decode):
            by_role[ROLES[i % len(ROLES)]].append(t)
        result["roles"] = {r: {"us_median": statistics.median(v), "us_min": min(v), "us_max": max(v), "n": len(v)}
                           for r, v in by_role.items()}
        result["role_mapping"] = "by position within layer: " + ", ".join(ROLES)
    else:
        result["roles"] = None
        result["pooled_us_median"] = statistics.median(kernel_times) if kernel_times else None
        result["warning"] = ("Target kernel count is not a multiple of 4 x layers after dropping prefill; "
                             "role attribution skipped rather than guessed.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "top_kernels_by_total_us"}, indent=2))


if __name__ == "__main__":
    main()
