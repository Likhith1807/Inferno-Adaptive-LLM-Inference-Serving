"""
End-to-end decode benchmark with the custom kernel inside vLLM.

One invocation = one engine configuration in a fresh process (so GPU memory and compilation state never leak
between configurations). It measures

  * decode throughput (output tokens/s) with 1, 2, 4 and 8 concurrent sequences, each generating 256 tokens with
    ``ignore_eos`` from a short prompt, so the time is decode-dominated; median of ``--reps`` timed repetitions;
  * the greedy token ids of the first prompt, for a token-level agreement check against another configuration;
  * how many linear-layer calls used the custom kernel (proof that the patch was actually exercised).

Configurations (all Qwen2.5-1.5B-Instruct on the same GPU, one process per configuration):
    fp16          unquantized weights
    awq_marlin    AWQ-Int4 with vLLM's Marlin kernel (the default vLLM picks for this checkpoint on this GPU)
    awq_classic   AWQ-Int4 with the original AWQ GEMM kernel
    awq_inferno   AWQ-Int4 with the custom kernel for decode steps of <= --max-batch tokens (stock kernel otherwise)

Engine modes: ``eager`` (no CUDA graphs, every kernel launched from Python) and ``graphs`` (CUDA graphs captured for
decode). Compare configurations only within the same mode.

    python -m inferno.experiments.run_e2e_kernel --config awq_inferno --mode eager --out results/e2e_kernel/x.json
"""
import argparse
import json
import os
import statistics
import time

PROMPTS = [
    "Explain in one paragraph how a GPU executes a matrix multiplication.",
    "List three reasons why decoding is memory bound.",
    "Write a short note about why quantization reduces latency.",
    "Describe the roofline model in two sentences.",
    "What is a warp in CUDA?",
    "Why does batching improve throughput?",
    "Give an example of a bank conflict.",
    "How does paged attention save memory?",
]
MODELS = {"fp16": ("Qwen/Qwen2.5-1.5B-Instruct", None), "awq_marlin": ("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "awq_marlin"),
          "awq_classic": ("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "awq"), "awq_inferno": ("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "awq")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(MODELS))
    ap.add_argument("--mode", default="eager", choices=["eager", "graphs"])
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"     # engine in this process, so the patch applies
    import torch

    if args.config == "awq_inferno":
        from inferno.kernels import vllm_int4
        vllm_int4.install(max_batch=args.max_batch)
    from vllm import LLM, SamplingParams

    model, quant = MODELS[args.config]
    kw = dict(model=model, quantization=quant, dtype="float16", max_model_len=512, gpu_memory_utilization=0.5,
              max_num_seqs=16, seed=0)
    if args.mode == "eager":
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    t0 = time.time()
    llm = LLM(**kw)
    startup = time.time() - t0
    sp = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True)

    layer_err = None
    if args.config == "awq_inferno":
        from inferno.kernels import vllm_int4
        errs = []
        model_obj = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model \
            if hasattr(llm.llm_engine, "model_executor") else None
        if model_obj is None:
            model_obj = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
        for _, m in model_obj.named_modules():
            if hasattr(m, "inferno_w"):
                errs.append(vllm_int4.verify_layer(m))
        layer_err = {"n_layers_checked": len(errs), "max_rel_err": max(errs) if errs else None}

    llm.generate(PROMPTS[:1], SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True))     # warm-up
    ids = llm.generate(PROMPTS[:1], sp)[0].outputs[0].token_ids
    results = {}
    for c in [int(x) for x in args.concurrency.split(",")]:
        prompts = (PROMPTS * 2)[:c]
        llm.generate(prompts, SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True))     # warm this batch size
        rates = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            outs = llm.generate(prompts, sp)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            rates.append(sum(len(o.outputs[0].token_ids) for o in outs) / dt)
        results[str(c)] = {"tok_per_s_median": statistics.median(rates), "tok_per_s_all": rates}
        print(f"[{args.config}/{args.mode}] concurrency {c}: {statistics.median(rates):.1f} tok/s  {['%.0f' % r for r in rates]}", flush=True)

    stats = {}
    if args.config == "awq_inferno":
        from inferno.kernels import vllm_int4
        stats = vllm_int4.stats()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"config": args.config, "mode": args.mode, "model": model, "quantization": quant, "max_batch": args.max_batch,
               "tokens": args.tokens, "reps": args.reps, "startup_sec": startup, "gpu": torch.cuda.get_device_name(0),
               "results": results, "first_prompt_token_ids": list(ids), "layer_check": layer_err, "kernel_calls": stats},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
