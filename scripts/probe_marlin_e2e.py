"""End-to-end check of the Marlin batch-size fault on the real Qwen2.5-1.5B-AWQ checkpoint.

Greedy decode of the same prompts at batch 1 (reference) and at batch B (the prompt repeated B times, plus distinct
prompts). With a correct kernel almost every batched copy reproduces the batch-1 text (small batch-variance
divergences late in the sequence are normal); a kernel with 10-20% errors at that batch size produces divergent or
degenerate text early.

    python scripts/probe_marlin_e2e.py --quantization awq_marlin --batches 1,16,32,48,64
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the factorial of n.\n",
    "Explain in one paragraph why the sky is blue.",
    "List three prime numbers greater than 100:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantization", default="awq_marlin")
    ap.add_argument("--batches", default="1,16,32,48,64")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    kw = dict(model="Qwen/Qwen2.5-1.5B-Instruct-AWQ", quantization=args.quantization, dtype="float16",
              max_model_len=512, gpu_memory_utilization=0.6, max_num_seqs=128, seed=0)
    if args.eager:
        kw["enforce_eager"] = True
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    ref = {p: llm.generate([p], sp, use_tqdm=False)[0].outputs[0].text for p in PROMPTS}
    res = {"quantization": args.quantization, "eager": args.eager, "max_tokens": args.max_tokens, "batches": {}}
    for B in [int(b) for b in args.batches.split(",")]:
        batch = [PROMPTS[i % len(PROMPTS)] for i in range(B)]
        outs = llm.generate(batch, sp, use_tqdm=False)
        exact = sum(o.outputs[0].text == ref[p] for o, p in zip(outs, batch))
        # first-divergence position in characters, averaged over mismatches
        div = []
        for o, p in zip(outs, batch):
            a, b = o.outputs[0].text, ref[p]
            if a != b:
                k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                div.append(k)
        bad_example = next((o.outputs[0].text for o, p in zip(outs, batch) if o.outputs[0].text != ref[p]), None)
        res["batches"][B] = {"exact_match": exact, "n": B, "mean_first_divergence_char": (sum(div) / len(div)) if div else None,
                             "example_mismatch": bad_example[:160] if bad_example else None}
        print(f"{args.quantization:10s} eager={args.eager} B={B:3d} exact={exact}/{B} "
              f"first_div={res['batches'][B]['mean_first_divergence_char']}  ex={bad_example[:80] if bad_example else ''!r}",
              flush=True)
    res["reference"] = ref
    out = args.out or f"results/marlin_fault/e2e_{args.quantization}{'_eager' if args.eager else ''}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
