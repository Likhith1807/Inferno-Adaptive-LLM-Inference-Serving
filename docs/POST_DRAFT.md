# Post and resume draft (numbers generated from data; edit the voice, not the figures)

## LinkedIn post

I built Inferno, a testbed that measures LLM inference from a CUDA kernel up to the request router, on an 8 GB laptop GPU. What I found, including where my own idea failed:

1. A hand-written fused int4 GEMV kernel hits 95% of DRAM bandwidth and runs 3.67x faster than fp16 cuBLAS on llama7B_mlp_up (byte-ratio ceiling 3.76x). Inside vLLM it matches the default Marlin path at one sequence and loses at higher concurrency: I first measured it beating Marlin, then caught that my Marlin baseline was wrong.
2. My adaptive routing policy lost to a fixed config: p99 +112% to +204% (8 paired reps, Holm-corrected). A controlled experiment showed why: a second *active* replica on the same GPU multiplies p99 by 2.1x. Replicas need separate devices.
3. An MPI router adds ~2 us per hop in isolation and nothing measurable end to end versus plain HTTP.

Every number comes from a script over raw logs, with paired statistics and reproducible runs. Negative results are in the report. Repo: {link}

Attach: `docs/figures/kernel_speedup.png` (hero) or `docs/results/contention/contention_light.png`.

## Resume bullets

- Wrote a fused int4-dequantize GEMV CUDA kernel reaching 95% of DRAM bandwidth (3.67x over fp16 cuBLAS) on RTX 4060; validated against a CPU reference, benchmarked with L2-defeating rotation (and a correction when 6x-L2 turned out to leak), run-to-run spread; integrated into vLLM as a torch custom op and compared end to end against Marlin and AWQ kernels (parity at batch 1).
- Built an open-loop LLM serving benchmark (TTFT/TPOT/p99, goodput, tokens/J) with paired seeded workloads and Holm-corrected tests; pre-registered and rejected an adaptive-routing hypothesis, then isolated the cause (GPU co-residency contention) with a controlled experiment.
- Designed an MPI request router with per-hop timestamps and a GPU-free microbenchmark; found it indistinguishable from an HTTP router end to end and traced a 1000x gap between isolated and in-situ hop latency to event-loop scheduling.
- Automated environment setup (date-pinned CUDA 12.x stack, no-root toolchain) and multi-GPU cloud benchmarking with a self-terminating cost cap (Terraform; not yet applied).