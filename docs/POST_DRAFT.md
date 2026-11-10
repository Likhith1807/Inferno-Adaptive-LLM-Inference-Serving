# Post and resume draft (numbers generated from data; edit the voice, not the figures)

## LinkedIn post

I built Inferno, a testbed that measures LLM inference from a CUDA kernel up to the request router, on an 8 GB laptop GPU. What I found, including where my own idea failed:

1. A hand-written fused int4 GEMV kernel hits 95% of DRAM bandwidth and runs 3.67x faster than fp16 cuBLAS on llama7B_mlp_up (byte-ratio ceiling 3.76x). Inside vLLM it matches the default Marlin path at one sequence and loses at higher concurrency: I first measured it beating Marlin, then caught that my Marlin baseline was wrong.
2. My adaptive routing policy lost to a fixed config: p99 +112% to +204% (8 paired reps, Holm-corrected). A controlled experiment showed why: a second *active* replica on the same GPU multiplies p99 by 2.1x. Replicas need separate devices.
3. An MPI router adds ~2 us per hop in isolation and nothing measurable end to end versus plain HTTP.

Every number comes from a script over raw logs, with paired statistics and reproducible runs. Negative results are in the report. Repo: {link}

Attach: `docs/figures/kernel_speedup.png` (hero) or `docs/results/contention/contention_light.png`.

## Resume bullets

- Implemented a fused INT4 dequantization + matrix-vector CUDA kernel, verified against a double-precision CPU reference; measured 3.67x over FP16 cuBLAS on a 11008 x 4096 layer at batch 1 (95% of peak DRAM bandwidth; byte-ratio ceiling 3.76x).
- Benchmarked the kernel against vLLM's Marlin and AWQ paths on every Qwen2.5 decode layer at batch 1-64; it won only layers worth 15% of decode GEMM time and lost up to 7x at batch 16, so I kept the library path and redirected effort; the harness's reference check also flagged batch-dependent wrong results from Marlin on this machine.
- Built an LLM inference benchmark spanning CUDA, vLLM and MPI on an 8 GB RTX 4060; ran a 96-run paired campaign that rejected an adaptive-routing hypothesis. A controlled 5-rep study isolated the cause: at 60% load, activating a second GPU-sharing replica raised p99 from 6.21 s to 13.02 s, while loading it idle gave 6.26 s.
- Designed an MPI request router with per-hop timestamps and a GPU-free microbenchmark; found it indistinguishable from an HTTP router end to end and traced a 1000x gap between isolated and in-situ hop latency to event-loop scheduling.
- Automated environment setup (date-pinned CUDA 12.x stack, no-root toolchain) and multi-GPU cloud benchmarking with a self-terminating cost cap (Terraform; not yet applied).