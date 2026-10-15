# Turning results into a post that engineers respect

What lands with GPU and systems engineers: a specific claim, a chart, an honest caveat, and code they can run.
What does not: superlatives, unexplained speedups, or numbers with no method behind them.

**Rule: fill every `{...}` below from `results/.../report/report.md` or `kernel_bench.json`. If a number is not
in a generated file, it does not go in the post.**

## LinkedIn post (about 150 words)

> I built Inferno, a serving testbed that measures LLM inference from the CUDA kernel up to the request router.
>
> Three things I measured on an 8 GB laptop GPU (and a {N}-GPU cloud box):
>
> 1. A hand-written fused int4 GEMV kernel reaches {X}% of the theoretical DRAM bandwidth, {Y}x faster than fp16
>    cuBLAS but {Z}x {slower/faster} than {library kernel}. Nsight says the gap is {counter-based reason}.
> 2. MPI request routing adds {A} us per hop. Coordination overhead is {B}% of a request at {N} replicas.
> 3. A load- and length-aware router over pre-loaded replicas changed p99 by {P}% (paired, n={n}, Holm p={p})
>    on bursty load, and did {not} help on steady load, as predicted.
>
> Everything is reproducible: raw logs, seeded workloads, paired statistics, one-command reruns.
> Repo and write-up: {link}

Attach: the kernel bandwidth chart (hero) or the p99 comparison figure. Post the replay dashboard as a short
screen recording.

## Longer write-up outline (blog / dev.to / arXiv-style)

1. The question and why it matters (one paragraph)
2. Method: open-loop load, paired seeded streams, what "best fixed baseline" means
3. Kernel: the optimization ladder as a table with per-stage effective bandwidth; profiler evidence for each jump
4. Routing: MPI vs HTTP overhead, scaling curve, the crossover
5. Adaptive controller: design, decision timeline, results including where it does not help
6. Threats to validity (copy the list from `docs/methodology.md`)
7. Reproduce it: three commands

## Resume bullets (choose the ones your data supports)

- Wrote a fused int4-dequantize GEMV CUDA kernel reaching {X}% of DRAM bandwidth ({Y}x over fp16 cuBLAS) on an
  {GPU}; profiled with Nsight Compute to attribute the remaining gap to {reason}.
- Built an open-loop LLM serving benchmark (TTFT/TPOT/p99, goodput, tokens/J) with paired seeded workloads and
  Holm-corrected significance testing; found {headline result with CI}.
- Designed an MPI request router with per-hop timestamps; measured {A} us/hop overhead and a scaling crossover at
  {N} replicas versus an HTTP router baseline.
- Implemented a hysteresis-based adaptive routing controller over heterogeneous vLLM replicas; {p99 change} versus the best fixed
  configuration across three load patterns (report the sign honestly).
- Automated multi-GPU cloud benchmarking with Terraform and a self-terminating cost cap.

## Where to share

LinkedIn (chart + 3 numbers + link), Hacker News (Show HN, lead with the kernel or the negative result), r/LocalLLaMA
(the laptop-GPU angle), NVIDIA Developer Forums (kernel section, Nsight findings), a Hugging Face Space or GitHub
Pages for the showcase site (`make site`, then host `docs/demo/`).
