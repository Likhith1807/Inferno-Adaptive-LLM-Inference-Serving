"""Chapter 6 (system design and architecture) and Chapter 7 (methodology)."""
from __future__ import annotations

from . import data
from .refs import cite
from .sec_lit import _Out
from .style import BUL, CALLOUT, EQ, FIG, H, NUM, P, TABLE


# ============================================================================================ chapter 6
def chapter_design(figs):
    F = data.facts()
    out = _Out(H(1, "System design and architecture"))
    out.append(P(
        "This chapter describes what was built. Design decisions are given with the constraint that forced them, because on an 8&nbsp;GB "
        "consumer GPU most of them were forced rather than chosen."))

    out += H(2, "Design constraints")
    out.append(TABLE([
        ["Constraint", "Consequence for the design"],
        ["8&nbsp;GB of GPU memory, shared with the Windows display stack", "A 1.5B-parameter model family; two replicas at 0.36 of GPU memory each (0.72 total, leaving headroom for the display and for vLLM's start-up memory check)."],
        ["A running vLLM server cannot change its batch cap or quantization", "Adaptivity is routing across pre-loaded heterogeneous replicas; reconfiguration is measured, not used (about " + f"{F['cold']:.0f}" + " s per restart)."],
        ["One physical GPU", "Replicas time-slice the device. This is a defining property of the study, not a footnote, and gets its own controlled experiment (Section 9.6)."],
        ["Windows host, Linux toolchain in WSL2", "All measurements run inside WSL2; GPU performance counters are gated by a Windows driver setting, which left the kernel profiling question open."],
        ["Single operator, no cluster", "The cloud phase is provided as infrastructure code with a spend cap but was not executed; no multi-node number appears in this report."],
    ], [0.36, 0.64], "Constraints and their consequences.", font=8))

    out += H(2, "Overall architecture")
    out.append(P(
        "The figure below shows the layers. A seeded workload generator produces request streams; an open-loop client replays them through one of "
        "three front ends (a policy router in the client process, an HTTP least-outstanding baseline, or the MPI dispatcher) to a pool of resident "
        "vLLM replicas; per-request latencies and GPU power are recorded; analysis is a pure function of the recorded files."))
    out += FIG(figs["architecture"], "System architecture, from workload generation to statistical analysis.", 1.0)
    out += H(3, "Component boundaries")
    out.append(BUL([
        "<b>Policies are pure functions.</b> A routing policy receives a <font name='Mono'>PoolState</font> snapshot (in-flight counts, "
        "per-replica capacity, recent TTFT p95) and returns a replica index. There is no I/O in policy code, so every decision can be unit-tested and "
        "logged with a reason string.",
        "<b>Baselines share the pool.</b> The fixed baselines send all traffic to one replica, but both replicas stay loaded and idle replicas "
        "hold their memory. The adaptive policy therefore cannot win by being handed a better memory situation than the baseline.",
        "<b>MPI carries control and results, not tensors.</b> Point-to-point messages only; the routing and message-protocol logic has no MPI import "
        "and is tested without MPI.",
        "<b>The kernel is standalone.</b> The benchmark needs only nvcc and cuBLAS, so it builds anywhere CUDA does and can be profiled without a Python "
        "stack in the way.",
        "<b>Analysis reads files.</b> Everything after an experiment is a pure function of per-request records and a summary table, so results are "
        "re-derivable and this report is generated, not typed.",
    ]))

    out += H(2, "The replica pool")
    out.append(TABLE([
        ["Tier", "Name", "Weights", "Seqs", "Batched tokens", "GPU mem.", "Intended role"],
        ["0", "fast", "AWQ-Int4 (awq_marlin)", "16", "512", "0.36", "Low-latency decode; small prefill chunks so long prompts cannot stall running decodes for long."],
        ["1", "bulk", "AWQ-Int4 (awq_marlin)", "64", "4096", "0.36", "Throughput and long prefill; large chunks give fast prefill at the price of higher time-per-output-token jitter."],
    ], [0.06, 0.08, 0.16, 0.08, 0.12, 0.1, 0.4], "The two-tier pool used by every routing experiment (Seqs = max_num_seqs, Batched tokens = max_num_batched_tokens, GPU mem. = gpu_memory_utilization).", font=7.6))
    out.append(P(
        f"Both tiers use the same int4 weights so that the only differences are the two scheduler parameters that the chunked-prefill literature "
        f"identifies as the main throughput-latency knobs {cite('sarathi', 'vllmdocs')}. Making the tiers differ in precision as well would confound "
        f"routing with quantization effects."))

    out += H(2, "The adaptive controller")
    out.append(P(
        "The controller is a three-level, hysteresis-based state machine chosen to be interpretable and cheap. Its only mutable state is the level."))
    out += FIG(figs["controller"], "Controller levels and transitions. Escalation is immediate; de-escalation requires sustained calm.", 0.94)
    out.append(TABLE([
        ["Level", "Name", "Routing rule"],
        ["0", "calm", "Every request goes to tier 0."],
        ["1", "pressured", "Requests with at least 512 prompt tokens go to the last tier (isolating long prefills); others stay on tier 0."],
        ["2", "overloaded", "Every request goes to the least-loaded replica by in-flight count divided by capacity."],
    ], [0.08, 0.16, 0.76], "Routing rule by controller level.", font=8))
    out.append(P(
        "Once per tick (1&nbsp;s) the controller <b>escalates</b> one level if tier-0 utilization is at least 0.70 or the recent TTFT p95 exceeds the "
        "0.5&nbsp;s SLO. It <b>de-escalates</b> one level only after five consecutive calm ticks, where calm means pool utilization at most 0.25 "
        "and TTFT p95 at most half the SLO. A hard <i>spill</i> rule redirects a request to the least-loaded replica if its chosen replica is "
        "at 0.90 utilization or higher. The thresholds were fixed before the campaign and are not tuned to its results (Section 7.5)."))
    out.append(P(
        "The other policies are trivial by design: <b>fixed-fast</b> and <b>fixed-bulk</b> pin all traffic to one tier; <b>static-lb</b> routes each "
        "request to the least-utilized replica with no state and no length awareness."))

    out += H(2, "Serving harness and client")
    out.append(P(
        f"The harness launches each vLLM replica as a separate OpenAI-compatible server process {cite('vllmdocs')}, waits for readiness, and records "
        f"cold-start time. The client is an open-loop asynchronous HTTP client using streaming responses. For each request it records the scheduled "
        f"arrival time, the send time, the time of the first streamed token, the completion time and the number of output tokens. "
        f"Output length is pinned with <font name='Mono'>ignore_eos</font> so that all policies do identical work per request. GPU power is sampled "
        f"from <font name='Mono'>nvidia-smi</font> every 200&nbsp;ms in a separate process."))

    out += H(2, "The MPI router")
    out.append(P(
        f"The router is a small MPI program {cite('mpi', 'openmpi')} with one <i>dispatcher</i> at rank 0 and <i>workers</i> at ranks 1 to <i>N</i>. The "
        f"dispatcher receives requests, applies the routing policy, and sends each job to a worker with a non-blocking send. Each worker owns one replica "
        f"connection on an asyncio loop and polls for messages with a probe every 200&nbsp;&micro;s, so it can multiplex many outstanding requests; "
        f"results return to rank 0. Every hop appends a timestamp from a monotonic clock. All ranks run on one host, where the monotonic clock is shared, "
        f"so per-hop differences are exact rather than estimated (Section 7.7)."))
    out += FIG(figs["mpipath"], "Request path through the MPI router with the timestamps recorded at each hop.", 1.0)
    out.append(P(
        "An HTTP router with a least-outstanding policy is provided as the comparison, so that the question &lsquo;does MPI help&rsquo; has a control. "
        "Open MPI 5 requires <font name='Mono'>--map-by :OVERSUBSCRIBE</font> when ranks outnumber cores, which the microbenchmarks use."))

    out += H(2, "The fused int4 kernel")
    out += H(3, "Weight format")
    out.append(P(
        "Weights are asymmetric group-wise int4 with group size <i>G</i>&nbsp;=&nbsp;128, the layout that AWQ and GPTQ checkpoints use. A weight "
        "is reconstructed as <i>w</i>&nbsp;=&nbsp;<i>s</i>(<i>q</i>&nbsp;&minus;&nbsp;<i>z</i>) with <i>q</i>&nbsp;&isin;&nbsp;[0,15]. Eight nibbles are packed "
        "into each 32-bit word (nibble <i>j</i> of word <i>w</i> holds <i>k</i>&nbsp;=&nbsp;8<i>w</i>&nbsp;+&nbsp;<i>j</i>), stored row-major as "
        "<font name='Mono'>uint32[N][K/8]</font>, with fp16 scales and zero points as <font name='Mono'>[N][K/G]</font>."))
    out += FIG(figs["layout"], "Packed weight layout. Consecutive k values share a 32-bit word; each group of 128 shares one scale and one zero point.", 0.92)
    out += H(3, "The algebraic form")
    out.append(P(
        "For row <i>n</i>, substituting the dequantization into the dot product and factoring the group constants gives"))
    out += EQ(r"y_n=\sum_{g}s_{n,g}\left(\sum_{k\in g}q_{n,k}\,x_k\;-\;z_{n,g}\sum_{k\in g}x_k\right)")
    out.append(P(
        "so the inner loop multiplies raw nibbles by activations and the zero-point correction reuses a sum of activations that is the same for every "
        "row. The final implementation accumulates that activation sum inside the main loop at the granularity of the chunk each lane processes "
        "(Section 7.4 explains why an earlier group-level version was wrong). Versions v5 and v6 keep the same algebra."))
    out += H(3, "Integration into vLLM")
    out.append(P(
        f"To measure the kernel where it matters, it was patched into vLLM 0.11 (<font name='Mono'>src/inferno/kernels/vllm_int4.py</font>). The kernel is built as a "
        f"shared library with a C entry point and called through <font name='Mono'>ctypes</font> on PyTorch&rsquo;s current stream; it is registered as a torch custom op with a fake "
        f"implementation so that CUDA-graph capture can record it. At model load, the AutoAWQ tensors that vLLM holds (four-bit values packed along the output axis in an "
        f"interleaved order) are repacked once into the kernel&rsquo;s layout. The AWQ linear method is patched so that a decode step of at most a configurable number of tokens uses "
        f"the custom kernel and everything else, including prefill, falls back to vLLM&rsquo;s own path {cite('vllmdocs')}. Nothing else in vLLM changes."))
    out += H(3, "The optimization ladder")
    out += FIG(figs["access"], "Memory access pattern of the thread-per-row kernel (v0) versus the warp-per-row kernel (v1).", 0.95)
    out.append(TABLE([
        ["Version", "Change", "Motivation"],
        ["v0", "One thread per row", "Baseline; each lane strides through its own row, so a warp&rsquo;s loads are uncoalesced."],
        ["v1", "One warp per row; 32-bit loads; shuffle reduction", "Adjacent lanes read adjacent words: coalesced."],
        ["v2", "128-bit loads; activations staged in shared memory (half precision)", "Fewer, wider load instructions; but lane stride creates a predicted 16-way shared-memory bank conflict."],
        ["v3", "v2 with one pad word per 32 activations", "Shifts lanes across banks and removes the conflict."],
        ["v4", "Algebraic form; streaming (<font name='Mono'>__ldcs</font>) loads; two loads in flight", "Cut per-weight instructions; avoid polluting L2 with data read once; more memory-level parallelism."],
        ["v5", "Register reuse: each 32-weight chunk decoded once into registers; activations read through the read-only cache (no shared-memory staging)", "Removes the per-block staging phase and the shared-memory footprint; the same code serves B = 1..16 activation vectors."],
        ["v6", "v5 with 16 lanes per row (two rows per warp)", "K = 1536 has 48 chunks per row: 32 lanes leave half with two chunks and half with one; 16 lanes give every lane exactly three."],
    ], [0.11, 0.38, 0.51], "The kernel optimization ladder. Each step changes one thing.", font=8))
    out += FIG(figs["banks"], "Shared-memory bank conflict before and after padding (predicted; the profiler counter that would confirm it was not obtainable, Section 11.2). Versions v5 and v6 no longer stage activations in shared memory.", 0.95)
    out.append(P(
        f"Design source: shared-memory banking, coalescing and warp-shuffle practice follow the NVIDIA programming and best-practices guides "
        f"{cite('cudaguide')}. The 16-way figure is an analytical prediction from the access pattern; this report does not claim that a profiler confirmed it."))
    return out


# ============================================================================================ chapter 7
def chapter_methodology(figs):
    F = data.facts()
    cal = F["cal"]
    out = _Out(H(1, "Methodology"))
    out.append(P(
        "This chapter is the core of the report. It states every protocol decision, the reason for it, and what it protects against. The "
        "organizing principle is that a comparison is only as credible as the weakest thing that could differ between its arms."))

    # ---------------------------------------------------------------- 7.1
    out += H(2, "Principles")
    out.append(NUM([
        "<b>Pre-specification.</b> Hypotheses, metrics, decision rules and policy thresholds are fixed before the confirmatory data are collected, and "
        "the verdict text is produced by code from the data, not written by the author.",
        "<b>Strongest available baseline.</b> The comparator for the adaptive policy is the best fixed configuration <i>chosen on the tested metric</i>, "
        "in the same pool, not a weak default.",
        "<b>Pairing.</b> Repetition <i>k</i> uses byte-identical request streams for all policies, so differences are within-stream.",
        "<b>Randomized order.</b> The order in which policies run inside each repetition is shuffled with a recorded seed to cancel drift.",
        "<b>Instrument checked before results are trusted.</b> Kernels pass a correctness gate before timing; the measurement code has unit tests; "
        "the metric code has unit tests with hand-computed expected values.",
        "<b>Report negative results.</b> If the pre-specified hypothesis fails, it is reported as failed, with a mechanism tested by a separate controlled experiment.",
        "<b>No hand-typed numbers.</b> Every number in this report is read from a raw result file by the scripts that build it.",
    ]))
    out += FIG(figs["pipeline"], "Measurement and analysis pipeline. Arrows are file boundaries, so each stage can be re-run independently.", 1.0)

    # ---------------------------------------------------------------- 7.2
    out += H(2, "Workload model and load calibration")
    out += H(3, "Arrival processes")
    out.append(P(
        f"All experiments use an <b>open-loop</b> client: a request is issued at its scheduled time regardless of whether earlier requests have "
        f"finished, and its latency is measured from the scheduled time. This follows Schroeder et&nbsp;al. {cite('openclosed')}: a closed-loop client "
        f"slows its own arrivals as latency grows and so hides exactly the queueing this study is about. Three patterns are used:"))
    out += FIG(figs["openclosed"], "Open-loop versus closed-loop load generation. Under overload only the open-loop client exposes queue growth.", 0.85)
    out.append(TABLE([
        ["Pattern", "Arrival process", "Request lengths"],
        ["steady", "Poisson at a constant rate", "Lognormal prompt (median 128 tokens, sigma 0.8) and output (median 256, sigma 0.6), capped at 1536 prompt and 512 output tokens."],
        ["bursty", "Poisson at a base rate, with a burst of higher rate for 10 s every 40 s", "As steady."],
        ["mixed-length", "Poisson at a constant rate", "Two-component mixture: 85% short (prompt median 96, output median 128) and 15% long (prompt median 900, output median 384)."],
    ], [0.13, 0.32, 0.55], "Load patterns. Rates are set by calibration (below).", font=8))
    out.append(P(
        "Prompt text is synthetic and produced to a target token count; the study does not measure model output quality, and output length is "
        "fixed with <font name='Mono'>ignore_eos</font>. Section 7.9 discusses what that does and does not license."))
    out += H(3, "Capacity calibration")
    out.append(P(
        f"A guessed request rate is a hidden free parameter: the first smoke run at a guessed rate saturated the GPU, so tail latency measured queue growth "
        f"rather than routing quality. Rates are therefore <i>derived</i> from measured capacity. The calibration script drives one reference replica "
        f"(the bulk tier) open-loop at a rate well above what it can serve, counts the output tokens completed inside a window that starts after "
        f"warm-up, and converts tokens per second to requests per second using the mean output length of the length profile (computed from 20,000 "
        f"sampled lengths after caps). Every load pattern is then a fixed fraction &rho; of that capacity:"))
    out.append(TABLE([
        ["Pattern", "Measured capacity (req/s)", "Rule", "Resulting rate (req/s)"],
        ["steady", f"{cal['steady_rps']:.2f}", "0.60 &times; capacity", f"{cal['rates']['steady']['rate_requests_per_sec']:.3f}"],
        ["bursty (base)", f"{cal['steady_rps']:.2f}", "0.35 &times; capacity", f"{cal['rates']['bursty']['base_rate_requests_per_sec']:.3f}"],
        ["bursty (burst)", f"{cal['steady_rps']:.2f}", "1.50 &times; capacity", f"{cal['rates']['bursty']['burst_rate_requests_per_sec']:.3f}"],
        ["mixed-length", f"{cal['mixed_rps']:.2f}", "0.60 &times; its own capacity", f"{cal['rates']['mixed_length']['rate_requests_per_sec']:.3f}"],
    ], [0.22, 0.26, 0.26, 0.26], "Calibrated arrival rates, generated from the calibration file.", font=8.4, align=["LEFT", "RIGHT", "LEFT", "RIGHT"]))
    out.append(P(
        "The bursty pattern deliberately exceeds measured capacity during bursts, since burst absorption is the case adaptive routing is supposed to help. "
        "Steady and mixed-length load sit at 60% of one replica&rsquo;s capacity, where a well-tuned fixed configuration should be adequate, which "
        "makes them the fairness check: an adaptive policy that loses there is paying overhead for nothing."))

    # ---------------------------------------------------------------- 7.3
    out += H(2, "Metric definitions")
    out.append(TABLE([
        ["Metric", "Definition", "Notes"],
        ["TTFT", "First-token time minus scheduled arrival", "Includes client-side and server-side queueing."],
        ["TPOT", "(Completion time minus first-token time) / (output tokens minus 1)", "Mean inter-token time for the request."],
        ["End-to-end latency", "Completion time minus scheduled arrival", "p50, p95, p99 over requests after warm-up."],
        ["Throughput", "Completed output tokens / busy span", "Busy span = max(arrival window, time of last completion). Drain time is included, so a slow configuration cannot report the offered load."],
        ["Goodput", "Requests meeting both TTFT &le; 0.5 s and TPOT &le; 0.05 s, per second", "SLO-conditioned, following DistServe."],
        ["Energy, tokens per joule", "Trapezoidal integral of sampled power; tokens divided by joules", "200 ms sampling; whole-GPU power, including the idle replica."],
        ["Warm-up exclusion", "Requests scheduled in the first 10 s are dropped", "Plus a pre-run warm-up burst."],
    ], [0.17, 0.42, 0.41], "Metric definitions.", font=7.8))
    out += EQ(r"\mathrm{Goodput}=\frac{1}{T}\sum_i\mathbf{1}\left[\mathrm{TTFT}_i\leq0.5\,\mathrm{s}\ \wedge\ \mathrm{TPOT}_i\leq0.05\,\mathrm{s}\right]")
    out.append(P(
        f"Percentiles use linear interpolation (NumPy default). Tail percentiles come from few samples per run (a 75&nbsp;s run at a few requests per second yields "
        f"hundreds of requests), so p99 is noisy within a run; the study therefore aggregates across eight repetitions and reports confidence intervals "
        f"rather than reading any single run. This is the practice recommended for tail metrics {cite('tail', 'hoefler')}."))
    out.append(CALLOUT("Two metric bugs found and fixed",
                       "Throughput was originally tokens divided by the arrival window, which made every policy that eventually finished report the "
                       "offered load. A related early bug let an asyncio timer wake slightly early on Windows and produce negative TTFT; the client now "
                       "loops until the scheduled time. Both are covered by regression tests (Chapter 8)."))

    # ---------------------------------------------------------------- 7.4
    out += H(2, "Kernel methodology")
    out += H(3, "Roofline framing and the byte-ratio ceiling")
    out.append(P(
        f"At batch size 1 the matrix-vector product is memory-bound {cite('roofline')}, so speedup over the fp16 path is limited by the ratio of bytes that must "
        f"cross DRAM, not by arithmetic. Counting weights, scales, zero points, the input vector and the output vector, with fp16 scales and zeros and "
        f"group size <i>G</i>:"))
    out += EQ(r"\mathrm{ceiling}=\frac{2NK+2K+2N}{NK/2+4NK/G+2K+2N}\quad\left(G=128:\ \approx 3.76\times\right)")
    out.append(P(
        f"The commonly quoted &lsquo;4&times;&rsquo; ignores the scale and zero-point bytes ({4 / 128:.4f} bytes per weight at <i>G</i>&nbsp;=&nbsp;128). "
        f"Reporting speedup <i>relative to this per-shape ceiling</i> is stricter and more informative than reporting it against 4&times;. An earlier "
        f"draft of this project used 4&times;; it was corrected after the calculation above (Section 7.10)."))
    out += H(3, "Correctness gate")
    out.append(P(
        "Every kernel on every shape is checked against a double-precision CPU reference on the same quantized data before any timing is recorded. "
        "The gate is a relative error below 5&times;10<super>&minus;3</super> (the output is fp16, whose rounding alone is about 5&times;10<super>&minus;4</super>). "
        "A kernel that fails is reported as INCORRECT and its time is not used. This gate caught the first version of v4, which subtracted the "
        "zero-point term of a whole 128-weight group once per 32-weight chunk and so over-corrected by roughly four times; the reported relative error was about "
        "2 to 3."))
    out += H(3, "Timing discipline")
    out.append(BUL([
        "<b>Defeating the cache.</b> The benchmark allocates several copies of the weights and rotates through them, so total footprint is at least "
        f"max(768&nbsp;MB, 24 &times; L2) and every launch streams from DRAM (L2 is {F['l2_mb']:.0f}&nbsp;MB on this GPU). Without this, small layers would be served from L2 and would "
        "report bandwidth above the DRAM peak. The first version used 192&nbsp;MB (6 &times; L2); a sweep of the footprint showed that this let the L2 serve a few percent of the "
        "traffic, which inflated every bandwidth figure (Section 7.10, Table in Section 9.1).",
        f"<b>Achievable bandwidth.</b> The theoretical peak is computed from the reported memory clock and bus width. A read-only streaming kernel over a 512&nbsp;MB buffer is "
        f"timed as well ({F['stream_gbps']:.0f}&nbsp;GB/s, {100 * F['stream_gbps'] / F['peak_gbps']:.0f}% of theoretical), and results are reported against both denominators.",
        "<b>Back-to-back launches.</b> Times are per-launch averages over batches of launches, with median, 10th and 90th percentile.",
        f"<b>Run-to-run repeat.</b> The whole benchmark is invoked {F['kernel_runs']} independent times; the reported time is the median of run medians, and "
        "the <i>spread</i> (max minus min over median) is reported next to it. Layers where fewer than four of the runs agree within 2% are flagged and not used for conclusions.",
        "<b>Effective bandwidth</b> counts only bytes that must cross DRAM and is compared with the theoretical peak "
        f"({F['peak_gbps']:.0f}&nbsp;GB/s). Comparisons between kernels moving different byte counts use time, not bandwidth.",
        "<b>Runs within 2%.</b> Alongside the spread, the number of the runs whose median lies within 2% of the reported median is stored. A single outlier run (a laptop clock state) inflates "
        "the spread but leaves the median stable; the count distinguishes that case from genuine instability.",
        "<b>Shapes.</b> Four layer shapes taken from Qwen2.5-1.5B (QKV, output projection, gate/up, down) and two from Llama-7B, chosen because the "
        "first four are what the serving experiments run and the last two are large enough to be bandwidth-dominated.",
    ]))
    out += H(3, "In-situ comparison against the vLLM kernels")
    out.append(P(
        f"An isolated microbenchmark can mislead about what a serving stack actually runs. The three int4 paths were therefore timed <i>in situ</i>: a "
        f"decode of the AWQ Qwen2.5-1.5B model in vLLM 0.11 was profiled with <font name='Mono'>torch.profiler</font>, kernels were "
        f"grouped by layer role, and kernel-time medians over the decode steps were compared. The three paths are vLLM&rsquo;s Marlin kernel "
        f"{cite('marlin', 'vllmdocs')}, its classic AWQ GEMM kernel, and the custom kernel patched into the AWQ layers, all in the same setup. "
        f"Every path was measured under <i>both</i> engine modes: eager (kernels launched one by one from Python, the GPU idling between launches) and "
        f"CUDA-graph decode (the GPU stays busy, as in real serving). Measuring both is a check, and it mattered (Section 7.10). Because single processes differ by up to about 10% "
        f"(GPU clock state), every path and mode is profiled in three fresh processes and the per-role time is the median across processes, with the range kept."))
    out += H(3, "End-to-end decode benchmark")
    out.append(P(
        "Kernel time is not token time. The end-to-end benchmark launches each configuration (fp16; AWQ with Marlin; AWQ with the classic kernel; AWQ with the custom kernel for "
        "decode steps of at most 8 tokens; and a variant limited to 2 tokens) in a <i>fresh process</i>, with CUDA-graph decode, and measures output tokens per second while 1, 2, 4 and 8 sequences "
        "decode 256 tokens each (<font name='Mono'>ignore_eos</font>, greedy sampling). Each cell is the median of five timed repetitions inside a process; the process itself is repeated three times "
        "in a different configuration order each time, and reported values are means of those medians with a <i>t</i>-based interval and the paired ratio to the reference configuration. With three "
        "process repetitions the interval is wide and no <i>p</i>-value is computed. Two checks accompany it: each patched layer of the loaded checkpoint is compared with vLLM&rsquo;s own "
        "dequantize-and-matmul on random input, and the greedy token stream of the custom configuration is compared with the stock ones. The engine runs in-process (no separate engine "
        "process) so that the patch applies to it."))
    out += H(3, "Profiler counters")
    out.append(P(
        f"The plan called for Nsight Compute counters to explain the differences between kernel versions {cite('nsight')}. On this machine the profiler "
        f"fails with <font name='Mono'>ERR_NVGPUCTRPERM</font> until a Windows driver setting is changed by an administrator. The setting was not changed "
        f"and no counter values appear in this report. Explanations of bank conflicts and of the slowdown of v4 on some shapes are therefore stated as "
        f"hypotheses, and the alternatives that were tested and rejected are documented in Chapter 9."))

    # ---------------------------------------------------------------- 7.5
    out += H(2, "Campaign design: adaptive versus fixed")
    out += H(3, "Factors, cells and repetitions")
    out.append(P(
        "The confirmatory campaign is a full factorial of four <b>policies</b> (fixed-fast, fixed-bulk, static-lb, adaptive) and three <b>load patterns</b> "
        "(steady, bursty, mixed-length), with eight <b>repetitions</b>: 4 &times; 3 &times; 8 = 96 runs of 75&nbsp;s each, of which the first 10&nbsp;s are "
        "excluded. All replicas of the pool are loaded for every policy."))
    out += FIG(figs["paired"], "Paired design: within a repetition all policies see the identical seeded request stream, in a randomized order.", 1.0)
    out += H(3, "Pairing and randomization")
    out.append(P(
        "The seed for repetition <i>k</i> is 42&nbsp;+&nbsp;<i>k</i>, and it fixes arrival times, prompt lengths and output lengths. Every policy replays "
        "the same stream within a repetition, so the paired difference removes between-stream variability, which is large for tail latency. The "
        "order of the four policies within each repetition is a seeded random permutation, so slow effects (laptop GPU thermal state, background load) "
        "do not align with any policy."))
    out += H(3, "Choice of baseline")
    out.append(P(
        "The hypothesis compares the adaptive policy against the <i>best fixed</i> configuration for each load pattern. The best fixed configuration is "
        "chosen among fixed-fast, fixed-bulk and static-lb on the same metric being tested (p99), among configurations with zero request errors. "
        "Choosing the baseline after seeing the data on the tested metric makes the test conservative: it gives the baseline every chance."))
    out += H(3, "Statistical protocol")
    out.append(BUL([
        "Headline values are means with 95% t-based confidence intervals over the eight repetitions.",
        f"The adaptive-versus-best-fixed comparison is <b>paired</b> per repetition. Two tests are computed: the paired Student <i>t</i>-test {cite('student')} and the "
        f"Wilcoxon signed-rank test {cite('wilcoxon')}. A difference counts as significant only if <b>both</b> agree, and the Wilcoxon <i>p</i> "
        f"is Holm-adjusted {cite('holm')} across the three load patterns.",
        f"Effect size is Cohen&rsquo;s paired <i>d<sub>z</sub></i> {cite('cohen')}; a bootstrap confidence interval of the paired difference is also computed "
        f"{cite('efron')}.",
        "A <b>throughput non-inferiority</b> guard requires the adaptive policy to reach at least 98% of the best fixed throughput; otherwise a latency win is "
        "not counted, since it could be bought by shedding capacity.",
        "The verdict for each pattern is generated by code and has an outcome for &lsquo;adaptive better&rsquo;, &lsquo;no significant difference&rsquo; and "
        "&lsquo;adaptive worse&rsquo;.",
    ]))
    out += H(3, "Why eight repetitions")
    out.append(P(
        "The exact Wilcoxon signed-rank test with <i>n</i> pairs cannot produce a two-sided <i>p</i> smaller than 2/2<super><i>n</i></super>, however large the "
        "effect. With Holm correction over three patterns, the smallest attainable adjusted <i>p</i> is three times that. Requiring the smallest "
        "attainable adjusted <i>p</i> to fall below 0.05 fixes the minimum sample:"))
    out.append(TABLE([
        ["Pairs n", "Smallest two-sided p", "Smallest Holm-adjusted p (3 patterns)", "Can reach 0.05?"],
        *[[str(n), f"{2 / 2 ** n:.4f}", f"{min(1, 3 * 2 / 2 ** n):.4f}", "yes" if 3 * 2 / 2 ** n < 0.05 else "no"] for n in (4, 5, 6, 7, 8)],
    ], [0.15, 0.25, 0.4, 0.2], "Attainable significance of the Wilcoxon test by number of pairs (computed).", font=8.4))
    out.append(P(
        "Seven pairs is therefore the theoretical minimum, and only just: the adjusted <i>p</i> would be 0.047, so a single tie or a single discordant pair would "
        "remove significance. Eight pairs (adjusted 0.023) was chosen for headroom. Studies with fewer than seven pairs (for example "
        "the five-repetition contention experiment) are labelled as unable to reach conventional significance, and effect sizes and intervals are "
        "reported instead of <i>p</i>-values."))
    out += H(3, "Decision rule for H<sub>1</sub>")
    out.append(P(
        "H<sub>1</sub> is confirmed for a load pattern only if adaptive p99 is lower than the best fixed p99, both tests agree at Holm-adjusted 0.05, and the "
        "throughput guard holds. It is <i>rejected</i> if the difference is significant in the opposite direction. Otherwise it is undetermined. The "
        "controller thresholds (Section 6.4) were fixed before the campaign began; no threshold was adjusted after seeing campaign results."))

    # ---------------------------------------------------------------- 7.6
    out += H(2, "Mechanism study: co-residency contention")
    out.append(P(
        "The campaign result was negative in a direction that required explanation. A candidate mechanism, that two servers on one GPU interfere, "
        "was formulated as a hypothesis and tested with a separate controlled experiment rather than asserted. It is <b>explanatory, not confirmatory</b>: "
        "it was designed after seeing campaign data and is labelled that way."))
    out.append(TABLE([
        ["Condition", "What runs", "What it isolates"],
        ["single", "One replica loaded; all traffic to it.", "Reference."],
        ["split-idle", "Both replicas loaded; all traffic to one, the other idle.", "Cost of a resident but idle second server (memory split, batch caps)."],
        ["split-active", "Both replicas loaded; traffic split across both at the same total rate.", "Cost of a second <i>active</i> server: GPU time-slicing and smaller per-server batches."],
    ], [0.15, 0.45, 0.4], "Contention study conditions.", font=8))
    out.append(P(
        "Total offered load is identical in all conditions and is expressed as a fraction &rho; of capacity (&rho;&nbsp;=&nbsp;0.3 and 0.6). The comparison "
        "split-idle against single isolates residency; split-active against split-idle isolates activity. Five repetitions per cell were used, so only "
        "effect sizes and paired ratios are interpreted."))
    out += FIG(figs["timeslice"], "Why two active servers on one GPU can hurt: time-slicing between processes and smaller per-server batches.", 0.9)

    # ---------------------------------------------------------------- 7.7
    out += H(2, "MPI router methodology")
    out += H(3, "Microbenchmarks")
    out.append(P(
        "Two microbenchmarks characterize the messaging layer independent of any model. <b>Ping-pong</b> measures one-way latency by halving the round trip "
        "for message sizes from 64&nbsp;B to 256&nbsp;KB. <b>Dispatch throughput</b> measures jobs per second through rank 0 with 1, 2, 4, 8 and 15 workers. "
        f"This machine has {F['cores']} logical cores, so the largest configurations run oversubscribed and their numbers are labelled as such. "
        f"Medians over repeated trials are reported."))
    out += H(3, "Clock argument and per-hop timestamps")
    out.append(P(
        "The router timestamps every hop with the operating system&rsquo;s monotonic clock. Because all ranks run on one host, that clock is shared, and "
        "differences between ranks are meaningful without clock-offset estimation. This argument would not hold across machines; a multi-node extension "
        "would need a clock-synchronization step, which is why cross-node claims are not made."))
    out += H(3, "Scaling and router comparison")
    out.append(P(
        "The scaling experiment measures throughput with 1 and 2 replicas behind (a) the HTTP least-outstanding router and (b) the MPI router, with "
        "paired repetitions. Because replicas share one GPU, this experiment measures <i>coordination and contention overhead</i>, not the scaling that "
        "independent GPUs would give. Efficiency is throughput divided by the replica count times single-replica throughput. The in-situ MPI "
        "overhead is measured from the recorded hop timestamps."))

    # ---------------------------------------------------------------- 7.8
    out += H(2, "Configuration sweep and drain handling")
    out.append(P(
        "The sweep crosses quantization (fp16, int8, int4) with batch caps (1, 8, 32, 64) at one steady load, with three repetitions per cell, and measures "
        "latency, throughput, memory, power and cold-start time. Two protocol decisions matter. First, configurations with a low batch cap cannot sustain "
        "the offered load; if such a run were allowed to drain indefinitely it would stall the sweep, so requests still unfinished 60&nbsp;s after the last "
        "arrival are cancelled and counted as failures, and each cell reports its completed fraction. Second, latency percentiles of a cell that cannot "
        "finish its work are not comparable with those of a cell that can, so the results table shows the completed fraction beside latency and the "
        "discussion does not rank incomplete cells by latency."))

    # ---------------------------------------------------------------- 7.9
    out += H(2, "Threats to validity")
    out.append(TABLE([
        ["Threat", "Type", "Mitigation and residual risk"],
        ["Single GPU shared by all replicas", "External / construct", "Deliberately studied (contention experiment). Conclusions about routing do <b>not</b> extend to one-GPU-per-replica deployments; that is the main scope limit."],
        ["Small model (1.5B parameters)", "External", "Constrained by 8&nbsp;GB. Kernel shapes from a 7B model are included in the kernel study only."],
        ["Synthetic prompts and length distributions", "External", "Lognormal and bimodal lengths approximate real traffic; an empirical-trace mode exists but a public trace was not used. Prefix-sharing patterns of real traffic are not represented."],
        ["Output length pinned with ignore_eos", "Construct", "Makes work identical across policies; removes variability in length that real requests have."],
        ["Quantized quality not evaluated", "Construct", "Speed and energy are reported, not accuracy. No claim is made that int4 is quality-equivalent for a task."],
        ["Laptop GPU thermal and power state", "Internal", "Randomized policy order per repetition; GPU clocks were not locked, so residual drift is possible."],
        ["Run length (75 s) and n = 8", "Statistical", "Sufficient for large effects; small effects (a few percent) would not be detected. Tail estimates are noisy per run."],
        ["Windows host with WSL2", "Internal", "GPU scheduling under WSL2 may differ from native Linux; not measured."],
        ["Post-hoc mechanism study", "Statistical", "Labelled explanatory; five repetitions; effect sizes are large so the qualitative conclusion is robust, the exact ratio is not."],
        ["Library kernel comparison at kernel level only", "Construct", "In-situ kernel time excludes launch gaps; the server-level check is the capacity comparison."],
        ["Unverified counters and untested cloud phase", "Coverage", "Reported as open items rather than filled with estimates."],
    ], [0.27, 0.15, 0.58], "Threats to validity.", font=7.6))

    # ---------------------------------------------------------------- 7.10
    out += H(2, "Audit trail: errors found by running the protocol")
    out.append(P(
        "The value of a protocol is measured partly by what it caught. The following defects were found during the work by checks that the protocol "
        "required; each changed a result or a claim, and each is fixed and covered by a test or by a generator."))
    out.append(TABLE([
        ["Defect", "How it was found", "Effect and fix"],
        ["v4 kernel wrong (relative error 2-3)", "CPU-reference correctness gate", "Speed numbers for v4 would have been meaningless; fixed by accumulating activation sums per chunk."],
        ["Throughput equalled offered load", "Sanity check against known saturating case", "Metric redefined over the busy span including drain."],
        ["Guessed request rates saturated the GPU", "First smoke run", "Capacity calibration introduced."],
        ["Negative TTFT on Windows", "Range check on recorded latencies", "Client loops until scheduled time."],
        ["<font name='Mono'>quantization=awq</font> ran a different kernel than assumed", "In-situ profiling", "Serving pinned to awq_marlin / gptq_marlin; both paths measured and reported."],
        ["Ceiling quoted as 4&times;", "Recomputing bytes from the layout", "Ceiling computed per shape (about 3.76&times;); all text and figures regenerated."],
        ["Bank conflict stated as 32-way", "Re-deriving the access pattern for half-precision activations", "Corrected to 16-way in code comments, docs and generated text."],
        ["Early campaign numbers read as a 3&times; loss", "Interim looks", "Not reported as a result; tested by the contention experiment first."],
        ["Minimum repetitions stated as eight", "Tabulating attainable p-values while writing this report", "Seven is the theoretical minimum (Holm-adjusted 0.047, no margin); eight was kept for headroom and the repository documents were corrected."],
        ["Kernel benchmark rotated only 192 MB of weights", "A rotation-size sweep, prompted by a kernel exceeding 100% of theoretical bandwidth", "A non-LRU L2 kept part of the working set and made a 23 MB kernel about 6% too fast. Default raised to 768 MB (24 x L2); every kernel number regenerated. Earlier bandwidth percentages were inflated by that amount."],
        ["In-situ Marlin timing about twice too slow", "Cross-check against end-to-end tokens per second, then re-profiling in both engine modes", "The claim that the custom kernel beat Marlin on all four shapes was wrong and is retracted: Marlin is at the bandwidth limit at batch 1. The invalid file was retired; all three paths were re-measured together. The cause of the earlier value could not be established."],
        ["Test that ran a Linux binary on Windows", "Test suite on the Windows host", "Skipped off Linux."],
    ], [0.3, 0.27, 0.43], "Defects found by the protocol and their resolution.", font=7.6))
    return out
