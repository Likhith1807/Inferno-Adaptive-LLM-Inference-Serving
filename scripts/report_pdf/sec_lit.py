"""Chapters 4 and 5: literature review, and previous work / starting point of the project."""
from __future__ import annotations

from . import data
from .refs import cite
from .style import CALLOUT, EQ, FIG, H, P, TABLE


class _Out(list):
    """list whose append() also accepts a list of flowables (H, TABLE, CALLOUT and EQ return lists)."""

    def append(self, x):
        self.extend(x) if isinstance(x, list) else super().append(x)


def chapter_literature(figs):
    out = _Out(H(1, "Literature review"))
    out.append(P(
        "The review is organized by the layers this study touches: the hardware behaviour of LLM inference, serving-system "
        "scheduling, prefill/decode interference, multi-instance routing, quantization and its kernels, GPU sharing, message passing in "
        "high-performance computing, and benchmarking and statistical practice. Each section ends with what it implies for the design "
        "of this study. A synthesis and gap analysis closes the chapter."))

    # ------------------------------------------------------------------ 4.1
    out += H(2, "Hardware behaviour of transformer inference")
    out += H(3, "Prefill, decode and the KV cache")
    out.append(P(
        f"Autoregressive generation with the Transformer architecture {cite('vaswani')} caches the keys and values of previous tokens so "
        f"each new token attends over stored state instead of recomputing it. A request then divides into a prompt-processing phase and a "
        f"token-generation phase. Characterizing production-like workloads, the Splitwise study reports that the prompt phase is "
        f"compute-intensive while the token-generation phase is memory-intensive, with distinct latency, throughput, memory and power "
        f"profiles {cite('splitwise')}. This asymmetry is the root of most serving-system design and of the heterogeneous-tier design "
        f"used later in this report."))
    out += H(3, "The roofline view")
    out.append(P(
        f"The roofline model {cite('roofline')} bounds attainable performance by the minimum of a compute roof and a memory roof, and "
        f"identifies a kernel's regime by its <i>arithmetic intensity</i>: floating-point operations per byte moved from DRAM."))
    out += EQ(r"P_{\mathrm{attainable}}=\min\left(P_{\mathrm{peak}},\ B_{\mathrm{DRAM}}\cdot I\right),\qquad I=\frac{\mathrm{FLOPs}}{\mathrm{bytes\ moved}}")
    out.append(P(
        "For a matrix-vector product with an <i>N&times;K</i> weight matrix in fp16, FLOPs = 2<i>NK</i> and bytes &asymp; 2<i>NK</i>, so "
        "<i>I</i> &asymp; 1 FLOP/byte, far to the left of any modern GPU's ridge point: decode at batch size 1 is memory-bound "
        f"(figure below). Pope et al. analyse this trade-off when partitioning large Transformer inference across accelerators and show "
        f"how batch size and memory bandwidth govern cost and latency {cite('pope')}. FlashAttention {cite('flash', 'flash2')} is the "
        "canonical example of the same principle applied to attention: reduce traffic between GPU high-bandwidth memory and on-chip "
        "SRAM rather than reduce arithmetic."))
    out += FIG(figs["roofline"], "Roofline view of prefill and decode: decode at small batch sits on the memory roof, prefill near the compute roof.", 0.9)
    out.append(CALLOUT("Implication for this study",
                       "The kernel study (Chapter 9) therefore uses achieved DRAM bandwidth, and speedup against the <i>byte-ratio ceiling</i>, as "
                       "the figures of merit, and the serving study treats batching as the mechanism that moves decode toward the compute roof."))

    # ------------------------------------------------------------------ 4.2
    out += H(2, "Batching and scheduling in LLM serving systems")
    out.append(P(
        f"Early inference servers batched at request granularity, so a batch waited for its longest sequence. <b>Orca</b> {cite('orca')} "
        f"introduced <i>iteration-level scheduling</i>: the scheduler invokes the execution engine for a single model iteration at a "
        f"time, so finished requests leave and new ones join between iterations; it pairs this with <i>selective batching</i>, "
        f"batching only the operations that allow it. On GPT-3 175B the authors report a large throughput improvement over NVIDIA "
        f"FasterTransformer at comparable latency. This <i>continuous batching</i> is now standard."))
    out.append(P(
        f"<b>vLLM</b> {cite('vllm')} attacks the memory side. KV-cache memory per request is large and grows and shrinks dynamically, so "
        f"contiguous allocation wastes memory through fragmentation and duplication and limits batch size. PagedAttention stores the KV cache "
        f"in fixed-size blocks mapped to non-contiguous physical memory, borrowing virtual-memory paging, and reports 2-4&times; higher "
        f"throughput than prior systems at similar latency. The engine used throughout this study is vLLM {cite('vllmdocs')}."))
    out.append(P(
        f"<b>Sarathi-Serve</b> {cite('sarathi')} addresses the throughput-latency tension inside a single replica. Long prompts, if "
        f"prefilled in one iteration, stall the decode of every other request in the batch. <i>Chunked prefill</i> splits a prompt into "
        f"near-equal chunks and builds <i>stall-free</i> schedules that add new work without pausing ongoing decodes. The bound on tokens "
        f"per iteration (in vLLM, <font name='Mono'>max_num_batched_tokens</font>) becomes a control knob trading TTFT of long prompts "
        f"against decode smoothness for everyone else. This is the knob that differentiates the two replica tiers in this study."))
    out.append(P(
        f"Orthogonal techniques reduce work per token rather than schedule it: speculative decoding {cite('spec')} lets a cheap draft model "
        f"propose tokens that the target model verifies in parallel; SGLang {cite('sglang')} adds a runtime for structured multi-call "
        f"programs with prefix-cache reuse. Neither is used here; both are complementary to the routing question."))

    # ------------------------------------------------------------------ 4.3
    out += H(2, "Prefill/decode interference and disaggregation")
    out.append(P(
        f"<b>DistServe</b> {cite('distserve')} observes that colocating prefill and decode across all requests causes strong "
        f"prefill-decode interference and couples the resource allocation and parallelism plans of both phases. It assigns the phases to "
        f"different GPUs, co-optimizes each phase's parallelism for TTFT and TPOT targets, and optimizes <i>goodput</i>, the rate of requests "
        f"that meet the latency objectives; the authors report serving several times more requests, or meeting much tighter latency "
        f"objectives, than the state of the art. <b>Splitwise</b> {cite('splitwise')} reaches a similar architecture from a "
        f"characterization and power/cost perspective, splitting the phases across machines so each can run on hardware suited to it."))
    out.append(P(
        f"Two observations matter here. First, both systems put the specialized phases on <i>separate devices</i>; their premise is that "
        f"co-locating heterogeneous work on one device creates interference. Second, the goodput metric they popularized is adopted in this "
        f"study. The heterogeneous-tier design used here (a small-chunk decode-oriented replica and a large-chunk throughput-oriented replica) "
        f"is a lightweight relative of phase specialization; but because the two tiers share one GPU, the interference premise applies to "
        f"the <i>replicas themselves</i>, which the results confirm."))

    # ------------------------------------------------------------------ 4.4
    out += H(2, "Multi-instance scheduling, routing and adaptive serving")
    out.append(P(
        f"General model-serving systems introduced several ideas this study revisits. <b>Clipper</b> {cite('clipper')} interposes between "
        f"applications and models and uses caching, adaptive batching and bandit-based model selection to control latency, throughput and "
        f"accuracy. <b>INFaaS</b> {cite('infaas')} lets users state performance and accuracy goals and selects among model variants and "
        f"hardware to meet them, autoscaling as load changes. <b>AlpaServe</b> {cite('alpaserve')} shows that model parallelism can be used "
        f"for statistical multiplexing across devices under bursty load, exposing a trade-off between parallelism overhead and multiplexing "
        f"gains."))
    out.append(P(
        f"For LLMs specifically, <b>Llumnix</b> {cite('llumnix')} reschedules requests across model instances at run time, using live "
        f"migration of requests and their in-memory state, to improve load balancing and isolation and to differentiate priorities; it reports "
        f"order-of-magnitude tail-latency improvements against strong baselines. Its adaptivity acts by <i>moving work between instances</i>; "
        f"this study's adaptivity acts by <i>choosing the instance for new work</i>, a cheaper and weaker mechanism appropriate to what a "
        f"stock vLLM server exposes."))
    out.append(P(
        "A pattern across these systems is that instances are assumed to have independent accelerators. None of the works reviewed here "
        "isolates what happens when independent inference servers share one device, which is the question Section&nbsp;9.4 answers. This is a "
        "statement about the set of papers reviewed, not a claim that no such study exists."))

    # ------------------------------------------------------------------ 4.5
    out += H(2, "Quantization and quantized kernels")
    out += H(3, "Algorithms")
    out.append(P(
        f"Weight-only post-training quantization compresses weights to 4 or 8 bits while activations stay in higher precision. "
        f"<b>LLM.int8()</b> {cite('llmint8')} handles outlier features with mixed-precision decomposition; <b>SmoothQuant</b> "
        f"{cite('smoothquant')} migrates activation outliers into weights to enable 8-bit weight-and-activation quantization; "
        f"<b>GPTQ</b> {cite('gptq')} quantizes one-shot using approximate second-order information, reaching 3-4 bits per weight with small "
        f"accuracy loss; <b>AWQ</b> {cite('awq')} protects salient weight channels by observing activation statistics, avoiding backpropagation "
        f"and reconstruction, and received the MLSys 2024 best paper award. Group-wise asymmetric quantization, in which each group of "
        f"consecutive weights (typically 128) shares a scale and a zero point, is the format targeted by the kernel in this study."))
    out += H(3, "Kernels")
    out.append(P(
        f"A quantized model is only faster if its kernels avoid materializing dequantized weights in DRAM. <b>MARLIN</b> {cite('marlin')} "
        f"is a family of mixed-precision linear kernels for 4-bit weights; the authors show that batch sizes up to 16-32 can be supported "
        f"with close to the maximum (nominally 4&times;) quantization speedup, with gradually decreasing benefit up to 64-128, using "
        f"asynchronous memory access, task scheduling and pipelining, and report up to 2.8&times; end-to-end speedup inside vLLM. The "
        f"'nominal 4&times;' refers to the weight payload; this report shows that with group-128 fp16 scales and zero points the "
        f"byte-ratio ceiling for a matrix-vector product is closer to 3.76&times; (Section&nbsp;7.4). vLLM ships several quantized paths "
        f"{cite('vllmdocs')}: the original AWQ GEMM kernel (selected by <font name='Mono'>quantization=awq</font>) and Marlin-based paths "
        f"(<font name='Mono'>awq_marlin</font>, <font name='Mono'>gptq_marlin</font>). Marlin is designed for parallel (batched) clients; "
        f"this study measures them in situ, end to end and at saturated serving: on this GPU Marlin is at the bandwidth limit at batch 1 and clearly faster "
        f"with concurrent sequences, while the classic AWQ kernel is slower in both regimes."))
    out.append(CALLOUT("Implication for this study",
                       "A kernel comparison is regime-dependent. Any claim about a quantized kernel must state the batch size, the layer shapes, and "
                       "whether the measurement is isolated or in situ. This report does all three.", ))

    # ------------------------------------------------------------------ 4.6
    out += H(2, "Sharing a GPU between processes")
    out.append(P(
        f"NVIDIA's documentation describes the options for sharing a device among processes {cite('mps')}. By default, work from different "
        f"processes is time-sliced on the GPU; the Multi-Process Service (MPS) can allow kernels from multiple processes to run "
        f"concurrently, and Multi-Instance GPU (MIG) partitions supported data-centre GPUs into isolated instances. The consumer laptop "
        f"GPU and WSL2 environment used here provide default time-slicing; MPS and MIG were not evaluated (MIG is unavailable on this class of "
        f"GPU, and MPS support under WSL2 was not explored). Time-slicing implies that two active servers each see a fraction of GPU "
        f"time, pay context-switch overhead, and, because each holds only part of the request stream, run smaller batches than one merged "
        f"server would. Section&nbsp;9.4 measures this effect directly."))

    # ------------------------------------------------------------------ 4.7
    out += H(2, "Message passing in high-performance computing")
    out.append(P(
        f"MPI {cite('mpi')} is the standard message-passing interface of HPC; Open MPI {cite('openmpi')} is a widely used implementation. "
        f"Performance of message passing is commonly reasoned about with latency-bandwidth (alpha-beta) models and, more richly, with LogP "
        f"{cite('logp')}, whose parameters capture latency, overhead, gap and processor count. MPI implementations typically switch between "
        f"an <i>eager</i> protocol for small messages and a <i>rendezvous</i> protocol for large ones, producing a latency step at a size "
        f"threshold; the step observed in Chapter&nbsp;9 between 32&nbsp;KB and 256&nbsp;KB is consistent with such a switch, though the exact "
        f"threshold was not determined."))
    out.append(P(
        "MPI is rarely used for the control plane of LLM serving: production systems use HTTP or gRPC front ends and collective libraries "
        "for tensor-parallel communication. It is used here because the HPC-oriented objective of the project calls for an MPI program, "
        "and because that choice is informative: an honest evaluation needs an HTTP baseline, which this study includes."))

    # ------------------------------------------------------------------ 4.8
    out += H(2, "Benchmarking methodology and statistics")
    out += H(3, "Workload generation and measurement")
    out.append(P(
        f"Schroeder, Wierman and Harchol-Balter show that <i>open</i> workload generators (arrivals independent of completions) and "
        f"<i>closed</i> ones (arrivals triggered by completions) can lead to very different conclusions about scheduling policies, and "
        f"that the choice must match the system being modelled {cite('openclosed')}. A closed-loop client silently reduces offered load "
        f"as latency rises, hiding queueing. Dean and Barroso emphasise that tail latency, not the mean, governs user-visible "
        f"quality in large services {cite('tail')}. Little's law {cite('little')} (L = &lambda;W) connects arrival rate, latency and "
        f"in-flight concurrency and is used here to reason about which batch caps are ever active."))
    out += H(3, "Rigorous performance evaluation")
    out.append(P(
        f"A line of work argues that performance claims must be statistically defensible. Georges, Buytaert and Eeckhout show how "
        f"non-determinism makes single-run comparisons misleading and advocate confidence intervals and appropriate tests "
        f"{cite('georges')}. Hoefler and Belli survey practice in parallel computing, find that it is often unclear whether reported "
        f"improvements are deterministic or observed by chance, and give twelve guidelines for experimental design and reporting "
        f"{cite('hoefler')}. Mytkowicz et al. demonstrate that innocuous setup choices can bias results substantially "
        f"{cite('mytkowicz')}, and Kalibera and Jones treat how many repetitions and at what levels are needed "
        f"{cite('kalibera')}. MLPerf Inference {cite('mlperf')} standardizes inference benchmarking, including latency-bound "
        f"scenarios. These works motivate the paired designs, repetition counts, spread reporting and correctness gates used here."))
    out += H(3, "Statistical tools")
    out.append(P(
        f"The comparison uses the paired Student t-test {cite('student')} and the Wilcoxon signed-rank test {cite('wilcoxon')}, "
        f"which is distribution-free and exact for small samples; Holm's step-down procedure {cite('holm')} to control the family-wise error "
        f"rate across load patterns; Cohen's paired effect size <i>d<sub>z</sub></i> {cite('cohen')}; and the bootstrap {cite('efron')} for "
        f"confidence intervals of paired differences. A property of the exact Wilcoxon test drives the study design: with <i>n</i> pairs "
        f"all in one direction, the smallest attainable two-sided <i>p</i> is 2/2<super><i>n</i></super>, which is 0.0625 for <i>n</i>&nbsp;=&nbsp;5 and "
        f"0.0078 for <i>n</i>&nbsp;=&nbsp;8."))

    # ------------------------------------------------------------------ 4.9
    out += H(2, "Synthesis and gap analysis")
    out.append(TABLE([
        ["Dimension", "State of the literature", "Gap addressed by this study"],
        ["Kernel comparison", "Marlin reports near-maximal speedup for batches 16-32 and beyond; AWQ kernels widely used.",
         "Batch-1 and small-batch comparison on a small GPU with a byte-exact ceiling and a measured bandwidth reference; in-situ and end-to-end (vLLM integration) results; regime dependence made explicit; correctness gate and spread reported."],
        ["Adaptive routing", "Llumnix, Clipper, INFaaS, AlpaServe adapt via migration, model selection or placement; instances assumed on separate accelerators.",
         "Pre-specified, paired test of routing over pre-loaded heterogeneous replicas against the strongest fixed baselines in the same pool; negative result with mechanism."],
        ["GPU co-residency", "Interference studied inside a batch (prefill vs decode); disaggregation puts phases on separate devices.",
         "Controlled measurement of active vs idle second replica at fixed total load."],
        ["MPI in serving", "MPI standard in HPC; not standard for LLM control planes.", "Per-hop instrumented MPI router vs HTTP baseline, isolated and in situ."],
        ["Methodology", "Strong general guidance (open loop, tails, paired stats) but often not applied in LLM-serving evaluations.",
         "End-to-end application: capacity-relative load, seeded pairing, randomized order, drain-aware metrics, generated verdicts, error audit trail."],
    ], [0.17, 0.41, 0.42], "Gap analysis.", font=7.6))
    return out


def chapter_previous_work():
    F = data.facts()
    out = _Out(H(1, "Previous work and starting point"))
    out += H(2, "Closest prior systems and how this work relates")
    out.append(P(
        "The table compares the systems most closely related to the questions asked here. The purpose is positioning, not ranking: the "
        "systems solve different problems at different scales, on data-centre hardware, and none was re-run in this study."))
    out.append(TABLE([
        ["System", "What it decides", "Where adaptivity acts", "Accelerator assumption", "Relation to this study"],
        ["Orca " + cite("orca"), "Which requests run each iteration", "Iteration-level scheduling", "One model instance (multi-GPU for large models)", "Continuous batching is what vLLM already provides here; not re-studied."],
        ["vLLM " + cite("vllm"), "KV-cache block allocation, batching", "Memory management, scheduler", "Single or multi-GPU per engine", "The serving engine used throughout; its knobs define the replica tiers."],
        ["Sarathi-Serve " + cite("sarathi"), "How prefills are chunked", "Per-iteration token budget", "Per replica", "The prefill chunk size distinguishes tier 0 (512) from tier 1 (4096)."],
        ["DistServe " + cite("distserve"), "Which GPUs run prefill vs decode", "Static placement optimized for goodput", "Separate GPUs per phase", "Goodput metric adopted; premise that co-location interferes is tested for replicas."],
        ["Splitwise " + cite("splitwise"), "Which machines run which phase", "Phase splitting with cluster provisioning", "Separate machines per phase", "Characterization motivates decode being memory-bound."],
        ["Llumnix " + cite("llumnix"), "Where running requests live", "Live migration across instances", "Independent instances", "Adaptivity by migration; here only by placement of new work."],
        ["Clipper / INFaaS " + cite("clipper", "infaas"), "Which model variant serves a query", "Adaptive batching, bandit / SLO-driven selection", "Cluster of serving nodes", "Ancestors of routing among configurations; not LLM-specific."],
        ["AlpaServe " + cite("alpaserve"), "Model placement and parallelism", "Statistical multiplexing under bursts", "Cluster of GPUs", "Motivates bursty load patterns; multiplexing needs separate devices."],
        ["MARLIN " + cite("marlin"), "How 4-bit GEMM executes", "Kernel design for batched clients", "Ampere+ GPUs, batch &ge; 1", "Library comparator for the custom batch-1 kernel."],
    ], [0.14, 0.17, 0.2, 0.19, 0.30], "Related systems and relation to this work.", font=7.3))
    out += H(2, "Starting point of this project")
    out.append(P(
        "The project began from a research brief with a ten-week roadmap and four hypotheses (H1-H4), and a repository scaffold. The "
        "scaffold provided the directory structure, configuration files, a workload generator, an adaptive-controller rule set, a "
        "statistics module and unit tests, but the serving harness, the experiment runners, the MPI router and the CUDA kernel were "
        "skeletons that raised <font name='Mono'>NotImplementedError</font> or contained only a naive one-thread-per-element dequantization "
        "kernel. The brief specified an 8-billion-parameter model, vLLM or TGI serving, batch sizes 1 to 64, int8 and int4 quantization, an "
        "OpenMPI multi-replica router, a CUDA dequantization kernel, and cloud deployment."))
    out += H(2, "What was retained and what changed")
    out.append(TABLE([
        ["Original plan", "What happened", "Why"],
        ["Llama-3-8B fp16 as the study model", "Qwen2.5-1.5B-Instruct (fp16, GPTQ-Int8, AWQ-Int4)", "An 8B fp16 model needs about 16 GB; the GPU has 8 GB. The 1.5B family has official releases at all three precisions."],
        ["Adaptive controller switches batch size and quantization live", "Adaptivity implemented as routing across pre-loaded heterogeneous replicas", "vLLM cannot change batch cap or precision on a running server; a restart costs about " + f"{F['cold']:.0f}" + " s (measured)."],
        ["Request rates chosen a priori", "Rates derived from measured saturation capacity (utilization fractions)", "The first smoke run at a guessed rate saturated the GPU, so measured tail latency would have reflected queue growth rather than policy."],
        ["Throughput = tokens / arrival window", "Throughput over the busy span including drain", "The first metric made every policy report the offered load."],
        ["<font name='Mono'>quantization=awq</font>", "<font name='Mono'>awq_marlin</font> / <font name='Mono'>gptq_marlin</font>", "Profiling showed 'awq' runs the older kernel; server-level throughput differs 2.1&times;."],
        ["Kernel ceiling stated as 4&times;", "Ceiling computed from bytes: about 3.76&times;", "Group-128 fp16 scales and zero points add 0.03 bytes per weight."],
        ["10-week timeline", "Phases without a timeline", "Requested by the project owner; work ordered by dependency."],
        ["Ten-week 'adaptive wins' framing", "Pre-specified hypothesis test that could and did fail", "The methodology requires a falsifiable claim; the result was negative."],
    ], [0.27, 0.33, 0.40], "Deviations from the initial plan and their reasons.", font=7.5))
    out += H(2, "Previous work in this repository")
    out.append(P(
        "Reusable pieces that predate this report and were kept, with modification, are the workload generator (Poisson arrivals, "
        "bursty schedule, lognormal lengths; extended with a bimodal mixture and an empirical-trace mode), the confidence-interval and "
        "paired-test module (extended with Holm correction, effect sizes and a bootstrap), the rule-based adaptive controller (replaced by a "
        "hysteresis-based tier policy that operates on a pool state), and the unit-test scaffolding. Everything else described in this "
        "report was implemented during the study."))
    return out
