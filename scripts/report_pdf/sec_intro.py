"""Cover, abstract, contents, and chapters 1-3 (introduction, problem statement, objectives)."""
from __future__ import annotations

from reportlab.lib.units import mm
from reportlab.platypus import NextPageTemplate, PageBreak, Paragraph, Spacer, Table, TableStyle

from . import data
from .refs import cite
from .style import (ACCENT, EQ, H, INK, INK2, MUTED, NUM, BUL, P, S, TABLE, TEXT_W, UH, colors, make_toc)


def cover():
    hw = data.hardware()
    gpu = (hw.get("gpus") or [{}])[0]
    from reportlab.lib.styles import ParagraphStyle

    title = ParagraphStyle("t", fontName="Sans-B", fontSize=30, leading=35, textColor=INK, spaceAfter=10)
    sub = ParagraphStyle("s", fontName="Sans", fontSize=13.5, leading=19, textColor=INK2, spaceAfter=22)
    lab = ParagraphStyle("l", fontName="Sans-B", fontSize=8, leading=11, textColor=MUTED)
    val = ParagraphStyle("v", fontName="Sans", fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=7)
    out = [Spacer(1, 52 * mm), Paragraph("TECHNICAL REPORT", ParagraphStyle("k", fontName="Sans-B", fontSize=9, textColor=ACCENT, spaceAfter=8)),
           Paragraph("Inferno: Adaptive LLM Inference Serving, Measured End to End", title),
           Paragraph("A rigorous experimental study spanning a hand-written CUDA kernel, an MPI request router and a "
                     "load-aware routing controller on a single 8&nbsp;GB GPU", sub)]
    rows = [[Paragraph("HARDWARE", lab), Paragraph(f"{gpu.get('name', 'GPU')} (8 GB, sm_89, 24 SMs), {gpu.get('driver_version', '')} driver; "
                                                   f"Windows 11 with WSL2 (Ubuntu)", val)],
            [Paragraph("SOFTWARE", lab), Paragraph(f"vLLM {hw.get('packages', {}).get('vllm', '')}, PyTorch {hw.get('packages', {}).get('torch', '')} "
                                                   "(CUDA 12.8 build), CUDA toolkit 12.4, Open MPI 5.0.10, Python 3.12", val)],
            [Paragraph("MODEL", lab), Paragraph("Qwen2.5-1.5B-Instruct in fp16, GPTQ-Int8 and AWQ-Int4 (official releases)", val)],
            [Paragraph("SCOPE", lab), Paragraph("Problem statement, objectives, literature review, prior work, complete methodology, "
                                                "results and discussion", val)],
            [Paragraph("STATUS", lab), Paragraph("All numbers in this report are generated from raw measurement files by scripts in the repository. "
                                                 "Negative results are reported as such.", val)],
            [Paragraph("AUTHOR", lab), Paragraph("Project owner. Report prepared September 2026.", val)]]
    t = Table(rows, colWidths=[28 * mm, TEXT_W - 28 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5d8de")),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    out += [t, NextPageTemplate("body"), PageBreak()]
    return out


def _abstract_layer():
    from .sec_results import layer_facts
    f = layer_facts()
    if not f:
        return ""
    txt = (f" (6) Timed against Marlin, classic AWQ and fp16 on every decode layer at batch 1 to 64, the kernel clearly wins only layers worth "
           f"{f['wins_share']:.0f}% of decode GEMM time, at batch 1, and is up to {f['max_ratio']:.1f}&times; slower above that, so by a pre-stated rule "
           "Marlin stays the serving default.")
    if f["mf"]:
        txt += (f" The same benchmark&rsquo;s reference check found Marlin returning wrong results (relative error up to {f['mf']['worst_rel_err_faulty']:.2f}) "
                f"at batch {', '.join(map(str, f['mf']['faulty_batches']))} on this machine, a finding not yet reproduced elsewhere.")
    return txt


def abstract():
    F = data.facts()
    k = F["kern"]
    h1 = F["h1"]
    lo, hi = min(r["pct_change"] for r in h1), max(r["pct_change"] for r in h1)
    out = UH("Abstract", page_break=False)
    out.append(P(
        "Serving large language models (LLMs) efficiently is a systems problem that spans GPU kernels, batching and scheduling, "
        "quantization, multi-replica routing and the statistics used to judge any of it. This report documents <b>Inferno</b>, an "
        "experimental testbed that studies all of these layers together on one 8&nbsp;GB laptop GPU, and asks a single question: "
        "<i>can an adaptive routing policy over pre-loaded, heterogeneous replicas beat every fixed configuration under realistic, bursty "
        "traffic, and where do the gains or losses come from?</i>"))
    e = F["e2e"]
    r1, r4 = F["e2e_ratio_b8"]["1"]["mean"], F["e2e_ratio_b8"]["4"]["mean"]
    out.append(P(
        f"The study makes six measurements. (1) A hand-written fused int4 dequantize-and-matrix-vector CUDA kernel, verified against a "
        f"double-precision CPU reference before any timing, reaches {k['pct']:.0f}% of theoretical DRAM bandwidth "
        f"({100 * k['gbps'] / F['stream_gbps']:.0f}% of the measured streaming-read bandwidth) on the best layer and "
        f"{k['speedup']:.2f}&times; the speed of fp16 cuBLAS, against a byte-ratio ceiling of {k['ceil']:.2f}&times;; the bandwidth figures were corrected "
        f"during the work after a benchmark flaw (an L2 leak of about 6%) was found. (2) Patched into vLLM 0.11 as a torch custom op and verified on the real "
        f"checkpoint, the kernel decodes Qwen2.5-1.5B at {e['inferno_b8']['1']:.0f} tokens/s with one sequence, against {e['awq_marlin']['1']:.0f} for vLLM's "
        f"default Marlin path ({r1:.2f}&times;), {e['awq_classic']['1']:.0f} for its classic AWQ kernel and {e['fp16']['1']:.0f} for fp16: parity with Marlin, not a win, "
        f"and with four concurrent sequences Marlin is faster ({r4:.2f}&times;). An earlier claim that the kernel beat Marlin was a measurement error and is "
        f"retracted. (3) In a 96-run paired campaign "
        f"(four policies, three load patterns, eight seeded repetitions) the pre-specified hypothesis that an adaptive router beats the "
        f"strongest fixed configuration is <b>rejected</b>: adaptive p99 latency was {lo:+.0f}% to {hi:+.0f}% relative to the best fixed "
        f"baseline (Holm-corrected p&nbsp;=&nbsp;0.023, the smallest attainable at n&nbsp;=&nbsp;8). (4) A controlled follow-up isolates the cause: "
        f"a second <i>active</i> replica on the same GPU multiplies p99 by {F['cont_active'][0]:.2f}&times; and {F['cont_active'][1]:.2f}&times; "
        f"at 30% and 60% of capacity, while a resident but idle replica costs {100 * (F['cont_idle'][0] - 1):+.1f}%; a headroom analysis of the existing data shows "
        f"that no pattern-aware choice between the two fixed tiers could have gained more than {F['headroom']['max_headroom_pct']:.1f}% of p99. (5) An MPI request router "
        f"adds about {F['mpi_pp'][64]['one_way_us_est']:.0f}&nbsp;&micro;s per hop in isolation and is statistically indistinguishable from an HTTP "
        f"router end to end."
        + _abstract_layer()))
    out.append(P(
        "Beyond the results, the report documents the measurement methodology in detail: open-loop load, capacity-relative load "
        "calibration, paired seeded workloads with randomized execution order, correctness gates before timing, L2-defeating kernel "
        "benchmarking, exact non-parametric tests with multiple-comparison control, and an audit trail of the errors found and fixed "
        "along the way, including two that overturned earlier claims. It states plainly what the evidence does not show, chiefly that the adaptive idea remains untested where each "
        "replica has its own GPU."))
    out.append(P("<b>Keywords:</b> LLM inference serving, CUDA, int4 quantization, memory-bound kernels, MPI, adaptive routing, GPU "
                 "co-residency, benchmarking methodology, paired statistics.", "small"))
    return out


def contents():
    out = [PageBreak(), Paragraph("Contents", S["h1"]), make_toc()]
    return out


def chapter_intro():
    out = H(1, "Introduction")
    out += H(2, "Context and motivation")
    out.append(P(
        f"Transformer-based language models {cite('vaswani')} generate text one token at a time. Each request therefore has two phases with very "
        "different hardware behaviour: a <i>prefill</i> phase that processes the whole prompt in parallel and is compute-intensive, and a "
        f"<i>decode</i> phase that produces one token per step and is memory-intensive {cite('splitwise')}. In decode, every step re-reads all "
        "model weights and the growing key-value (KV) cache from DRAM to perform, per weight, roughly one multiply-add. Runtime is then set "
        f"by bytes moved rather than arithmetic, the regime described by the roofline model {cite('roofline')}. Two consequences dominate "
        "the design of serving systems: batching many requests raises arithmetic intensity and throughput, and shrinking the bytes per "
        f"weight through quantization {cite('gptq', 'awq')} can speed decode up by nearly the compression ratio if the kernels are good "
        f"enough {cite('marlin')}."))
    out.append(P(
        f"Production engines such as vLLM {cite('vllm')} implement continuous batching in the style of Orca {cite('orca')}, paged KV-cache "
        f"management, chunked prefill {cite('sarathi')} and quantized kernels. Around them sits a second layer of decisions that is much less "
        f"standardized: how many replicas to run, how each is configured (batch cap, prefill chunk size, precision), and how requests are "
        f"routed among them as load shifts {cite('llumnix', 'clipper', 'alpaserve')}. This layer is where the question of <i>adaptivity</i> lives: a "
        "configuration that is best at low load is rarely best under a burst, and reconfiguring a running server is expensive."))
    out += H(2, "Why a single small GPU")
    out.append(P(
        "This study deliberately uses one consumer GPU (an RTX 4060 Laptop with 8&nbsp;GB) and a 1.5-billion-parameter model. That choice "
        "is a constraint, not a preference: an 8-billion-parameter model in fp16 needs about 16&nbsp;GB and cannot run here. It has an "
        "upside that shapes the whole report. Decode on this GPU is memory-bound in exactly the way it is on data-centre GPUs, the "
        "experiments are cheap enough to repeat many times, and every effect can be measured with paired designs and confidence "
        "intervals rather than single runs. It also has a cost, which the report treats as a first-class limitation: replicas that "
        "share one GPU contend with each other, which turned out to be the single most important finding (Chapter&nbsp;9)."))
    out += H(2, "Contributions")
    out.append(NUM([
        "<b>A verified fused int4 GEMV kernel</b> with a seven-stage optimization ladder that reaches the memory limit on large layers, a correctness gate that "
        "caught a real bug, an L2-defeating benchmark harness with run-to-run spread, a small-batch sweep that locates where a CUDA-core kernel stops paying off, "
        "and an integration into vLLM with an end-to-end comparison against the kernels vLLM actually runs (parity with the default Marlin path at one sequence).",
        "<b>A capacity-relative, open-loop, paired benchmarking methodology</b> for LLM serving, including load calibration, seeded "
        "workloads, randomized execution order, drain-aware throughput and energy accounting.",
        "<b>A pre-specified test of adaptive routing</b> with a 96-run campaign, the strongest fixed baselines, and generated verdicts "
        "that state null and negative results in plain language: the hypothesis was rejected.",
        "<b>A controlled mechanism study</b> that explains the negative result (GPU co-residency contention) rather than merely reporting it.",
        "<b>An MPI request router with per-hop instrumentation</b>, GPU-free coordination microbenchmarks, and a real-replica comparison "
        "against an HTTP router.",
        "<b>An audit trail</b> of measurement errors found and corrected during the work, including a benchmark flaw that inflated bandwidth figures and a baseline "
        "measurement that had wrongly made the custom kernel look faster than Marlin; offered as methodology in its own right.",
    ]))
    out += H(2, "How to read this report")
    out.append(P(
        "Chapters 2 and 3 state the problem and the falsifiable objectives. Chapters 4 and 5 review the literature and the state of "
        "prior work. Chapters 6 and 7 give the system design and the complete methodology, and are written to be sufficient for "
        "replication. Chapters 8 to 10 report results, discussion and limitations. Appendices contain reproduction commands, "
        "configurations, full data tables and a glossary. References marked with a dagger were checked against the publisher or "
        "preprint page while preparing this report; others are cited from the author's knowledge and should be confirmed before "
        "formal submission."))
    return out


def chapter_problem():
    out = H(1, "Problem statement")
    out += H(2, "The engineering problem")
    out.append(P(
        "A serving operator runs one or more model replicas on GPUs and receives a stream of requests whose arrival rate and length "
        "distribution change over time. Each request carries latency expectations: the first token should appear quickly "
        "(time to first token, TTFT) and subsequent tokens should stream steadily (time per output token, TPOT). The operator chooses a "
        "<i>configuration</i> for each replica (numeric precision, maximum concurrent sequences, prefill chunk size) and a <i>routing "
        "rule</i> that assigns each request to a replica. The question is whether a routing rule that reacts to observed load can "
        "outperform the best single static configuration, and by how much."))
    out += H(2, "Formalization")
    out.append(P(
        "Let a request <i>i</i> arrive at time <i>a<sub>i</sub></i> with prompt length <i>p<sub>i</sub></i> and output length "
        "<i>o<sub>i</sub></i>. A policy &pi; maps the request and the observable system state <i>s</i> (in-flight counts per replica, recent "
        "latency statistics) to a replica index. Under policy &pi; the request experiences a first-token time and a completion time, "
        "giving TTFT<sub><i>i</i></sub> and TPOT<sub><i>i</i></sub>. With latency objectives &tau;<sub>TTFT</sub> and &tau;<sub>TPOT</sub>, "
        f"<i>goodput</i> is the rate of requests meeting both {cite('distserve')}:"))
    out += EQ(r"\mathrm{Goodput}(\pi)=\frac{1}{T}\sum_{i}\mathbf{1}\left[\mathrm{TTFT}_i\leq\tau_{\mathrm{TTFT}}\ \wedge\ \mathrm{TPOT}_i\leq\tau_{\mathrm{TPOT}}\right]")
    out.append(P(
        "The comparison of interest is between an adaptive policy &pi;<sub>A</sub> and the best of a set of fixed policies "
        "<i>F</i> = {&pi;<sub>1</sub>, &hellip;, &pi;<sub>m</sub>}, evaluated on the tail of the latency distribution, because operators "
        f"are held to tail behaviour {cite('tail')}:"))
    out += EQ(r"\Delta_p=\frac{p_{99}(\pi_A)-\min_{\pi\in F}p_{99}(\pi)}{\min_{\pi\in F}p_{99}(\pi)}\qquad H_1:\ \Delta_p<0\ \text{ at non-inferior throughput}", size=14.5)
    out.append(P(
        "Two properties make this hard to answer credibly. First, the value of adaptivity depends entirely on how strong the fixed "
        "baselines are: a policy that beats a badly chosen baseline proves nothing. Second, the answer depends on hardware facts that "
        "papers rarely isolate, such as what happens when two replicas share a device."))
    out += H(2, "Why it is hard")
    out.append(BUL([
        "<b>Reconfiguration is expensive.</b> vLLM cannot change its batch cap or numeric precision on a live server; that requires "
        "a restart, which this study measures at roughly "
        f"{data.facts()['cold']:.0f}&nbsp;s, several times the duration of a traffic burst. Adaptivity must therefore act by routing across "
        "pre-loaded configurations.",
        "<b>Heterogeneity cuts both ways.</b> A low-latency configuration and a high-throughput configuration each win in some "
        "regimes, but co-locating them has a cost.",
        "<b>Tail metrics need many samples and paired designs.</b> With few repetitions, run-to-run variation on a thermally-limited "
        "laptop GPU can be larger than the effect being studied.",
        "<b>Measurement artefacts are easy to create.</b> Closed-loop load generators hide queueing "
        f"{cite('openclosed')}; caches make small kernels look faster than the memory system allows; wall-clock throughput that ignores "
        "backlog drain makes every policy look alike. Each of these was encountered and is addressed in Chapter&nbsp;7.",
    ]))
    out += H(2, "Scope, assumptions and non-goals")
    out.append(TABLE([
        ["In scope", "Out of scope"],
        ["Batch-1 decode kernel (int4 weights, fp16 activations, group size 128)", "Tensor-core kernels for large batches; training; attention kernels"],
        ["Routing across pre-loaded replicas on one GPU; HTTP and MPI routers", "Multi-GPU tensor or pipeline parallelism; live KV-cache migration"],
        ["Latency (TTFT, TPOT, e2e), throughput, goodput, energy per token", "Model quality: checkpoint accuracy after quantization is not evaluated"],
        ["Synthetic prompt text with server-verified token counts", "Real user prompts and semantic effects (e.g. prefix caching)"],
        ["One model family (Qwen2.5-1.5B) in fp16 / int8 / int4", "Larger models; the 7B cloud configuration is written but not run"],
    ], [0.5, 0.5], "Scope of the study."))
    out.append(P(
        "<b>Assumptions.</b> (i) The workload is a stationary or piecewise-stationary Poisson process whose parameters are known to "
        "the experimenter (not to the policy). (ii) Output length is controlled by the client and identical across configurations "
        "(the <font name='Mono'>ignore_eos</font> flag is set), so throughput comparisons are not confounded by different "
        "generation lengths. (iii) All processes run on one host, so a single monotonic clock is shared across processes "
        "(used to timestamp MPI hops)."))
    return out


def chapter_objectives():
    out = H(1, "Objectives, research questions and hypotheses")
    out += H(2, "Overall aim")
    out.append(P(
        "To determine, with defensible statistics and honest reporting, whether and why load-aware routing over heterogeneous replicas "
        "improves LLM serving on a single GPU; and to characterize the memory-bound kernel, the multi-replica communication layer and the "
        "energy/latency trade-offs that any such policy sits on top of."))
    out += H(2, "Research questions")
    out.append(TABLE([
        ["ID", "Research question"],
        ["RQ1", "How close can a hand-written fused int4 GEMV get to the memory roofline, and how does it compare with the kernels vLLM runs?"],
        ["RQ2", "Does an adaptive routing policy achieve lower p99 latency than the strongest fixed configuration at non-inferior throughput under steady, bursty and mixed-length load?"],
        ["RQ3", "If not, what is the mechanism? In particular, what does it cost to run a second active replica on the same GPU?"],
        ["RQ4", "What does MPI-based request routing cost, in isolation and inside a real serving path, and how does it scale with replicas?"],
        ["RQ5", "How do quantization level and batch cap trade latency, throughput and energy, and what does reconfiguration cost?"],
    ], [0.09, 0.91], "Research questions."))
    out += H(2, "Hypotheses with operational definitions")
    out.append(P(
        "Hypotheses were fixed, together with the analysis code and decision rules, before the campaign was run. Each has a stated "
        "boundary condition so that a null or negative result is informative."))
    out.append(TABLE([
        ["ID", "Statement", "Operational test", "Boundary condition"],
        ["H1", "Adaptive routing lowers p99 versus the best fixed configuration at non-inferior throughput.",
         "Per load pattern: paired (by repetition) difference in p99 against the fixed policy with the lowest mean p99 among error-free policies; significant only if the paired t-test and the Wilcoxon signed-rank test agree at Holm-adjusted &alpha; = 0.05; throughput &ge; 98% of the baseline.",
         "Under perfectly steady load a well-tuned fixed configuration is expected to match adaptive."],
        ["H2", "MPI multi-replica routing scales sub-linearly and its coordination overhead is measurable.",
         "Scaling efficiency = throughput(N) / (N &times; throughput(1)); per-hop overhead from inter-rank timestamps; MPI vs HTTP paired comparison.",
         "Replicas share one GPU, so linear scaling is not expected."],
        ["H3", "A hand-written fused int4 kernel reaches a fraction of the library kernel's throughput, and the gap is explained by profiler counters.",
         "Median-of-invocations kernel time vs cuBLAS fp16 and in-situ vLLM kernels; counter-level explanation via Nsight Compute.",
         "Batch 1 and very small batches; counters need GPU counter access."],
        ["H4", "Systems-level adaptivity, not model changes, drives efficiency gains (tokens per joule).",
         "Adaptive tokens/joule relative to the best fixed configuration, per load pattern.",
         "Energy sampled from the GPU only, at 200 ms resolution."],
    ], [0.06, 0.24, 0.45, 0.25], "Hypotheses, tests and boundary conditions.", font=7.5))
    out += H(2, "Measurable objectives and acceptance criteria")
    out.append(TABLE([
        ["#", "Objective", "Acceptance criterion"],
        ["O1", "Reproducible environment", "A clean checkout builds and runs the smoke test; environment doctor passes; resolution date-pinned."],
        ["O2", "Trustworthy measurement harness", "Open-loop client, TTFT/TPOT/e2e/goodput, power and energy; unit-tested against mock servers."],
        ["O3", "Correct, fast kernel", "All kernel variants match a CPU reference on every shape in every invocation; timing with L2 defeated and spread reported."],
        ["O4", "Capacity-relative load", "Request rates derived from measured saturation capacity, not guessed."],
        ["O5", "Adequately powered H1 test", "&ge; 8 paired repetitions (7 is the theoretical minimum at which a Holm-corrected Wilcoxon test over three patterns can fall below 0.05, with no margin; 8 leaves headroom)."],
        ["O6", "Mechanism isolation", "A controlled experiment that varies only how the GPU is shared, at fixed total load."],
        ["O7", "MPI characterization", "GPU-free microbenchmarks plus a paired real-replica comparison with an HTTP router."],
        ["O8", "Generated, auditable reporting", "Every number in reports traces to a raw file; verdict text produced by code."],
    ], [0.06, 0.28, 0.66], "Objectives and acceptance criteria."))
    out += H(2, "Falsifiability and what would change the conclusions")
    out.append(P(
        "H1 would have been supported had the adaptive policy shown a significantly lower p99 than the best fixed configuration on at "
        "least one non-steady pattern with non-inferior throughput. It was not. The conclusions would change if (i) the adaptive policy "
        "were re-evaluated with replicas on separate GPUs and won, (ii) a stronger fixed baseline were found that the campaign "
        "excluded, or (iii) the measurement instruments were shown to be biased. Chapter&nbsp;7 describes the checks that guard against (iii); "
        "Chapter&nbsp;11 lists what remains open for (i)."))
    return out
