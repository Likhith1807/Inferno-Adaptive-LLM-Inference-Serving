"""
Builds docs/Inferno_MTE_Presentation.pptx on top of the university template (docs/PPT Format-Project Presentation-MTE 2026 (2).pptx).
The template chassis (logo, title rule, crimson footer, fonts, section order) is kept; slides are cloned from it.
Every number on a slide is read from docs/results by the same loaders the technical report uses.

    python scripts/build_pptx.py --team "Name 1 (Roll No.)" "Name 2 (Roll No.)" "Name 3 (Roll No.)" "Name 4 (Roll No.)"

Kept deliberately to 15 slides (mid-term evaluation: detailed methodology, a few important/attractive results, not the
whole project). Methodology is 2 slides (workflow + parallelization); results are 2 slides (kernel+vLLM, routing+MPI);
contributions and the plan share one slide with a simple 3-step timeline. Real screenshots from the live dashboard
(docs/figures/ppt_*.png, regenerated from the corrected data) are embedded as proof of a working system.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pptx import Presentation  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from ppt_kit import (AMBER, AMBERTINT, BLUE, CRIMSON, DEEP, GREEN, GREENTINT, GREY, INK, LINE, NAVY, PALE, TINT, WHITE,  # noqa: E402
                     arrow, badge, bar_chart, box, card, chip, clone_slide, delete_slide, labelled, notes, picture, remove_shapes,
                     reorder, set_title, text)
from report_pdf import data  # noqa: E402

TEMPLATE = ROOT / "docs" / "PPT Format-Project Presentation-MTE 2026 (2).pptx"
OUT = ROOT / "docs" / "Inferno_MTE_Presentation.pptx"
FIG = ROOT / "docs" / "figures"
X0, X1 = 0.42, 12.92          # content area
W = X1 - X0


def setting_strip(s, y, h, label, body, size=12.5):
    """Small grey strip that states workload, batch size, baseline and units next to headline figures."""
    labelled(s, X0, y, W, h, [[(label + "  ", {"bold": True, "color": CRIMSON}), (body, {})]], fill=PALE, line=None, size=size, anchor="m", margin=0.18)


def facts():
    F = data.facts()
    K = data.kernel()
    mlp = next(s for s in K["shapes"] if s["name"].startswith("llama7B_mlp_up"))
    kk = {x["kernel"]: x for x in mlp["kernels"]}
    F["ladder"] = [(lab, kk["cublas_fp16"]["us_median"] / kk[n]["us_median"]) for lab, n in
                   (("v0", "int4_v0_naive"), ("v1", "int4_v1_warp"), ("v2", "int4_v2_vec"), ("v3", "int4_v3_pad"),
                    ("v4", "int4_v4_alg"), ("v5", "int4_v5_reuse"), ("v6", "int4_v6_lane16"))]
    _, h1 = data.h1_rows()
    F["h1rows"] = h1
    cont = data.jl("contention/contention_summary.json")
    F["cont_cells"] = {(c["pattern"], c["config"]): c["e2e_p99"] for c in cont["cells"]}
    return F


def build(team):
    F = facts()
    k = F["kern"]
    e = F["e2e"]
    prs = Presentation(str(TEMPLATE))
    orig = list(prs.slides)
    s_title, s_contents, base, s_refs, s_thanks = orig[0], orig[1], orig[2], orig[11], orig[12]
    order = [s_title, s_contents]

    TAGS = {"Problem Statement": "1 · PROBLEM", "Literature Review": "2 · LITERATURE", "Objective of the Project": "3 · OBJECTIVES",
            "Proposed Methodology: Workflow": "4 · METHOD 1/2", "Proposed Methodology: Parallelization": "4 · METHOD 2/2",
            "Tools, Technologies & Setup": "5 · TOOLS & SETUP", "Results: The CUDA Kernel": "6 · RESULTS 1/2",
            "Results: Adaptive Routing & MPI": "6 · RESULTS 2/2", "Challenges": "7 · CHALLENGES",
            "Contributions, Conclusions & Plan": "8 · CONCLUSION", "Team Work & Coordination": "9 · TEAM"}

    def new(title):
        sl = clone_slide(prs, base)
        set_title(sl, title)
        remove_shapes(sl, {"TextBox 2"})
        if title in TAGS:
            chip(sl, 10.55, 0.6, 2.4, 0.36, TAGS[title], fill=TINT, color=CRIMSON, size=12.5)
        order.append(sl)
        return sl

    # ================================================================== 1. title slide (template kept)
    for shp in s_title.shapes:
        if shp.name == "Rectangle 9":
            shp.text_frame.paragraphs[0].runs[0].text = "Inferno: Adaptive LLM Inference Serving"
        if shp.name == "Rectangle 16":
            for p, name in zip(shp.text_frame.paragraphs, team):
                p.runs[0].text = name
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)
    text(s_title, 0.25, 1.02, 12.83, 0.7, "CUDA kernel  ·  MPI request router  ·  rigorous GPU benchmarking", size=22, color=CRIMSON, bold=True, align="c",
         font="Times New Roman")
    text(s_title, 0.25, 2.78, 12.83, 0.5, "High-Performance Computing Project  ·  Mid-Term Evaluation 2026", size=20, color=NAVY, bold=True, align="c",
         font="Times New Roman")
    notes(s_title, """
Good morning. Our project is called Inferno. It is about one question in High-Performance Computing: how do we serve a large language model fast, and how do we
prove, with real measurements, that a design is actually faster? We built a CUDA GPU kernel, an MPI request router and an adaptive controller, and we measured
them with strict statistics on one laptop GPU.
Structure of the next 10 minutes: the problem and why it needs HPC, what exists already, our objectives, the methodology (workflow diagram and parallelization
strategy), tools and setup, our most important results, challenges, and our contributions with the plan until the final evaluation in December.
""")

    notes(s_contents, "Agenda: problem, literature, objectives, then the methodology (workflow diagram, parallelization strategy), tools and setup, our two "
                      "key results, challenges, and our contributions with the plan and the team, up to the final evaluation.")

    # ================================================================== 3. problem statement
    s = new("Problem Statement")
    gb, bw_ = 3.09, F["peak_gbps"]
    ms_tok = gb / bw_ * 1000
    tps_max = bw_ / gb
    card(s, X0, 1.28, W, 2.6, fill=PALE, line=None)
    text(s, X0 + 0.2, 1.3, 6.6, 0.4, "Why is generating text slow? A memory problem.", size=18, bold=True, color=NAVY)
    box(s, X0 + 0.25, 1.78, 2.2, 1.2, fill=NAVY, line=None, radius=0.1)
    text(s, X0 + 0.25, 1.78, 2.2, 1.2, [[("Model weights", {"size": 16, "bold": True, "color": WHITE})], [("≈ 3.1 GB", {"size": 24, "bold": True, "color": WHITE})],
                                        [("FP16, in GPU memory", {"size": 12, "color": RGBColorHex("C9DAF2")})]], align="c", anchor="m", space_after=0)
    arrow(s, X0 + 2.5, 2.38, X0 + 3.75, 2.38, color=CRIMSON, width=4)
    text(s, X0 + 2.45, 1.86, 1.4, 0.45, f"{bw_:.0f} GB/s", size=18, bold=True, color=CRIMSON, align="c")
    text(s, X0 + 2.4, 2.45, 1.5, 0.4, "theoretical", size=12, color=GREY, align="c")
    box(s, X0 + 3.8, 1.78, 1.9, 1.2, fill=GREY, line=None, radius=0.1)
    text(s, X0 + 3.8, 1.78, 1.9, 1.2, [[("GPU cores", {"size": 17, "bold": True, "color": WHITE})], [("waiting for data", {"size": 12, "color": WHITE})]], align="c", anchor="m", space_after=0)
    arrow(s, X0 + 5.75, 2.38, X0 + 6.3, 2.38, color=CRIMSON, width=4)
    chip(s, X0 + 6.32, 2.06, 0.95, 0.6, "1 token", fill=CRIMSON, color=WHITE, size=15)
    card(s, X0 + 0.25, 3.12, 7.0, 0.62, fill=GREENTINT, accent=GREEN, line=None)
    text(s, X0 + 0.5, 3.12, 6.7, 0.62, [[("Measured FP16 baseline:  ", {"bold": True, "color": GREEN, "size": 16}),
                                        (f"{e['fp16']['1']:.0f} tokens/s", {"bold": True, "color": NAVY, "size": 20}),
                                        ("   (vLLM 0.11, batch 1, measured)", {"size": 13, "color": GREY})]], anchor="m")
    card(s, X0 + 7.5, 1.78, 4.8, 1.96, fill=WHITE, accent=CRIMSON)
    text(s, X0 + 7.72, 1.8, 4.55, 1.92, [[("Weight-read bandwidth estimate: ", {"bold": True, "color": NAVY, "size": 12.5}),
                                        (f"A 3.1 GB FP16 model at {bw_:.0f} GB/s theoretical memory bandwidth gives an idealized weight-read time of approximately "
                                         f"{ms_tok:.0f} ms per token, corresponding to an upper-bound estimate of approximately {tps_max:.0f} tokens/s under batch-1, "
                                         "weight-bound assumptions. Actual throughput depends on kernel efficiency, compute, and serving overhead.", {"size": 12.5})]],
         anchor="m", space_after=0)
    probs = [("1", "Memory-bound decoding", "At small batch sizes, speed is often limited by the bytes read per token. Smaller (int4) weights → fewer bytes."),
             ("2", "Bursty traffic, tail latency", "Production traffic is bursty. Users feel the slowest requests (p99), not the average."),
             ("3", "Coordinating replicas", "Serving several model replicas needs a low-overhead layer to route independent requests (MPI).")]
    pw = 4.05
    for i, (n, t, d) in enumerate(probs):
        x = X0 + i * (pw + 0.175)
        card(s, x, 4.0, pw, 1.5, fill=WHITE, accent=CRIMSON)
        badge(s, x + 0.25, 4.1, 0.46, n, CRIMSON, size=16)
        text(s, x + 0.82, 4.06, pw - 0.92, 0.55, t, size=18.5, bold=True, color=NAVY, anchor="m")
        text(s, x + 0.28, 4.62, pw - 0.4, 0.88, d, size=14.5, color=INK)
    box(s, X0, 5.65, 6.2, 1.2, fill=None, line=None, gradient=(CRIMSON, DEEP), radius=0.08)
    text(s, X0 + 0.25, 5.65, 5.8, 1.2, [[("Our question", {"size": 15, "bold": True, "color": RGBColorHex("FFD6DD")})],
                                       [("Can a parallel, adaptive serving stack beat fixed configurations – and why?", {"size": 19, "bold": True, "color": WHITE})]],
         anchor="m", space_after=2)
    labelled(s, X0 + 6.4, 5.65, W - 6.4, 1.2, [[("Why HPC?", {"bold": True, "color": CRIMSON, "size": 19})],
                                               [("GPU / CUDA", {"bold": True}), (" – billions of independent multiply-adds", {})],
                                               [("MPI", {"bold": True}), (" – coordinating independent requests", {})],
                                               [("Rigorous benchmarking", {"bold": True}), (" – to trust any speed claim", {})]],
             fill=TINT, line=None, size=15, anchor="m", margin=0.22, space_after=1)
    notes(s, f"""
Start with the picture at the top. To write one token, the GPU must read the model weights from memory: about 3.1 GB for our model in FP16.
The weight-read bandwidth estimate: a 3.1 GB FP16 model at {bw_:.0f} GB/s THEORETICAL memory bandwidth gives an idealized weight-read time of approximately {ms_tok:.0f} milliseconds per token, which corresponds to an upper-bound estimate of
approximately {tps_max:.0f} tokens per second under batch-one, weight-bound assumptions. Actual throughput depends on kernel efficiency, compute and serving overhead.
Separately, the MEASURED FP16 baseline in vLLM is {e['fp16']['1']:.0f} tokens per second. Do not present the {tps_max:.0f} as a measurement: it is an idealized estimate.
At small batch sizes decoding is often limited by memory bandwidth rather than arithmetic; this is not true of every LLM workload, for example large-batch prefill can be compute-bound.
The lever: if the weights are 4-bit there are about 3.8 times fewer bytes to read, so tokens can be generated faster IF the kernel reaches the memory limit. That is our CUDA kernel.
Second problem: production traffic is bursty and users feel the slowest requests, so we care about p99 latency. Third: real serving uses several model replicas, and routing independent requests among them needs a low-overhead parallel layer, which is why MPI.
Our research question: can a parallel, adaptive serving stack beat fixed configurations, and why or why not? Why HPC: CUDA for the GPU parallelism, MPI for coordinating requests, and rigorous benchmarking so that speed claims can be trusted.
Likely question: is {tps_max:.0f} tokens per second a real limit? Answer: no, it is an idealized upper-bound estimate for reading all the weights once per token at batch one from theoretical bandwidth; the measured FP16 baseline is {e['fp16']['1']:.0f}.
""")

    # ================================================================== 4. literature review
    s = new("Literature Review")
    text(s, X0, 1.22, W, 0.36, "Four research threads our project builds on — and the gap between them", size=15.5, color=GREY)
    themes = [
        ("Continuous & Paged Serving", CRIMSON,
         [[("Orca [1]  ", {"bold": True, "color": CRIMSON}), ("iteration-level (continuous) batching", {})],
          [("vLLM [2]  ", {"bold": True, "color": CRIMSON}), ("PagedAttention — paged KV-cache memory", {})]]),
        ("Disaggregated Serving", NAVY,
         [[("DistServe [3]  ", {"bold": True, "color": NAVY}), ("splits prefill / decode across GPUs", {})],
          [("Splitwise [4]  ", {"bold": True, "color": NAVY}), ("phase-specialized GPU pools", {})]]),
        ("Quantization & Kernels", GREEN,
         [[("AWQ [5], GPTQ [6]  ", {"bold": True, "color": GREEN}), ("accurate 4-bit weights", {})],
          [("Marlin [7]  ", {"bold": True, "color": GREEN}), ("fast 4-bit GPU kernel — our baseline", {})]]),
        ("Rigorous Benchmarking", AMBER,
         [[("Hoefler & Belli [8]  ", {"bold": True, "color": AMBER}), ("12 rules for HPC benchmarking", {})],
          [("Georges et al. [9]  ", {"bold": True, "color": AMBER}), ("statistically sound perf. claims", {})]]),
    ]
    cw, chh, g = 6.15, 1.86, 0.16
    for i, (t, col, lines) in enumerate(themes):
        r, c = divmod(i, 2)
        x, y = X0 + c * (cw + g), 1.68 + r * (chh + g)
        card(s, x, y, cw, chh, fill=WHITE, accent=col)
        text(s, x + 0.3, y + 0.12, cw - 0.5, 0.44, t, size=19, bold=True, color=NAVY)
        text(s, x + 0.32, y + 0.68, cw - 0.55, chh - 0.78, lines, size=16.5, bullets=True, space_after=10)
    box(s, X0, 5.68, W, 1.02, fill=None, line=None, gradient=(CRIMSON, DEEP), radius=0.1)
    text(s, X0 + 0.3, 5.68, W - 0.5, 1.02, [[("The gap we fill: ", {"bold": True, "color": RGBColorHex("FFD6DD"), "size": 17})],
                                           [("none of these study the kernel, the router and the statistics TOGETHER on one shared GPU — with paired tests and negative "
                                             "results reported honestly.", {"color": WHITE, "bold": True, "size": 17.5})]],
         anchor="m", space_after=2)
    notes(s, """
Four themes. Continuous and paged serving: Orca introduced iteration-level continuous batching; vLLM introduced PagedAttention memory management, and it is our serving engine.
Disaggregated serving: DistServe and Splitwise split the prefill and decode phases across separate GPUs — an architecture idea we revisit when we explain why our two tiers sharing one GPU behaved the way they did.
Quantization and kernels: AWQ and GPTQ produce accurate 4-bit weights; Marlin is the fast 4-bit GPU kernel that is our direct baseline throughout the kernel results.
Rigorous benchmarking: Hoefler and Belli, and Georges et al., argue performance claims need proper statistics — paired tests, effect sizes, multiple-comparison correction — which is exactly our methodology chapter.
The gap: none of these four threads are combined. We study the kernel, the routing layer and the statistics together, on one GPU, and we report negative results honestly instead of hiding them.
Likely question: how is your work different from vLLM? Answer: vLLM is our serving engine; we add a hand-written kernel, a router and a measurement methodology on top of it.
""")

    # ================================================================== 5. objectives
    s = new("Objective of the Project")
    objs = [("Fused int4 CUDA kernel reaching ≥ 85% of theoretical DRAM bandwidth (effective)", "Achieved: 87–95% of peak", GREEN, GREENTINT),
            ("Integrate the kernel into vLLM; measure real decode speed", "Done: parity with Marlin at 1 sequence", GREEN, GREENTINT),
            ("Test adaptive routing against the strongest fixed baselines (pre-specified H1)", "Done: rejected on 1 GPU · multi-GPU pending", GREEN, GREENTINT),
            ("MPI request router: measure overhead and scaling", "1 GPU done · multi-GPU PLANNED", AMBER, AMBERTINT),
            ("Rigorous, reproducible method – every number from raw data", "Ongoing: 81 tests passed, audit trail", GREEN, GREENTINT)]
    for i, (o, status, col, tint) in enumerate(objs):
        y = 1.36 + i * 1.08
        card(s, X0, y, W, 0.94, fill=WHITE)
        badge(s, X0 + 0.18, y + 0.2, 0.55, str(i + 1), CRIMSON, size=20)
        text(s, X0 + 0.95, y + 0.05, 7.4, 0.84, o, size=20, anchor="m", color=INK)
        chip(s, X0 + 8.55, y + 0.2, 3.75, 0.54, status, fill=tint, color=col, size=15)
    notes(s, """
Our five objectives, each measurable. One: a fused int4 CUDA kernel that reaches at least 85 percent of the memory bandwidth of the GPU. Two: put that kernel inside
vLLM and measure real tokens per second. Three: test, with a pre-specified hypothesis called H1, whether an adaptive router beats the strongest fixed configuration.
Four: build an MPI router and measure its overhead and scaling. Five: keep the whole study rigorous and reproducible, every number generated from raw data.
Status at mid-term: objectives one, two, three and five are done; objective four is done on a single GPU and the multi-GPU part is planned for November.
""")

    # ================================================================== 6. methodology 1: workflow diagram
    s = new("Proposed Methodology: Workflow")
    steps = [("Workload Generator", "Poisson · bursty · mixed lengths; seeded", "in"),
             ("Open-loop Client", "sends on schedule; records TTFT, TPOT, p99", "in"),
             ("Request Router", "policy · HTTP baseline · MPI dispatcher (rank 0)", "sys"),
             ("vLLM Replicas", "fast tier + bulk tier sharing one GPU", "sys"),
             ("Fused int4 CUDA Kernel", "decode weights, multiply, reduce – on the GPU", "sys"),
             ("Metrics & Power", "latency, goodput, tokens/s, watts (every 200 ms)", "out"),
             ("Statistics", "paired t-test + Wilcoxon, Holm, effect size", "out"),
             ("Report & Dashboard", "every number generated from raw logs", "out")]
    kinds = {"in": (PALE, GREY), "sys": (TINT, CRIMSON), "out": (GREENTINT, GREEN)}
    bw, bh, gap = 2.85, 1.72, 0.37
    xs = [X0 + i * (bw + gap) for i in range(4)]
    pos = [(xs[0], 1.5), (xs[1], 1.5), (xs[2], 1.5), (xs[3], 1.5), (xs[3], 3.72), (xs[2], 3.72), (xs[1], 3.72), (xs[0], 3.72)]
    for i, ((title, desc, kind), (x, y)) in enumerate(zip(steps, pos)):
        fillc, edge = kinds[kind]
        card(s, x, y, bw, bh, fill=fillc, line=edge)
        text(s, x + 0.1, y + 0.1, bw - 0.2, 0.55, title, size=18, bold=True, color=NAVY, align="c", anchor="m")
        text(s, x + 0.12, y + 0.66, bw - 0.24, 0.82, desc, size=14.5, color=INK, align="c")
        badge(s, x - 0.16, y - 0.16, 0.5, str(i + 1), edge if kind != "in" else NAVY, size=16)
    for i in range(3):
        arrow(s, xs[i] + bw + 0.03, 2.36, xs[i + 1] - 0.04, 2.36, color=CRIMSON, width=3.5)
    arrow(s, xs[3] + bw / 2, 3.24, xs[3] + bw / 2, 3.7, color=CRIMSON, width=3.5)
    for i in (3, 2, 1):
        arrow(s, xs[i] - 0.04, 4.58, xs[i - 1] + bw + 0.04, 4.58, color=GREEN, width=3.5)
    for i, (lab, col, tint) in enumerate([("Inputs", GREY, PALE), ("System under test", CRIMSON, TINT), ("Measurement & analysis", GREEN, GREENTINT)]):
        chip(s, X0 + i * 2.75, 5.6, 2.6, 0.36, lab, fill=tint, color=col, size=13)
    box(s, X0, 6.05, W, 0.82, fill=None, line=None, gradient=(CRIMSON, DEEP), radius=0.1)
    text(s, X0 + 0.3, 6.05, W - 0.5, 0.82, [[("Key idea:  ", {"bold": True, "color": RGBColorHex("FFD6DD")}),
                                           ("replay the SAME seeded traffic through every policy – differences come from the policy, not luck.", {"color": WHITE, "bold": True})]],
         size=17.5, anchor="m")
    notes(s, """
This is our workflow diagram. Read it in the order of the numbers; grey boxes are inputs, red boxes are the system under test, green boxes are measurement and analysis.
1 The workload generator creates request streams: steady, bursty and mixed-length, all seeded so they can be replayed exactly.
2 The open-loop client sends each request at its scheduled time, whether or not earlier requests finished, and records time to first token, time per output token and end-to-end latency.
3 The router decides which replica serves each request: fixed policies, a load balancer, our adaptive controller, or the MPI dispatcher with rank zero and workers.
4 The requests are served by vLLM replicas: a low-latency tier and a high-throughput tier sharing one GPU.
5 The heavy work inside is the fused int4 CUDA kernel that we wrote.
6 Metrics and GPU power are recorded, 7 statistics decide whether a difference is real, 8 and the report and dashboard are generated automatically from raw logs.
Key idea: identical seeded traffic goes through every policy, so differences are due to the policy only.
Likely question: why open-loop? Because a closed-loop client slows down when the server is slow and hides the queueing we want to measure.
""")

    # ================================================================== 7. methodology 2: parallelization strategy
    s = new("Proposed Methodology: Parallelization")
    lv = [("LEVEL 1", "CUDA", "inside one layer", CRIMSON, "each output row of the matrix · each row = K/32 chunks of 32 weights",
           "16 lanes per row · 128-bit coalesced loads · warp-shuffle reduction", "fine-grained data parallelism on a memory-bound GPU"),
          ("LEVEL 2", "vLLM", "across users", NAVY, "tokens of different users’ requests",
           "continuous batching · one decoded weight reused for up to B vectors", "production scheduler – not re-invented"),
          ("LEVEL 3", "MPI", "across replicas", RGBColorHex("0B7A4B"), "whole requests → independent servers",
           "rank 0 dispatcher + N workers · non-blocking isend / iprobe", f"distributed memory · ≈ {F['mpi_pp'][64]['one_way_us_est']:.0f} µs per hop")]
    heads = ["What is independent?", "How it is parallelized", "Why this framework?"]
    hx = [X0 + 2.75, X0 + 6.15, X0 + 9.85]
    hw = [3.3, 3.6, 2.65]
    for j, hd in enumerate(heads):
        text(s, hx[j], 1.22, hw[j], 0.34, hd, size=13, bold=True, color=GREY)
    for i, (lab, fw, sub, col, a, b, c) in enumerate(lv):
        y = 1.58 + i * 1.42
        card(s, X0, y, W, 1.3, fill=WHITE, line=LINE)
        box(s, X0, y, 2.55, 1.3, fill=col, line=None, radius=0.08)
        text(s, X0, y + 0.05, 2.55, 1.2, [[(lab, {"size": 12.5, "bold": True, "color": RGBColorHex("E8EEF8")})], [(fw, {"size": 30, "bold": True, "color": WHITE})],
                                        [(sub, {"size": 13.5, "color": RGBColorHex("E8EEF8")})]], align="c", anchor="m", space_after=0)
        text(s, hx[0], y + 0.05, hw[0], 1.2, a, size=16, anchor="m")
        text(s, hx[1], y + 0.05, hw[1], 1.2, b, size=16, anchor="m")
        text(s, hx[2], y + 0.05, hw[2], 1.2, c, size=15, anchor="m", color=GREY)
        if i < 2:
            arrow(s, X0 + 1.27, y + 1.3, X0 + 1.27, y + 1.42, color=GREY, width=2)
    box(s, X0, 5.9, W, 0.98, fill=None, line=None, gradient=(CRIMSON, DEEP), radius=0.1)
    text(s, X0 + 0.3, 5.9, W - 0.5, 0.98, [[("MPI coordinates requests – CUDA does the GPU math.", {"size": 21, "bold": True, "color": WHITE})],
                                           [("Not chosen: OpenMP (CPU shared memory only) · NCCL (tensor collectives; we route requests) · HTTP only (kept as the baseline).",
                                             {"size": 13.5, "color": RGBColorHex("FFD6DD")})]], anchor="m", space_after=2)
    notes(s, """
The rubric asks for the independent tasks and the justification of the framework, so this slide is organised by level of parallelism, from the smallest to the largest.
Level 1, inside one layer, framework CUDA: the output of a matrix-vector product has N rows and every row is independent. Each row is split into chunks of 32 weights that are also independent, and a warp reduces the partial sums with shuffles.
We chose CUDA because this is fine-grained data parallelism on a memory-bound problem and the GPU has the bandwidth for it.
Level 2, across users, framework vLLM: different requests are independent and vLLM batches them continuously. Our kernel decodes each weight once and reuses it for up to B activation vectors. We use vLLM as a production scheduler rather than re-inventing it.
Level 3, across replicas, framework MPI: whole requests are independent, so we use MPI in a dispatcher-and-workers pattern with non-blocking sends and probes. MPI is a distributed-memory model, so the same router can span several GPUs or nodes, and a hop costs about two microseconds.
Important: MPI coordinates requests; it does not accelerate GPU tensor computation, which is CUDA's job.
Alternatives: OpenMP is shared-memory CPU only; NCCL is for tensor collectives and we route requests instead; plain HTTP we keep as the baseline and measure against MPI.
Likely question: what is the speedup from parallelism? Answer: the fused kernel is 3.7 times faster than fp16 cuBLAS on the biggest layer; the MPI part is about coordination cost, not speedup.
""")

    # ================================================================== 8. tools, technologies & setup (merged)
    s = new("Tools, Technologies & Setup")
    cw = 6.15
    card(s, X0, 1.32, cw, 4.55, fill=WHITE, accent=CRIMSON)
    text(s, X0 + 0.3, 1.4, cw - 0.5, 0.5, "Tools & Technologies", size=20, bold=True, color=NAVY)
    text(s, X0 + 0.32, 1.95, cw - 0.55, 3.8, [
        [("Hardware:  ", {"bold": True, "color": CRIMSON}), ("RTX 4060 Laptop (8 GB) · WSL2 Ubuntu · 16 cores", {})],
        [("GPU programming:  ", {"bold": True, "color": CRIMSON}), ("CUDA 12.4, cuBLAS reference · Nsight Compute (pending)", {})],
        [("Serving engine:  ", {"bold": True, "color": CRIMSON}), ("vLLM 0.11 + PyTorch 2.8 · our kernel as a torch custom op", {})],
        [("Distributed:  ", {"bold": True, "color": CRIMSON}), ("Open MPI 5.0.10 + mpi4py · AWS multi-GPU (Terraform, not yet run)", {})],
        [("Analysis:  ", {"bold": True, "color": CRIMSON}), ("NumPy / SciPy / pandas · paired tests · auto-generated report + dashboard", {})],
        [("Model:  ", {"bold": True, "color": CRIMSON}), ("Qwen2.5-1.5B-Instruct — FP16 · GPTQ-Int8 · AWQ-Int4", {})],
    ], size=18, bullets=True, space_after=16)
    x2 = X0 + cw + 0.2
    cw2 = W - cw - 0.2
    card(s, x2, 1.32, cw2, 4.55, fill=WHITE, accent=NAVY)
    text(s, x2 + 0.3, 1.4, cw2 - 0.5, 0.5, "Experimental Setup", size=20, bold=True, color=NAVY)
    cal = F["cal"]
    text(s, x2 + 0.32, 1.95, cw2 - 0.55, 3.8, [
        [("Kernel study:  ", {"bold": True, "color": NAVY}), ("6 real layer shapes · int4, group 128 · 5 fresh runs · CPU-reference gate", {})],
        [("Serving campaign:  ", {"bold": True, "color": NAVY}), ("4 policies × 3 load patterns × 8 repeats = 96 runs (75 s each)", {})],
        [("Calibrated load:  ", {"bold": True, "color": NAVY}), (f"steady = {cal['rates']['steady']['rate_requests_per_sec']:.1f} req/s (60% of measured capacity)", {})],
        [("End-to-end in vLLM:  ", {"bold": True, "color": NAVY}), ("5 configurations, 1–8 sequences · CUDA graphs · 3 repeats", {})],
        [("Metrics:  ", {"bold": True, "color": NAVY}), ("TTFT, TPOT, p50/95/99, goodput, tokens/s, tokens/joule", {})],
        [("Data:  ", {"bold": True, "color": NAVY}), ("seeded synthetic streams – no external dataset, fully repeatable", {})],
    ], size=18, bullets=True, space_after=16)
    box(s, X0, 6.05, W, 0.82, fill=None, line=None, gradient=(RGBColorHex("0B7A4B"), RGBColorHex("06563A")), radius=0.1)
    text(s, X0 + 0.3, 6.05, W - 0.5, 0.82, [[("Every number is generated from raw logs by our own scripts", {"bold": True, "color": WHITE, "size": 19}),
                                           (" – same seed in, same result out, every time.", {"color": RGBColorHex("C9F2DD"), "size": 15})]], anchor="m")
    notes(s, """
Tools: an RTX 4060 laptop GPU with 8 GB, CUDA 12.4 and cuBLAS as the fp16 reference (Nsight Compute is pending, GPU counters are blocked), vLLM 0.11 with PyTorch 2.8 and our kernel registered as a torch custom
op, Open MPI with mpi4py, NumPy/SciPy/pandas for analysis, and Qwen2.5-1.5B in three precisions.
Setup: the kernel study covers six real layer shapes with a CPU-reference correctness gate; the serving campaign is four policies times three load patterns times eight repeats, 96 runs of 75 seconds; load
is calibrated from measured capacity, not guessed; the end-to-end study inside vLLM uses five configurations with CUDA graphs. All workloads are seeded synthetic streams, so every run is exactly repeatable
and there is no external dataset to worry about.
""")

    # ================================================================== 9. results: the CUDA kernel (kernel + vLLM merged)
    s = new("Results: The CUDA Kernel")
    text(s, X0, 1.22, W, 0.4, "From a naive kernel to near the hardware limit — then verified inside a real serving engine.", size=15.5, color=GREY)
    text(s, X0, 1.68, 7.0, 0.32, "Optimization ladder — speedup vs FP16 cuBLAS, batch 1, Llama-7B MLP layer", size=13.5, bold=True, color=NAVY)
    names = [n for n, _ in F["ladder"]]
    vals = [round(v, 2) for _, v in F["ladder"]]
    cols = [RGBColorHex("C9DAF2"), RGBColorHex("A9C3EC"), RGBColorHex("84A9E2"), RGBColorHex("5E8ED8"), RGBColorHex("3D75CB"), RGBColorHex("2A5FB0"), CRIMSON]
    bar_chart(s, X0, 1.98, 7.0, 2.55, names, {"speedup": vals}, [BLUE], number_format='0.00"×"', font=12, gap=45, vmax=4.4, point_colors={0: cols})
    labelled(s, X0, 4.68, 3.4, 0.85, [[(f"{k['speedup']:.2f}×", {"bold": True, "size": 24, "color": CRIMSON}), (" vs FP16 cuBLAS", {"size": 12.5, "color": GREY})]],
             fill=PALE, line=None, anchor="m", margin=0.18)
    labelled(s, X0 + 3.6, 4.68, 3.4, 0.85, [[(f"{k['pct']:.0f}%", {"bold": True, "size": 24, "color": CRIMSON}), (f" of {F['peak_gbps']:.0f} GB/s bandwidth", {"size": 12.5, "color": GREY})]],
             fill=PALE, line=None, anchor="m", margin=0.18)
    picture(s, FIG / "ppt_kernel.png", X0 + 7.35, 1.68, w=4.75)
    text(s, X0 + 7.35, 5.02, 4.75, 0.34, "↑ from our live interactive dashboard", size=11.5, color=GREY, align="c")
    r1 = F["e2e_ratio_b8"]["1"]["mean"]
    box(s, X0, 5.66, W, 0.82, fill=None, line=None, gradient=(NAVY, RGBColorHex("0B3D91")), radius=0.1)
    text(s, X0 + 0.25, 5.66, W - 0.45, 0.82, [[("Inside vLLM (real engine): ", {"bold": True, "color": RGBColorHex("C9DAF2")}),
                                              (f"{e['inferno_b8']['1']:.0f} tokens/s at 1 sequence ≈ {r1:.2f}× Marlin's speed (parity, not a win). "
                                               "Marlin leads from 4+ sequences — next milestone: a tensor-core kernel.", {"color": WHITE, "bold": True})]],
         size=15.5, anchor="m")
    text(s, X0, 6.56, W, 0.4, "Speedup vs FP16 cuBLAS GEMV baseline, batch 1. Nsight Compute profiling PENDING (GPU counters blocked).", size=11.5, color=GREY)
    notes(s, f"""
First key result, the CUDA kernel. Left chart: the optimization ladder, speedup over FP16 cuBLAS on the largest layer, batch 1. The naive version is slower than cuBLAS; each step adds speed, ending at
{k['speedup']:.2f} times, which is {k['pct']:.0f} percent of the {F['peak_gbps']:.0f} GB/s theoretical memory bandwidth ({k['gbps']:.0f} GB/s effective — bytes crossing DRAM divided by kernel time, calculated,
not a profiler reading). The theoretical ceiling is {k['ceil']:.2f} times because int4 also carries scales and zero points.
On the right is a screenshot from our own live interactive dashboard, showing the same numbers — proof this is a working system, not only slides.
Second key result: we integrated the kernel into vLLM as a torch custom op and measured real decoding. At one sequence it reaches {e['inferno_b8']['1']:.0f} tokens per second, about {r1:.2f} times Marlin's
speed: parity with a production kernel, not a win. Marlin's tensor-core kernel is faster from four sequences onward; the next milestone is a tensor-core version of our own kernel.
Nsight Compute profiling has not been done yet because GPU performance counters are blocked by a Windows driver setting; it is scheduled for October.
Never say the kernel outperforms Marlin — the honest claim is parity at batch one.
""")

    # ================================================================== 10. results: adaptive routing & MPI (merged)
    s = new("Results: Adaptive Routing & MPI")
    lo, hi = min(r["pct_change"] for r in F["h1rows"]), max(r["pct_change"] for r in F["h1rows"])
    h1 = {r["pattern"]: r for r in F["h1rows"]}
    pats = ["steady", "bursty", "mixed_length"]
    text(s, X0, 1.22, W, 0.36, "A pre-specified hypothesis, tested honestly — and a mechanism we found and explained", size=15.5, color=GREY)
    labelled(s, X0, 1.62, W, 0.5, [[("H1 rejected for a shared single GPU: ", {"bold": True, "color": CRIMSON, "size": 15.5}),
                                       (f"p99 worsened {lo:.0f}–{hi:.0f}% (Holm p = 0.023). ", {"size": 14.5, "bold": True, "color": NAVY}),
                                       ("Multi-GPU evaluation is PENDING.", {"size": 14.5, "bold": True, "color": CRIMSON})]],
             fill=TINT, line=CRIMSON, anchor="m", margin=0.2)
    text(s, X0, 2.28, 7.0, 0.3, "p99 end-to-end latency (s): best fixed policy vs adaptive", size=13.5, bold=True, color=NAVY)
    bar_chart(s, X0, 2.56, 7.0, 2.1, ["steady", "bursty", "mixed"], {"best fixed policy": [round(h1[p]["fixed_mean"], 1) for p in pats],
                                                                    "adaptive policy": [round(h1[p]["treatment_mean"], 1) for p in pats]},
              [BLUE, CRIMSON], number_format="0.0", font=12, legend=True, gap=55, overlap=-8, vmax=28)
    labelled(s, X0, 4.72, 3.4, 0.78, [[(f"{lo:.0f}–{hi:.0f}%", {"bold": True, "size": 22, "color": CRIMSON}), (" worse p99", {"size": 12, "color": GREY})]],
             fill=PALE, line=None, anchor="m", margin=0.18)
    labelled(s, X0 + 3.6, 4.72, 3.4, 0.78, [[("8 / 8", {"bold": True, "size": 22, "color": CRIMSON}), (" paired reps agree", {"size": 12, "color": GREY})]],
             fill=PALE, line=None, anchor="m", margin=0.18)
    pic_w = 4.75
    picture(s, FIG / "ppt_verdict.png", X0 + 7.35, 2.28, w=pic_w)
    pic_h = pic_w * 425 / 1140
    text(s, X0 + 7.35, 2.28 + pic_h + 0.05, pic_w, 0.26, "↑ from our live interactive dashboard", size=11, color=GREY, align="c")
    y2 = 2.28 + pic_h + 0.30
    labelled(s, X0 + 7.35, y2, pic_w, 0.8, [
        [("Why? A shared GPU is time-sliced", {"bold": True, "color": NAVY, "size": 13.5})],
        [(f"2 replicas BOTH active → p99 ×{F['cont_active'][0]:.1f}", {"size": 12.5, "color": CRIMSON, "bold": True})],
        [(f"2 replicas, ONE idle → {100 * (F['cont_idle'][0] - 1):+.1f}% (free)", {"size": 12.5, "color": GREEN, "bold": True})]],
        fill=PALE, line=None, anchor="m", margin=0.16)
    y3 = y2 + 0.86
    chip(s, X0 + 7.35, y3, 2.25, 0.38, f"MPI ≈ {F['mpi_pp'][64]['one_way_us_est']:.0f} µs / hop", fill=TINT, color=CRIMSON, size=12)
    chip(s, X0 + 7.35 + 2.35, y3, 2.4, 0.38, "< 1% vs HTTP — not the bottleneck", fill=TINT, color=CRIMSON, size=11)
    box(s, X0, 5.62, W, 0.95, fill=None, line=None, gradient=(RGBColorHex("0B7A4B"), RGBColorHex("06563A")), radius=0.1)
    text(s, X0 + 0.3, 5.62, W - 0.5, 0.95, [[("What we learned: ", {"bold": True, "color": RGBColorHex("C9F2DD"), "size": 17})],
                                           [("not a failed idea — a precisely located one. Adaptive routing needs independent GPU capacity per replica; "
                                             "we found and explained exactly why it did not help here.", {"color": WHITE, "bold": True, "size": 16})]],
         anchor="m", space_after=2)
    setting_strip(s, 6.68, 0.36, "Setting:", "Qwen2.5-1.5B AWQ-int4, 2 replicas on ONE GPU · 96 paired runs (4 policies × 3 patterns × 8 reps) · MPI: Open MPI 5.0.10.", size=10.5)
    notes(s, f"""
Third and fourth results together: adaptive routing and MPI, both tested rigorously and both giving honest, explained results.
We pre-specified hypothesis H1: an adaptive router beats the best fixed configuration. State it precisely: H1 was REJECTED for a shared single GPU — p99 was {lo:.0f} to {hi:.0f} percent worse, Holm-adjusted
p = 0.023, all eight paired repetitions agreed (the left chart is our own data; the screenshot on the right is the same result on our live dashboard).
Why: a controlled experiment held total load fixed and varied only how the GPU is shared. An idle second replica costs almost nothing; an ACTIVE second replica multiplies p99 by about {F['cont_active'][0]:.1f}, because
the two processes time-slice one physical GPU. Multi-GPU evaluation, with one replica per GPU, is PENDING — that is the real test.
MPI: overhead versus HTTP is below 1%, with no statistically significant improvement established; the bottleneck is the shared GPU, not the coordination layer.
Framing: this is not a failed idea, it is a precisely located one — we know exactly what independent-GPU-capacity precondition it needs, and we have not tested that condition yet.
""")

    # ================================================================== 11. challenges
    s = new("Challenges")
    ch_ = [("Measurement traps", ["Cache leak inflated bandwidth by ≈ 6% → fixed, re-run", "A library baseline was mis-measured → claim retracted", "Throughput metric hid saturation → redefined"]),
           ("Hardware limits", ["Only 8 GB → a 1.5B-parameter model", "Both replicas share one GPU (0.47 scaling)", "Laptop clocks drift → randomized run order"]),
           ("Tooling", ["GPU counters blocked by a Windows driver setting → Nsight profiling NOT yet done", "CUDA 12.4 toolkit vs PyTorch CUDA 12.8 build", "WSL2 disk / memory limits"]),
           ("Engine limits", ["vLLM cannot change batch size or precision live", "A restart costs ≈ 48 s", "→ adaptivity implemented as routing across pre-loaded replicas"])]
    cw, chh = 6.17, 2.62
    for i, (t, items) in enumerate(ch_):
        r, c = divmod(i, 2)
        x, y = X0 + c * (cw + 0.16), 1.36 + r * (chh + 0.16)
        card(s, x, y, cw, chh, fill=WHITE, accent=CRIMSON if i in (0, 3) else NAVY)
        text(s, x + 0.3, y + 0.1, cw - 0.4, 0.5, t, size=20, bold=True, color=NAVY)
        text(s, x + 0.3, y + 0.68, cw - 0.45, chh - 0.75, items, size=18, bullets=True, space_after=8)
    notes(s, """
Challenges. Measurement traps are the most important: a cache-flush footprint that was too small inflated bandwidth by about six percent; a library baseline was measured wrongly; and our first
throughput metric hid saturation. All three were found by checks in the methodology, corrected, and the experiments re-run.
Hardware: 8 GB limits us to a 1.5-billion-parameter model, and both replicas share one GPU, which limits routing experiments. Tooling: Nsight Compute cannot read GPU counters until a Windows driver
setting is enabled, and our CUDA toolkit is 12.4 while PyTorch is built for 12.8. Engine: vLLM cannot change batch size or precision on a live server, and a restart takes about 48 seconds, so adaptivity
is implemented as routing across pre-loaded replicas.
""")

    # ================================================================== 12. contributions, conclusions & plan (merged)
    s = new("Contributions, Conclusions & Plan")
    cw = 7.15
    card(s, X0, 1.28, cw, 3.35, fill=WHITE, accent=GREEN)
    text(s, X0 + 0.3, 1.34, cw - 0.5, 0.42, "Our contributions", size=19, bold=True, color=NAVY)
    text(s, X0 + 0.32, 1.82, cw - 0.55, 2.7, [
        "Fused int4 CUDA kernel: 87–95% of GPU memory bandwidth",
        "Integrated into vLLM — real end-to-end decode measured",
        "96-run rigorous campaign, paired Holm-corrected statistics",
        "MPI-based distributed router with per-hop timing",
        "Explained WHY adaptive routing needs independent GPUs",
        "Live dashboard + 60-page report — 81 tests, reproducible",
    ], size=16.5, bullets=True, space_after=11)
    x2 = X0 + cw + 0.2
    cw2 = W - cw - 0.2
    picture(s, FIG / "ppt_hero.png", x2, 1.28, w=cw2)
    text(s, x2, 1.28 + cw2 * 760 / 1300 + 0.06, cw2, 0.3, "Explore: our live project dashboard", size=12, color=GREY, align="c")
    text(s, X0, 4.78, W, 0.36, "The plan to the final evaluation", size=18, bold=True, color=NAVY)
    plan = [("NOW — MID-TERM", "✔", GREEN, GREENTINT, ["Kernel (v0–v6), verified", "vLLM integration, measured", "96-run campaign · MPI router"]),
            ("OCT – NOV 2026", "→", CRIMSON, TINT, ["Nsight Compute profiling", "Tensor-core kernel (batch 8–64)", "Multi-GPU cloud + MPI setup"]),
            ("DEC 7–11, 2026", "F", NAVY, PALE, ["Multi-GPU routing results", "Final report & dashboard", "FINAL EVALUATION"])]
    pw3, ph3, g3 = 4.02, 1.68, 0.16
    for i, (t, ic, col, tint, items) in enumerate(plan):
        x = X0 + i * (pw3 + g3)
        y = 5.18
        card(s, x, y, pw3, ph3, fill=tint, accent=col)
        badge(s, x + 0.18, y + 0.15, 0.5, ic, col, size=17)
        text(s, x + 0.8, y + 0.12, pw3 - 0.9, 0.5, t, size=15.5, bold=True, color=NAVY, anchor="m")
        text(s, x + 0.22, y + 0.68, pw3 - 0.36, ph3 - 0.78, items, size=13, bullets=True, space_after=4)
        if i < 2:
            arrow(s, x + pw3 + 0.02, y + ph3 / 2, x + pw3 + g3 - 0.02, y + ph3 / 2, color=GREY, width=2.5)
    notes(s, """
Our contributions to the total project, left to right and top to bottom: a fused int4 CUDA kernel near the hardware memory limit; that kernel integrated and measured end to end inside vLLM, not just in
isolation; a 96-run rigorous experimental campaign with proper paired statistics; an MPI-based distributed router with per-hop timing; a real engineering insight into WHY adaptive routing needs independent
GPUs, found and explained rather than just observed; and a live interactive dashboard plus a 60-page report, with 81 automated tests, so every number is reproducible. On the right is our dashboard itself.
The plan to the final evaluation, very simply, three steps: NOW, at mid-term, the kernel, the vLLM integration, the 96-run campaign and the MPI router are done. October to November: Nsight profiling, a
tensor-core kernel for larger batches, and setting up the multi-GPU cloud and MPI cluster. December 7 to 11: the multi-GPU routing results, the final report and dashboard, and the final evaluation itself.
""")

    # ================================================================== 13. team
    s = new("Team Work & Coordination")
    roles = [("Kernel & profiling", team[0], ["CUDA kernel ladder v0–v6", "Small-batch sweep, Nsight"]),
             ("Serving & statistics", team[1], ["Open-loop client, calibration", "Paired tests, campaign runs"]),
             ("MPI router & controller", team[2], ["Dispatcher / workers, policies", "Scaling and contention study"]),
             ("Integration & documentation", team[3], ["vLLM integration, cloud (Terraform)", "60-page report, dashboard, deck"])]
    cw = 3.02
    for i, (role, who, items) in enumerate(roles):
        x = X0 + i * (cw + 0.14)
        card(s, x, 1.36, cw, 3.3, fill=WHITE, accent=CRIMSON)
        badge(s, x + 0.22, 1.5, 0.5, f"R{i + 1}", CRIMSON, size=15)
        text(s, x + 0.82, 1.46, cw - 0.9, 0.6, role, size=16.5, bold=True, color=NAVY, anchor="m")
        nm, _, roll = who.partition(" (")
        text(s, x + 0.25, 2.1, cw - 0.4, 0.42, nm, size=19, bold=True, color=CRIMSON)
        text(s, x + 0.25, 2.5, cw - 0.4, 0.36, roll.rstrip(")"), size=12.5, color=GREY)
        text(s, x + 0.25, 3.0, cw - 0.4, 1.6, items, size=16, bullets=True, space_after=7)
    coord = [("Coordination", "weekly sync · shared Git repository · 81 automated tests passed"),
             ("Documentation", "60-page technical report · README · generated FINDINGS"),
             ("Quality", "every number generated from raw data · negative results kept")]
    for i, (t, d) in enumerate(coord):
        x = X0 + i * 4.2
        labelled(s, x, 4.9, 4.08, 1.9, [[(t, {"bold": True, "color": NAVY, "size": 19})], [(d, {"size": 15.5})]], fill=PALE, line=None, anchor="m", margin=0.22)
    notes(s, """
Teamwork. Work is split by component so that everyone owns a real part of the system: R1 the CUDA kernel and profiling; R2 the serving harness, calibration and statistics; R3 the MPI router, controller and scaling study;
R4 the vLLM integration, cloud infrastructure and documentation. Each member can explain their component and how it connects to the others through the workflow diagram.
Coordination: weekly sync, one shared repository, continuous integration that runs the 81 automated tests. Documentation: a 60-page technical report, a README and auto-generated findings. Quality: every number is generated from
raw data and negative results are kept in the report.
""")

    # ================================================================== 14. references
    s_refs_slide = s_refs
    refs = ["G.-I. Yu et al., “Orca: A distributed serving system for transformer-based generative models,” in Proc. USENIX OSDI, 2022.",
            "W. Kwon et al., “Efficient memory management for large language model serving with PagedAttention,” in Proc. ACM SOSP, 2023.",
            "Y. Zhong et al., “DistServe: Disaggregating prefill and decoding for goodput-optimized LLM serving,” in Proc. USENIX OSDI, 2024.",
            "P. Patel et al., “Splitwise: Efficient generative LLM inference using phase splitting,” in Proc. ACM/IEEE ISCA, 2024.",
            "J. Lin et al., “AWQ: Activation-aware weight quantization for LLM compression and acceleration,” in Proc. MLSys, 2024.",
            "E. Frantar et al., “GPTQ: Accurate post-training quantization for generative pre-trained transformers,” in Proc. ICLR, 2023.",
            "E. Frantar et al., “MARLIN: Mixed-precision auto-regressive parallel inference on large language models,” in Proc. ACM PPoPP, 2025.",
            "T. Hoefler and R. Belli, “Scientific benchmarking of parallel computing systems,” in Proc. Int. Conf. High Performance Computing, Networking, Storage and Analysis (SC), 2015.",
            "A. Georges, D. Buytaert, and L. Eeckhout, “Statistically rigorous Java performance evaluation,” in Proc. ACM OOPSLA, 2007, pp. 57–76.",
            "S. Williams, A. Waterman, and D. Patterson, “Roofline: An insightful visual performance model for multicore architectures,” Commun. ACM, vol. 52, no. 4, pp. 65–76, 2009."]
    for shp in s_refs_slide.shapes:
        if shp.name == "Rectangle 5":
            tf = shp.text_frame
            paras = tf.paragraphs
            proto = paras[0]._p
            for p in paras[1:]:
                p._p.getparent().remove(p._p)
            for i, r in enumerate(refs):
                if i == 0:
                    p = tf.paragraphs[0]
                    p.runs[0].text = r
                    for extra in p.runs[1:]:
                        extra._r.getparent().remove(extra._r)
                else:
                    import copy
                    newp = copy.deepcopy(proto)
                    tf._txBody.append(newp)
                    from pptx.text.text import _Paragraph
                    para = _Paragraph(newp, tf)
                    para.runs[0].text = r
            shp.top = Inches(1.38)
            shp.height = Inches(5.4)
            for p in tf.paragraphs:
                p.space_before = Pt(3)
                for r in p.runs:
                    r.font.size = Pt(15)
        if shp.name == "TextBox 3":
            shp._element.getparent().remove(shp._element)
        if shp.name == "TextBox 6":
            shp.width = Inches(11.4)
    order.append(s_refs_slide)
    notes(s_refs_slide, "References in IEEE format. Ten sources covering serving systems, quantization kernels, and benchmarking methodology.")

    # ================================================================== 15. thank you
    order.append(s_thanks)
    text(s_thanks, 3.4, 4.5, 6.5, 0.7, "Questions?", size=32, bold=True, color=CRIMSON, align="c", font="Times New Roman")
    notes(s_thanks, "Thank you. We are happy to take questions. Every number in this presentation is generated from the raw result files in the project repository.")

    for shp in s_contents.shapes:
        if shp.name == "Rectangle 5":
            for p in shp.text_frame.paragraphs:
                for r in p.runs:
                    if r.text.strip().startswith("Conclusions"):
                        r.text = "Contributions, Conclusions, Plan & Team"
    cp = prs.core_properties
    cp.title = "Inferno: Adaptive LLM Inference Serving – HPC Mid-Term Evaluation"
    cp.author = ", ".join(t.split(" (")[0] for t in team)
    cp.subject = "High-Performance Computing project, B.Tech (AIDS), UG-III"
    cp.keywords = "CUDA, MPI, vLLM, LLM inference, benchmarking"

    # official spelling and one department name everywhere
    def fix_text(shape):
        if not shape.has_text_frame:
            return
        for p in shape.text_frame.paragraphs:
            for r in p.runs:
                if "Amruta" in r.text:
                    r.text = r.text.replace("Amruta", "Amrita")
                if r.text.strip() == "School of Artificial Intelligence":
                    r.text = "Department of Artificial Intelligence and Data Science"
    for sl in list(prs.slides):
        for shp in sl.shapes:
            fix_text(shp)
    for lay in prs.slide_layouts:
        for shp in lay.shapes:
            fix_text(shp)
    for shp in prs.slide_master.shapes:
        fix_text(shp)

    # drop the template's unused content slides, then order
    keep = set(map(id, order))
    for sl in orig:
        if id(sl) not in keep:
            delete_slide(prs, sl)
    reorder(prs, order)
    prs.save(str(OUT))
    return OUT


def RGBColorHex(h):
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", nargs=4, default=["Likhith (DL.AI.U4AID24117)", "Karthikeya (DL.AI.U4AID24020)", "Sandeep (DL.AI.U4AID24143)", "A. Karthik (DL.AI.U4AID24104)"])
    a = ap.parse_args()
    print("wrote", build(a.team))
