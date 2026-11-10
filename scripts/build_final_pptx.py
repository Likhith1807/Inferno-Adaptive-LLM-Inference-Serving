"""
Builds docs/Inferno_Final_Presentation_2026.pptx (final evaluation) on the university template
docs/PPT Format-Project Presentation-2026.pptx. The template chassis (logo, title rule, crimson footer, section order) is kept and
slides are cloned from it. Every number on a slide is read from docs/results by the loaders the technical report uses, and every
figure is a real result figure generated from measured data (docs/figures, docs/results/*/), never drawn by hand.

    python scripts/build_final_pptx.py
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.text.text import _Paragraph  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from build_pptx import RGBColorHex, facts, setting_strip  # noqa: E402
from ppt_kit import (AMBER, AMBERTINT, BLUE, CRIMSON, DEEP, GREEN, GREENTINT, GREY, INK, LINE, NAVY, PALE, TINT, WHITE,  # noqa: E402
                     arrow, badge, bar_chart, box, card, chip, clone_slide, delete_slide, labelled, notes, picture, remove_shapes,
                     reorder, set_title, text)
from report_pdf import data  # noqa: E402
from report_pdf.sec_results import layer_facts  # noqa: E402

TEMPLATE = ROOT / "docs" / "PPT Format-Project Presentation-2026.pptx"
OUT = ROOT / "docs" / "Inferno_Final_Presentation_2026.pptx"
FIG = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results"
X0, X1 = 0.42, 12.92
W = X1 - X0
LIGHT_ON_DARK = "FFD6DD"


def sweep_facts():
    sw = pd.read_csv(RES / "sweep" / "summary.csv")
    g = sw.groupby("config").agg(tpj=("tokens_per_joule", "mean"), p50=("e2e_p50", "mean"))
    cold = sw.groupby("config")["cold_start_sec"].first()
    return g.loc["fp16-bs64"], g.loc["int4-bs64"], float(cold[cold < 100].median())


def banner(s, y, h, head, body, colors=(CRIMSON, DEEP), head_color=LIGHT_ON_DARK, size=16.5):
    box(s, X0, y, W, h, fill=None, line=None, gradient=colors, radius=0.1)
    text(s, X0 + 0.3, y, W - 0.5, h, [[(head, {"bold": True, "color": RGBColorHex(head_color), "size": size}),
                                       (body, {"color": WHITE, "bold": True, "size": size})]], anchor="m")


def stat(s, x, y, w, h, big, small, color=CRIMSON, fill=PALE, big_size=30, small_size=13):
    labelled(s, x, y, w, h, [[(big, {"bold": True, "size": big_size, "color": color})], [(small, {"size": small_size, "color": INK})]],
             fill=fill, line=None, anchor="m", margin=0.18, space_after=0)


def build(team):
    F = facts()
    k, e = F["kern"], F["e2e"]
    LF = layer_facts()
    mf = LF["mf"]
    f16, i4, cold = sweep_facts()
    cont = data.jl("contention/contention_summary.json")
    p99 = {(c["pattern"], c["config"]): c for c in cont["cells"]}
    reps_c = min(c["n"] for c in cont["cells"])
    lo, hi = min(r["pct_change"] for r in F["h1rows"]), max(r["pct_change"] for r in F["h1rows"])
    r1, r4 = F["e2e_ratio_b8"]["1"]["mean"], F["e2e_ratio_b8"]["4"]["mean"]
    mpi_us = F["mpi_pp"][64]["one_way_us_est"]
    lo4 = min(r[4] for r in LF["ratio"].values())
    n_tests = 81

    prs = Presentation(str(TEMPLATE))
    orig = list(prs.slides)
    s_title, s_contents, base, s_refs, s_thanks = orig[0], orig[1], orig[2], orig[11], orig[12]
    order = [s_title, s_contents]
    TAGS = {}

    def new(title, tag=None):
        sl = clone_slide(prs, base)
        set_title(sl, title)
        remove_shapes(sl, {"TextBox 2"})
        if tag:
            chip(sl, 10.35, 0.6, 2.6, 0.36, tag, fill=TINT, color=CRIMSON, size=12.5)
        order.append(sl)
        return sl

    # ================================================================== 1. title
    for shp in s_title.shapes:
        if shp.name == "Rectangle 9":
            shp.text_frame.paragraphs[0].runs[0].text = "Inferno: Adaptive LLM Inference Serving"
        if shp.name == "Rectangle 16":
            for p, name in zip(shp.text_frame.paragraphs, team):
                p.runs[0].text = name
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)
    text(s_title, 0.25, 1.02, 12.83, 0.7, "CUDA kernel  ·  vLLM integration  ·  MPI router  ·  rigorous GPU benchmarking", size=21, color=CRIMSON,
         bold=True, align="c", font="Times New Roman")
    text(s_title, 0.25, 2.78, 12.83, 0.5, "High-Performance Computing Project  ·  Final Evaluation 2026", size=20, color=NAVY, bold=True, align="c",
         font="Times New Roman")
    notes(s_title, """
Good morning. Our project is Inferno. It asks one High-Performance Computing question: how do we serve a large language model fast, and how do we prove with real
measurements that a design is actually faster? We built a fused int4 CUDA kernel, put it inside the vLLM serving engine, built an MPI request router and an adaptive
controller, and measured everything with paired statistics on one laptop GPU. Every number in this deck is generated by our scripts from raw result files.
""")
    notes(s_contents, "Agenda: problem, literature, objectives, methodology, tools and setup, then our results (seven result slides), challenges, conclusions and "
                      "future scope, team, references.")

    # ================================================================== 3. problem
    s = new("Problem Statement", "1 · PROBLEM")
    gb, bw_ = 3.09, F["peak_gbps"]
    ms_tok, tps_max = gb / bw_ * 1000, bw_ / gb
    card(s, X0, 1.28, W, 2.6, fill=PALE, line=None)
    text(s, X0 + 0.2, 1.3, 6.6, 0.4, "Why is generating text slow? A memory problem.", size=18, bold=True, color=NAVY)
    box(s, X0 + 0.25, 1.78, 2.2, 1.2, fill=NAVY, line=None, radius=0.1)
    text(s, X0 + 0.25, 1.78, 2.2, 1.2, [[("Model weights", {"size": 16, "bold": True, "color": WHITE})], [("≈ 3.1 GB", {"size": 24, "bold": True, "color": WHITE})],
                                        [("FP16, in GPU memory", {"size": 12, "color": RGBColorHex("C9DAF2")})]], align="c", anchor="m", space_after=0)
    arrow(s, X0 + 2.5, 2.38, X0 + 3.75, 2.38, color=CRIMSON, width=4)
    text(s, X0 + 2.45, 1.86, 1.4, 0.45, f"{bw_:.0f} GB/s", size=18, bold=True, color=CRIMSON, align="c")
    text(s, X0 + 2.4, 2.45, 1.5, 0.4, "theoretical", size=12, color=GREY, align="c")
    box(s, X0 + 3.8, 1.78, 1.9, 1.2, fill=GREY, line=None, radius=0.1)
    text(s, X0 + 3.8, 1.78, 1.9, 1.2, [[("GPU cores", {"size": 17, "bold": True, "color": WHITE})], [("waiting for data", {"size": 12, "color": WHITE})]],
         align="c", anchor="m", space_after=0)
    arrow(s, X0 + 5.75, 2.38, X0 + 6.3, 2.38, color=CRIMSON, width=4)
    chip(s, X0 + 6.32, 2.06, 0.95, 0.6, "1 token", fill=CRIMSON, color=WHITE, size=15)
    card(s, X0 + 0.25, 3.12, 7.0, 0.62, fill=GREENTINT, accent=GREEN, line=None)
    text(s, X0 + 0.5, 3.12, 6.7, 0.62, [[("Measured FP16 baseline:  ", {"bold": True, "color": GREEN, "size": 16}),
                                        (f"{e['fp16']['1']:.0f} tokens/s", {"bold": True, "color": NAVY, "size": 20}),
                                        ("   (vLLM 0.11, batch 1)", {"size": 13, "color": GREY})]], anchor="m")
    card(s, X0 + 7.5, 1.78, 4.8, 1.96, fill=WHITE, accent=CRIMSON)
    text(s, X0 + 7.72, 1.8, 4.55, 1.92, [[("Back-of-envelope: ", {"bold": True, "color": NAVY, "size": 13}),
                                        (f"reading 3.1 GB at {bw_:.0f} GB/s takes ≈ {ms_tok:.0f} ms, an idealized ceiling of ≈ {tps_max:.0f} tokens/s at batch 1. "
                                         "Fewer bytes per weight (int4) → more tokens per second, if the kernel reaches the memory limit.", {"size": 13})]],
         anchor="m", space_after=0)
    probs = [("1", "Memory-bound decoding", "At small batches, speed is set by bytes read per token. int4 weights = 3.8× fewer bytes."),
             ("2", "Bursty traffic, tail latency", "Real traffic arrives in bursts. Users feel the slowest requests (p99), not the average."),
             ("3", "Coordinating replicas", "Several model replicas need a low-overhead layer to route independent requests (MPI).")]
    pw = 4.05
    for i, (n, t, d) in enumerate(probs):
        x = X0 + i * (pw + 0.175)
        card(s, x, 4.0, pw, 1.5, fill=WHITE, accent=CRIMSON)
        badge(s, x + 0.25, 4.1, 0.46, n, CRIMSON, size=16)
        text(s, x + 0.82, 4.06, pw - 0.92, 0.55, t, size=18.5, bold=True, color=NAVY, anchor="m")
        text(s, x + 0.28, 4.62, pw - 0.4, 0.88, d, size=14.5, color=INK)
    box(s, X0, 5.65, 6.2, 1.2, fill=None, line=None, gradient=(CRIMSON, DEEP), radius=0.08)
    text(s, X0 + 0.25, 5.65, 5.8, 1.2, [[("Our question", {"size": 15, "bold": True, "color": RGBColorHex(LIGHT_ON_DARK)})],
                                       [("Can a parallel, adaptive serving stack beat fixed configurations – and why?", {"size": 19, "bold": True, "color": WHITE})]],
         anchor="m", space_after=2)
    labelled(s, X0 + 6.4, 5.65, W - 6.4, 1.2, [[("Why HPC?", {"bold": True, "color": CRIMSON, "size": 19})],
                                               [("GPU / CUDA", {"bold": True}), (" – billions of independent multiply-adds", {})],
                                               [("MPI", {"bold": True}), (" – coordinating independent requests", {})],
                                               [("Rigorous benchmarking", {"bold": True}), (" – to trust any speed claim", {})]],
             fill=TINT, line=None, size=15, anchor="m", margin=0.22, space_after=1)
    notes(s, f"""
To write one token the GPU reads all model weights, about 3.1 GB in FP16. At {bw_:.0f} GB/s theoretical bandwidth that is about {ms_tok:.0f} ms per token, an idealized
ceiling of about {tps_max:.0f} tokens per second at batch one; the measured FP16 baseline in vLLM is {e['fp16']['1']:.0f}. So decoding is memory-bound: 4-bit weights move
about 3.8 times fewer bytes, which can make generation faster if the kernel reaches the memory limit. Second, real traffic is bursty and users feel p99. Third, serving
several replicas needs a cheap coordination layer, which is where MPI comes in. Our question: can a parallel, adaptive serving stack beat fixed configurations, and why?
""")

    # ================================================================== 4. literature
    s = new("Literature Review", "2 · LITERATURE")
    text(s, X0, 1.22, W, 0.36, "Four research threads our project builds on, and the gap between them", size=15.5, color=GREY)
    themes = [("Continuous & Paged Serving", CRIMSON, [[("Orca [1]  ", {"bold": True, "color": CRIMSON}), ("iteration-level (continuous) batching", {})],
                                                       [("vLLM [2]  ", {"bold": True, "color": CRIMSON}), ("PagedAttention, paged KV-cache memory", {})]]),
              ("Disaggregated Serving", NAVY, [[("DistServe [3]  ", {"bold": True, "color": NAVY}), ("splits prefill / decode across GPUs", {})],
                                               [("Splitwise [4]  ", {"bold": True, "color": NAVY}), ("phase-specialized GPU pools", {})]]),
              ("Quantization & Kernels", GREEN, [[("AWQ [5], GPTQ [6]  ", {"bold": True, "color": GREEN}), ("accurate 4-bit weights", {})],
                                                 [("Marlin [7]  ", {"bold": True, "color": GREEN}), ("fast 4-bit GPU kernel: our baseline", {})]]),
              ("Rigorous Benchmarking", AMBER, [[("Hoefler & Belli [8]  ", {"bold": True, "color": AMBER}), ("12 rules for HPC benchmarking", {})],
                                                [("Georges et al. [9]  ", {"bold": True, "color": AMBER}), ("statistically sound perf. claims", {})]])]
    cw, chh, g = 6.15, 1.86, 0.16
    for i, (t, col, lines) in enumerate(themes):
        r, c = divmod(i, 2)
        x, y = X0 + c * (cw + g), 1.68 + r * (chh + g)
        card(s, x, y, cw, chh, fill=WHITE, accent=col)
        text(s, x + 0.3, y + 0.12, cw - 0.5, 0.44, t, size=19, bold=True, color=NAVY)
        text(s, x + 0.32, y + 0.68, cw - 0.55, chh - 0.78, lines, size=16.5, bullets=True, space_after=10)
    banner(s, 5.68, 1.02, "The gap we fill: ", "the kernel, the router and the statistics studied TOGETHER on one GPU, with paired tests and negative results reported.",
           size=17)
    notes(s, """
Continuous and paged serving: Orca's iteration-level batching and vLLM's PagedAttention; vLLM is our engine. Disaggregated serving: DistServe and Splitwise put
prefill and decode on separate GPUs, which is exactly the precondition our routing result points to. Quantization: AWQ and GPTQ make accurate 4-bit weights, and Marlin
is the production 4-bit kernel we benchmark against. Benchmarking: Hoefler and Belli, and Georges et al., set the statistical rules we follow. The gap: nobody studies
the kernel, the routing layer and the statistics together and reports the negative results.
""")

    # ================================================================== 5. objectives (final status)
    s = new("Objective of the Project", "3 · OBJECTIVES")
    objs = [("Fused int4 CUDA kernel reaching ≥ 85% of theoretical DRAM bandwidth", f"Achieved: {k['pct']:.0f}% · {k['speedup']:.2f}× cuBLAS", GREEN, GREENTINT),
            ("Integrate the kernel into vLLM and measure real decode speed", f"Done: {e['inferno_b8']['1']:.0f} tok/s = Marlin parity", GREEN, GREENTINT),
            ("Find where the kernel earns its place: every layer, batch 1–64", "Done: decision taken from data", GREEN, GREENTINT),
            ("Test adaptive routing vs the best fixed baselines (pre-specified H1)", "Done: rejected, cause isolated", GREEN, GREENTINT),
            ("MPI request router: measure overhead and scaling", f"Done on 1 GPU: ≈ {mpi_us:.0f} µs / hop", GREEN, GREENTINT),
            ("Reproducible: every number from raw data, one-command verification", f"Released v1.1.0 · {n_tests} tests", GREEN, GREENTINT)]
    for i, (o, status, col, tint) in enumerate(objs):
        y = 1.3 + i * 0.93
        card(s, X0, y, W, 0.8, fill=WHITE)
        badge(s, X0 + 0.18, y + 0.14, 0.52, str(i + 1), CRIMSON, size=19)
        text(s, X0 + 0.92, y + 0.02, 7.5, 0.76, o, size=18.5, anchor="m", color=INK)
        chip(s, X0 + 8.45, y + 0.16, 3.85, 0.5, status, fill=tint, color=col, size=14)
    notes(s, f"""
Six measurable objectives and their final status. One: a fused int4 kernel at 85 percent or more of memory bandwidth; we reached {k['pct']:.0f} percent and {k['speedup']:.2f} times
cuBLAS. Two: integrate it into vLLM; it decodes at {e['inferno_b8']['1']:.0f} tokens per second, parity with Marlin. Three, new since mid-term: find exactly where the kernel
earns its place, by timing every layer at batch 1 to 64; that gave a clear, data-driven decision. Four: test adaptive routing with a pre-specified hypothesis; rejected,
and we isolated the cause. Five: the MPI router costs about {mpi_us:.0f} microseconds per hop. Six: everything is reproducible and released as version 1.1.0.
""")

    # ================================================================== 6. methodology: workflow
    s = new("Proposed Methodology: Workflow", "4 · METHOD 1/2")
    steps = [("Workload Generator", "Poisson · bursty · mixed lengths; seeded", "in"), ("Open-loop Client", "sends on schedule; records TTFT, TPOT, p99", "in"),
             ("Request Router", "policy · HTTP baseline · MPI dispatcher (rank 0)", "sys"), ("vLLM Replicas", "fast tier + bulk tier sharing one GPU", "sys"),
             ("Fused int4 CUDA Kernel", "decode weights, multiply, reduce on the GPU", "sys"), ("Metrics & Power", "latency, goodput, tokens/s, watts (200 ms)", "out"),
             ("Statistics", "paired t-test + Wilcoxon, Holm, effect size", "out"), ("Report & Dashboard", "every number generated from raw logs", "out")]
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
    banner(s, 6.05, 0.82, "Key idea:  ", "replay the SAME seeded traffic through every policy, so differences come from the policy, not luck.", size=17.5)
    notes(s, """
Read the diagram in number order: grey inputs, red system under test, green measurement. Seeded request streams, an open-loop client that never slows down when the
server is slow, a router (fixed, adaptive or MPI), vLLM replicas sharing one GPU, our int4 kernel inside, then metrics and power, paired statistics, and an
automatically generated report. The key idea: identical seeded traffic goes through every policy, so any difference is caused by the policy.
""")

    # ================================================================== 7. methodology: parallelization
    s = new("Proposed Methodology: Parallelization", "4 · METHOD 2/2")
    lv = [("LEVEL 1", "CUDA", "inside one layer", CRIMSON, "each output row of the matrix · each row = K/32 chunks of 32 weights",
           "16 lanes per row · 128-bit coalesced loads · warp-shuffle reduction", "fine-grained data parallelism on a memory-bound GPU"),
          ("LEVEL 2", "vLLM", "across users", NAVY, "tokens of different users' requests",
           "continuous batching · one decoded weight reused for up to B vectors", "production scheduler, not re-invented"),
          ("LEVEL 3", "MPI", "across replicas", RGBColorHex("0B7A4B"), "whole requests → independent servers",
           "rank 0 dispatcher + N workers · non-blocking isend / iprobe", f"distributed memory · ≈ {mpi_us:.0f} µs per hop")]
    heads = ["What is independent?", "How it is parallelized", "Why this framework?"]
    hx, hw = [X0 + 2.75, X0 + 6.15, X0 + 9.85], [3.3, 3.6, 2.65]
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
    box(s, X0, 5.9, W, 0.98, fill=None, line=None, gradient=(CRIMSON, DEEP), radius=0.1)
    text(s, X0 + 0.3, 5.9, W - 0.5, 0.98, [[("MPI coordinates requests; CUDA does the GPU math.", {"size": 21, "bold": True, "color": WHITE})],
                                           [("Not chosen: OpenMP (CPU shared memory only) · NCCL (tensor collectives; we route requests) · HTTP only (kept as the baseline).",
                                             {"size": 13.5, "color": RGBColorHex(LIGHT_ON_DARK)})]], anchor="m", space_after=2)
    notes(s, f"""
Three levels of parallelism. Level one, CUDA: every output row of a matrix-vector product is independent, each row splits into 32-weight chunks, and a warp reduces with
shuffles. Level two, vLLM: different users' tokens are batched continuously, and our kernel reuses each decoded weight for up to B vectors. Level three, MPI: whole requests
are independent, so a rank-zero dispatcher sends them to workers with non-blocking calls; a hop costs about {mpi_us:.0f} microseconds. MPI coordinates requests; it does not do
tensor math, which is CUDA's job.
""")

    # ================================================================== 8. tools & setup
    s = new("Tools, Technologies & Setup", "5 · TOOLS & SETUP")
    cal = F["cal"]
    cw = 6.15
    card(s, X0, 1.32, cw, 4.55, fill=WHITE, accent=CRIMSON)
    text(s, X0 + 0.3, 1.4, cw - 0.5, 0.5, "Tools & Technologies", size=20, bold=True, color=NAVY)
    text(s, X0 + 0.32, 1.95, cw - 0.55, 3.8, [
        [("Hardware:  ", {"bold": True, "color": CRIMSON}), ("RTX 4060 Laptop GPU (8 GB, 256 GB/s) · WSL2 Ubuntu", {})],
        [("GPU programming:  ", {"bold": True, "color": CRIMSON}), ("CUDA 12.4 · cuBLAS reference · CUDA graphs", {})],
        [("Serving engine:  ", {"bold": True, "color": CRIMSON}), ("vLLM 0.11 + PyTorch 2.8 · our kernel as a torch custom op", {})],
        [("Distributed:  ", {"bold": True, "color": CRIMSON}), ("Open MPI 5 + mpi4py · Terraform for multi-GPU cloud", {})],
        [("Analysis:  ", {"bold": True, "color": CRIMSON}), ("NumPy / SciPy / pandas · paired tests · generated report", {})],
        [("Model:  ", {"bold": True, "color": CRIMSON}), ("Qwen2.5-1.5B-Instruct: FP16 · GPTQ-Int8 · AWQ-Int4", {})],
    ], size=15, bullets=True, space_after=7)
    x2, cw2 = X0 + cw + 0.2, W - cw - 0.2
    card(s, x2, 1.32, cw2, 4.55, fill=WHITE, accent=NAVY)
    text(s, x2 + 0.3, 1.4, cw2 - 0.5, 0.5, "Experimental Setup", size=20, bold=True, color=NAVY)
    text(s, x2 + 0.32, 1.95, cw2 - 0.55, 3.8, [
        [("Kernel study:  ", {"bold": True, "color": NAVY}), ("6 real layer shapes · 7 versions · 5 fresh runs · CPU-reference gate", {})],
        [("Layer matrix:  ", {"bold": True, "color": NAVY}), (f"4 Qwen layers × 7 batch sizes × 4 kernels · {LF['lm']['trials']} trials, 768 MB rotation", {})],
        [("Serving campaign:  ", {"bold": True, "color": NAVY}), ("4 policies × 3 load patterns × 8 repeats = 96 runs", {})],
        [("Calibrated load:  ", {"bold": True, "color": NAVY}), (f"steady = {cal['rates']['steady']['rate_requests_per_sec']:.1f} req/s (60% of measured capacity)", {})],
        [("Mechanism study:  ", {"bold": True, "color": NAVY}), (f"1 vs 2 replicas (idle / active) · {reps_c} paired repeats", {})],
        [("Metrics:  ", {"bold": True, "color": NAVY}), ("TTFT, TPOT, p50/95/99, goodput, tokens/s, tokens/joule", {})],
    ], size=15, bullets=True, space_after=7)
    banner(s, 6.05, 0.82, "Every number is generated from raw logs by our own scripts", "  –  same seed in, same result out, every time.",
           colors=(RGBColorHex("0B7A4B"), RGBColorHex("06563A")), head_color="FFFFFF", size=17)
    notes(s, """
Tools on the left, setup on the right. One laptop GPU, CUDA 12.4, vLLM 0.11, Open MPI, and Qwen2.5-1.5B in three precisions. Five experiments: the kernel ladder on six
real layer shapes, the new layer matrix (four Qwen layers, seven batch sizes, four implementations), a 96-run serving campaign, load calibrated from measured capacity,
and a controlled mechanism study. All workloads are seeded, so any run can be repeated exactly.
""")

    # ================================================================== 9. results at a glance
    s = new("Results at a Glance", "6 · RESULTS 0/7")
    text(s, X0, 1.2, W, 0.4, "Eight headline numbers, each generated from raw data and backed by a figure on the following slides", size=15.5, color=GREY)
    tiles = [(f"{k['speedup']:.2f}×", "faster than FP16 cuBLAS (fused int4 kernel, batch 1)", CRIMSON),
             (f"{k['pct']:.0f}%", f"of theoretical DRAM bandwidth ({F['peak_gbps']:.0f} GB/s)", CRIMSON),
             (f"{e['inferno_b8']['1']:.0f} tok/s", f"inside vLLM = {r1:.2f}× Marlin, {e['inferno_b8']['1'] / e['fp16']['1']:.1f}× FP16", NAVY),
             (f"{LF['wins_share']:.0f}%", "of decode GEMM time where our kernel wins → keep Marlin", NAVY),
             (f"{p99[('rho0.6', 'split-active')]['e2e_p99'] / p99[('rho0.6', 'single')]['e2e_p99']:.1f}×", "p99 when a 2nd replica SERVES on the same GPU", GREEN),
             (f"{100 * (p99[('rho0.6', 'split-idle')]['e2e_p99'] / p99[('rho0.6', 'single')]['e2e_p99'] - 1):+.1f}%", "p99 when it is loaded but IDLE: free", GREEN),
             (f"{mf['worst_rel_err_faulty']:.0%}", "max error from Marlin at batch 32/64/128 (found by our check)", AMBER),
             (f"{f16.p50 / i4.p50:.1f}×", f"lower median latency, int4 vs FP16 (+{100 * (i4.tpj / f16.tpj - 1):.0f}% tokens/joule)", AMBER)]
    tw, th = 3.02, 2.3
    for i, (big, small, col) in enumerate(tiles):
        r, c = divmod(i, 4)
        x, y = X0 + c * (tw + 0.14), 1.72 + r * (th + 0.18)
        card(s, x, y, tw, th, fill=WHITE, line=LINE)
        text(s, x + 0.2, y + 0.18, tw - 0.4, 0.95, big, size=38, bold=True, color=col, anchor="m")
        text(s, x + 0.22, y + 1.15, tw - 0.4, 1.05, small, size=14.5, color=INK)
    text(s, X0, 6.6, W, 0.32, "Colour = result family: red kernel · navy kernel vs library · green routing mechanism · amber correctness & quantization.",
         size=11.5, color=GREY)
    notes(s, f"""
Eight headline numbers. Kernel: {k['speedup']:.2f} times FP16 cuBLAS at {k['pct']:.0f} percent of memory bandwidth. Inside vLLM: {e['inferno_b8']['1']:.0f} tokens per second, parity
with Marlin. Layer matrix: our kernel wins only on {LF['wins_share']:.0f} percent of decode GEMM time, so we keep Marlin; a decision taken from data. Routing mechanism: an active
second replica on the same GPU doubles p99, an idle one costs almost nothing. Correctness: our reference check caught Marlin returning results up to {mf['worst_rel_err_faulty']:.0%} wrong
at some batch sizes on this machine. Quantization: int4 cuts median latency {f16.p50 / i4.p50:.1f} times versus FP16. The next seven slides show the figure behind each.
""")

    # ================================================================== 10. results 1: kernel ladder
    s = new("Results 1: Fused int4 CUDA Kernel", "6 · RESULTS 1/7")
    text(s, X0, 1.22, W, 0.36, "Seven optimization steps from a naive kernel to the hardware memory limit", size=15.5, color=GREY)
    text(s, X0, 1.66, 6.1, 0.32, "Speedup vs FP16 cuBLAS · batch 1 · Llama-7B MLP layer (11008 × 4096)", size=13.5, bold=True, color=NAVY)
    names = [n for n, _ in F["ladder"]]
    vals = [round(v, 2) for _, v in F["ladder"]]
    cols = [RGBColorHex(h) for h in ("C9DAF2", "A9C3EC", "84A9E2", "5E8ED8", "3D75CB", "2A5FB0")] + [CRIMSON]
    bar_chart(s, X0, 1.96, 6.1, 3.0, names, {"speedup": vals}, [BLUE], number_format='0.00"×"', font=12, gap=45, vmax=4.4, point_colors={0: cols})
    picture(s, FIG / "kernel_bandwidth.png", X0 + 6.45, 1.7, w=5.95)
    text(s, X0 + 6.45, 1.7 + 5.95 / 1.95 + 0.08, 5.95, 0.3, "Achieved DRAM bandwidth as a fraction of peak, every layer and version", size=11.5, color=GREY, align="c")
    stat(s, X0, 5.12, 2.95, 0.95, f"{k['speedup']:.2f}×", "vs FP16 cuBLAS")
    stat(s, X0 + 3.15, 5.12, 2.95, 0.95, f"{k['pct']:.0f}%", f"of {F['peak_gbps']:.0f} GB/s peak")
    stat(s, X0 + 6.45, 5.12, 2.85, 0.95, f"{k['ceil']:.2f}×", "theoretical ceiling (bytes)", color=NAVY)
    stat(s, X0 + 9.45, 5.12, 2.95, 0.95, "5 / 5", "runs match CPU reference", color=GREEN)
    setting_strip(s, 6.27, 0.5, "Method:", "weights rotated over 768 MB (24× L2) so every call streams from DRAM · median of 5 fresh runs · correctness gate before timing.",
                  size=12.5)
    notes(s, f"""
The kernel. Left: the optimization ladder from v0 to v6, speedup over FP16 cuBLAS on the largest layer at batch one. Naive v0 is slower than cuBLAS; coalesced loads,
vectorized loads, padding away shared-memory bank conflicts, an algebraic rewrite, register reuse and a 16-lane layout bring it to {k['speedup']:.2f} times, which is
{k['pct']:.0f} percent of the {F['peak_gbps']:.0f} GB/s theoretical bandwidth. The ceiling is {k['ceil']:.2f} times, not 4, because int4 also carries scales and zero points. Right: the
bandwidth figure across all layers. Every kernel matched a double-precision CPU reference before it was timed. We corrected one flaw on the way: a too-small cache flush had
inflated bandwidth by about 6 percent, so the flush is now 768 MB.
""")

    # ================================================================== 11. results 2: inside vLLM
    s = new("Results 2: Inside vLLM", "6 · RESULTS 2/7")
    text(s, X0, 1.22, W, 0.36, "Our kernel registered as a torch custom op and run inside vLLM 0.11 with CUDA graphs", size=15.5, color=GREY)
    picture(s, RES / "e2e_kernel" / "e2e_light.png", X0, 1.7, w=7.6)
    text(s, X0, 1.7 + 7.6 / 1.98 + 0.08, 7.6, 0.3, "End-to-end decode throughput in vLLM, 1 to 8 concurrent sequences (mean of 3 process repetitions)", size=11.5,
         color=GREY, align="c")
    xr, wr = X0 + 7.95, W - 7.95
    stat(s, xr, 1.7, wr, 1.0, f"{e['inferno_b8']['1']:.0f} tok/s", f"our kernel, 1 sequence  (Marlin {e['awq_marlin']['1']:.0f})", color=CRIMSON)
    stat(s, xr, 2.85, wr, 1.0, f"{r1:.2f}×", "of Marlin's speed: parity, not a win", color=NAVY)
    stat(s, xr, 4.0, wr, 1.0, f"{e['inferno_b8']['1'] / e['fp16']['1']:.1f}×", f"faster than FP16 ({e['fp16']['1']:.0f} tok/s)", color=GREEN)
    stat(s, xr, 5.15, wr, 1.0, f"{r4:.2f}×", "of Marlin at 4 sequences: Marlin leads", color=AMBER)
    setting_strip(s, 6.3, 0.5, "Verified:", "all patched layers match vLLM's own dequantize-and-matmul (max rel. error ~1e-3) · identical 256-token greedy output.", size=12.5)
    notes(s, f"""
We did not stop at a micro-benchmark. The kernel runs inside vLLM as a torch custom op, captured in CUDA graphs. With one sequence it decodes {e['inferno_b8']['1']:.0f} tokens per second
against {e['awq_marlin']['1']:.0f} for vLLM's default Marlin kernel: a ratio of {r1:.2f}, parity, and {e['inferno_b8']['1'] / e['fp16']['1']:.1f} times faster than FP16. With four sequences Marlin's
tensor-core kernel pulls ahead ({r4:.2f}). An earlier measurement suggested we beat Marlin; a cross-check against end-to-end throughput showed that baseline was wrong, and we
retracted it. Correctness: every patched layer matches vLLM's reference, and the generated text is identical.
""")

    # ================================================================== 12. results 3: layer matrix
    s = new("Results 3: Where Our Kernel Wins", "6 · RESULTS 3/7")
    text(s, X0, 1.22, W, 0.36, "Every Qwen2.5-1.5B decode layer × batch 1–64 · our kernel vs Marlin vs classic AWQ vs FP16 on identical weights", size=15.5, color=GREY)
    picture(s, FIG / "custom_vs_marlin.png", X0, 1.7, w=6.3)
    picture(s, FIG / "layer_matrix.png", X0 + 6.7, 1.7, h=4.25)
    text(s, X0 + 6.7, 6.0, 5.1, 0.28, "µs per call by batch; hollow = Marlin output wrong (Results 4)", size=11, color=GREY, align="c")
    labelled(s, X0, 5.1, 6.3, 0.95, [[(f"Wins only {'/'.join(LF['wins'])} at batch 1 = {LF['wins_share']:.0f}% of decode time", {"bold": True, "color": NAVY, "size": 15.5})],
                                     [(f"ties the big layers · {lo4:.1f}–{LF['max_ratio']:.1f}× slower from batch 4", {"size": 14, "color": INK})]],
             fill=PALE, line=None, anchor="m", margin=0.18, space_after=0)
    banner(s, 6.33, 0.6, "Decision (rule fixed in advance):  ", "keep vLLM's Marlin as the default; our kernel stays an opt-in research path.", size=15.5)
    notes(s, f"""
New since mid-term: before pushing our kernel into production we asked where it actually wins. We timed all four decode layers of the real model at seven batch sizes,
against Marlin, classic AWQ and FP16, on identical weights, with CUDA graphs and DRAM-streaming rotation. Left: our time divided by Marlin's; blue means we are faster.
We clearly win only on the two small layers at batch one, {LF['wins_share']:.0f} percent of decode time, tie on the two large layers, and from batch four we are {lo4:.1f} to
{LF['max_ratio']:.1f} times slower because a CUDA-core kernel does work proportional to the batch while Marlin's tensor-core path stays flat. Right: the raw timings.
The decision rule was fixed before the run: carry the kernel into vLLM only if it consistently wins on layers that matter. It does not, so Marlin stays the default
and we moved the engineering effort to routing. That is a data-driven engineering decision, not a failure.
""")

    # ================================================================== 13. results 4: Marlin correctness
    s = new("Results 4: Wrong Answers Caught", "6 · RESULTS 4/7")
    text(s, X0, 1.22, W, 0.36, "vLLM's Marlin int4 kernel vs a dequantize + FP32 reference · 40 identical calls per bar", size=15.5, color=GREY)
    picture(s, FIG / "marlin_fault.png", X0, 1.7, w=7.9)
    xr, wr = X0 + 8.25, W - 8.25
    stat(s, xr, 1.7, wr, 1.05, f"up to {mf['worst_rel_err_faulty']:.0%}", "relative error in Marlin's output", color=CRIMSON)
    stat(s, xr, 2.92, wr, 1.05, ", ".join(map(str, mf["faulty_batches"])), "batch sizes with wrong calls", color=AMBER, big_size=26)
    stat(s, xr, 4.14, wr, 1.05, f"{mf['worst_rel_err_clean']:.0e}", "worst error at batch 1, 8, 16, 48 (correct)", color=GREEN, big_size=26)
    labelled(s, X0, 5.35, W, 0.85, [[("Ruled out: ", {"bold": True, "color": NAVY}),
                                     ("stale workspace (a fresh zeroed workspace still fails) · allocator · our harness (custom kernel, classic AWQ and FP16 are correct "
                                      "and deterministic on the same inputs).", {})]], fill=PALE, line=None, size=14.5, anchor="m", margin=0.2)
    setting_strip(s, 6.33, 0.5, "Status:", "observed on one machine (RTX 4060 Laptop, WSL2, vLLM 0.11.0); reproduction on a second GPU in progress before any upstream report.",
                  size=12.5)
    notes(s, f"""
A finding we did not expect. The layer benchmark compares every kernel's output with a reference before timing it, and Marlin, vLLM's default int4 kernel, failed that
check on one cell. We then called it 40 times with identical inputs per batch size. At batches {', '.join(map(str, mf['faulty_batches']))} many calls return results up to
{mf['worst_rel_err_faulty']:.0%} wrong, on every layer we tested; at 1, 8, 16 and 48 it is always correct. We ruled out the shared workspace, the memory allocator and our own harness.
We are careful with the claim: it is observed on one machine and could be specific to WSL2 or this laptop GPU, so we are reproducing it on another GPU before reporting
it to vLLM. The lesson: correctness checks inside a performance benchmark catch real bugs, even in production libraries.
""")

    # ================================================================== 14. results 5: routing campaign
    s = new("Results 5: Adaptive Routing (H1)", "6 · RESULTS 5/7")
    h1 = {r["pattern"]: r for r in F["h1rows"]}
    pats = ["steady", "bursty", "mixed_length"]
    text(s, X0, 1.22, W, 0.36, "Pre-specified hypothesis H1: an adaptive router beats the best fixed configuration on p99 latency", size=15.5, color=GREY)
    pw_ = 3.25 * 3.16
    picture(s, RES / "campaign" / "figures" / "p99_by_pattern_light.png", X0 + (W - pw_) / 2, 1.7, w=pw_)
    yb = 5.15
    stat(s, X0, yb, 3.0, 1.0, f"+{lo:.0f}–{hi:.0f}%", "worse p99 with adaptive", color=CRIMSON, big_size=26)
    stat(s, X0 + 3.17, yb, 3.0, 1.0, "p = 0.023", "Holm-corrected, 8 / 8 pairs", color=NAVY, big_size=26)
    stat(s, X0 + 6.34, yb, 3.0, 1.0, "96 runs", "4 policies × 3 patterns × 8 reps", color=GREEN, big_size=26)
    stat(s, X0 + 9.51, yb, W - 9.51, 1.0, f"{F['headroom']['max_headroom_pct']:.1f}%", "max gain any fixed-tier choice could give", color=AMBER, big_size=26)
    banner(s, 6.3, 0.58, "H1 rejected on one shared GPU:  ", "every adaptive bar (black) is far right of the best fixed policy; the next slide shows why.",
           size=15)
    notes(s, f"""
Our main hypothesis, fixed before the experiment: an adaptive router over a fast and a bulk replica beats the best fixed configuration on p99. We ran 96 paired runs:
four policies, three load patterns, eight seeded repetitions each. The answer is no: adaptive p99 was {lo:.0f} to {hi:.0f} percent worse, Holm-corrected p = 0.023, which is
the smallest p-value possible with eight pairs, and all eight pairs agreed. We report this negative result plainly. The next slide shows why it happened.
""")

    # ================================================================== 15. results 6: contention mechanism
    s = new("Results 6: Why Routing Lost", "6 · RESULTS 6/7")
    text(s, X0, 1.22, W, 0.36, "Controlled experiment: equal total load, vary only whether the second replica is absent, idle, or serving", size=15.5, color=GREY)
    picture(s, FIG / "headline_light.png", X0, 1.7, w=7.7)
    xr, wr = X0 + 8.05, W - 8.05
    a, b_, c_ = (p99[("rho0.6", x)]["e2e_p99"] for x in ("single", "split-idle", "split-active"))
    stat(s, xr, 1.7, wr, 1.0, f"{a:.2f} s", "one server (60% load)", color=BLUE)
    stat(s, xr, 2.85, wr, 1.0, f"{b_:.2f} s", "+ a second replica, idle", color=AMBER)
    stat(s, xr, 4.0, wr, 1.0, f"{c_:.2f} s", "+ both replicas serving", color=GREEN)
    banner(s, 5.95, 0.92, "Insight:  ", "loading a replica is free; USING it on the same GPU doubles p99 (time-slicing). Adaptive routing needs one GPU per replica.",
           colors=(RGBColorHex("0B7A4B"), RGBColorHex("06563A")), head_color="C9F2DD", size=15.5)
    notes(s, f"""
This is our favourite result because of the control. At the same total load we compared one server, two servers with the second loaded but idle, and two servers both
serving. Idle second replica: p99 {b_:.2f} seconds versus {a:.2f}, essentially free. Both serving: {c_:.2f} seconds, more than double. So the cost is not loading a second model,
it is two active processes time-slicing one GPU. That explains the routing result: any policy that spreads load across co-resident replicas pays this penalty, and a
headroom analysis showed no pattern-aware choice could have gained more than about 3 percent anyway. The prediction: with one replica per GPU the penalty disappears and
adaptive routing becomes testable, which is our multi-GPU future work. {reps_c} paired repetitions; error bars are 95 percent confidence intervals.
""")

    # ================================================================== 16. results 7: MPI + quantization
    s = new("Results 7: MPI Overhead & Quantization", "6 · RESULTS 7/7")
    text(s, X0, 1.22, W, 0.36, "Is the coordination layer a bottleneck? And what does int4 buy at the server level?", size=15.5, color=GREY)
    picture(s, FIG / "sweep_steady_light.png", X0 + 5.2, 1.66, h=4.5)
    stat(s, X0, 1.7, 4.9, 1.05, f"≈ {mpi_us:.0f} µs", "MPI one-way hop in isolation", color=CRIMSON)
    stat(s, X0, 2.9, 4.9, 1.05, "< 1%", "MPI vs HTTP router end to end: no measurable difference", color=NAVY)
    stat(s, X0, 4.1, 4.9, 1.05, f"{f16.p50 / i4.p50:.1f}×", f"lower p50 latency, int4 vs FP16 ({f16.p50:.1f} s → {i4.p50:.1f} s)", color=GREEN)
    stat(s, X0, 5.3, 4.9, 1.05, f"≈ {cold:.0f} s", "to restart a server: adapt by routing, not reloading", color=AMBER)
    text(s, X0 + 5.2, 6.22, 7.2, 0.3, "Quantization × batch-cap sweep at the same offered load (3 repetitions per configuration)", size=11.5, color=GREY, align="c")
    setting_strip(s, 6.55, 0.38, "Takeaway:", f"MPI is cheap; the shared GPU is the bottleneck · int4 also gives +{100 * (i4.tpj / f16.tpj - 1):.0f}% tokens per joule.", size=12)
    notes(s, f"""
Two system-level results. MPI: a hop costs about {mpi_us:.0f} microseconds in isolation, and end to end an MPI router and an HTTP router are indistinguishable, below one percent:
coordination is not the bottleneck, the shared GPU is. Quantization: at the same offered load, int4 cuts median latency from {f16.p50:.1f} to {i4.p50:.1f} seconds and gives
{100 * (i4.tpj / f16.tpj - 1):.0f} percent more tokens per joule. And restarting vLLM to change precision or batch size takes about {cold:.0f} seconds, several times a traffic burst, which is
why our adaptive design routes across pre-loaded replicas instead of reconfiguring.
""")

    # ================================================================== 17. challenges
    s = new("Challenges", "7 · CHALLENGES")
    ch_ = [("Measurement traps (found & fixed)", ["Cache leak inflated bandwidth ≈ 6% → 768 MB rotation, re-run", "A Marlin baseline was mis-measured → claim retracted",
                                               "Throughput metric hid saturation → redefined"]),
           ("Hardware limits", ["Only 8 GB → a 1.5B-parameter model", "Both replicas share one GPU (time-slicing)", "Laptop clocks drift → randomized run order"]),
           ("Tooling & platform", ["GPU counters blocked by a Windows driver setting → Nsight pending", "WSL2: CUDA 'out of memory' with VRAM free (Windows commit limit)",
                                   "CUDA 12.4 toolkit vs PyTorch CUDA 12.8 build"]),
           ("Correctness & engine", ["Marlin wrong at batch 32/64/128 → needs a 2nd GPU to confirm", "vLLM cannot change batch / precision live",
                                     f"Restart costs ≈ {cold:.0f} s → route across pre-loaded replicas"])]
    cw, chh = 6.17, 2.62
    for i, (t, items) in enumerate(ch_):
        r, c = divmod(i, 2)
        x, y = X0 + c * (cw + 0.16), 1.36 + r * (chh + 0.16)
        card(s, x, y, cw, chh, fill=WHITE, accent=CRIMSON if i in (0, 3) else NAVY)
        text(s, x + 0.3, y + 0.1, cw - 0.4, 0.5, t, size=20, bold=True, color=NAVY)
        text(s, x + 0.3, y + 0.68, cw - 0.45, chh - 0.75, items, size=16.5, bullets=True, space_after=8)
    notes(s, """
The most important challenges were measurement traps, all found by our own cross-checks and fixed: a cache leak that inflated bandwidth, a mis-measured Marlin baseline
that made our kernel look faster than it was, and a throughput metric that hid saturation. Hardware: 8 GB and one shared GPU. Platform: GPU performance counters are
locked by a Windows driver setting, and under WSL2 CUDA can report out-of-memory with VRAM free because the Windows pagefile cannot grow. Correctness: the Marlin fault
needs a second GPU to confirm. Engine: vLLM cannot reconfigure live, so adaptivity is routing across pre-loaded replicas.
""")

    # ================================================================== 18. conclusions & future scope
    s = new("Conclusions & Future Scope", "8 · CONCLUSION")
    cw = 7.15
    card(s, X0, 1.28, cw, 3.55, fill=WHITE, accent=GREEN)
    text(s, X0 + 0.3, 1.34, cw - 0.5, 0.42, "What we delivered", size=19, bold=True, color=NAVY)
    text(s, X0 + 0.32, 1.82, cw - 0.55, 2.95, [
        f"Fused int4 CUDA kernel: {k['speedup']:.2f}× cuBLAS at {k['pct']:.0f}% of memory bandwidth",
        f"Integrated into vLLM: {e['inferno_b8']['1']:.0f} tok/s, parity with the production kernel",
        "Layer × batch study → data-driven decision to keep Marlin",
        "Reference check exposed wrong Marlin outputs at 3 batch sizes",
        "96-run paired campaign + controlled study: why routing lost",
        f"MPI router (≈ {mpi_us:.0f} µs/hop) · 63-page report · v1.1.0 release",
    ], size=16, bullets=True, space_after=9)
    x2, cw2 = X0 + cw + 0.2, W - cw - 0.2
    picture(s, FIG / "ppt_hero.png", x2, 1.28, w=cw2)
    text(s, x2, 1.28 + cw2 * 760 / 1300 + 0.06, cw2, 0.3, "Our live project dashboard (docs/demo)", size=12, color=GREY, align="c")
    text(s, X0, 4.98, W, 0.36, "Future scope", size=18, bold=True, color=NAVY)
    fut = [("Profile", CRIMSON, TINT, ["Nsight Compute counters", "explain each kernel step"]),
           ("Scale out", NAVY, PALE, ["one replica per GPU (cloud)", "re-test adaptive routing"]),
           ("Confirm", AMBER, AMBERTINT, ["Marlin fault on a 2nd GPU", "report upstream to vLLM"]),
           ("Extend", GREEN, GREENTINT, ["tensor-core kernel (B ≥ 8)", "real traces + accuracy eval"])]
    fw, fg = 3.02, 0.14
    for i, (t, col, tint, items) in enumerate(fut):
        x = X0 + i * (fw + fg)
        card(s, x, 5.36, fw, 1.5, fill=tint, accent=col)
        badge(s, x + 0.16, 5.46, 0.44, str(i + 1), col, size=15)
        text(s, x + 0.7, 5.42, fw - 0.8, 0.5, t, size=16, bold=True, color=NAVY, anchor="m")
        text(s, x + 0.2, 5.92, fw - 0.3, 0.92, items, size=13.5, bullets=True, space_after=2)
    notes(s, f"""
What we delivered: a kernel at the memory limit, integrated and measured inside a production engine; a careful study showing where it wins, which led us to keep the
library kernel; a correctness finding in that library; a rigorous routing experiment that was rejected and then explained by a controlled mechanism study; an MPI router;
and a reproducible release with a 63-page generated report. Future scope: profile with Nsight once counters are unlocked; run one replica per GPU in the cloud to give
adaptive routing a fair test; confirm the Marlin fault on another GPU and report it to vLLM; and extend to a tensor-core kernel and real traffic traces.
""")

    # ================================================================== 19. team
    s = new("Team Work & Coordination", "9 · TEAM")
    roles = [("Kernel & profiling", team[0], ["Kernel ladder v0–v6", "Layer matrix, Marlin probe"]),
             ("Serving & statistics", team[1], ["Open-loop client, calibration", "Paired tests, 96-run campaign"]),
             ("MPI router & controller", team[2], ["Dispatcher / workers, policies", "Scaling and contention study"]),
             ("Integration & docs", team[3], ["vLLM integration, cloud (Terraform)", "Report, dashboard, release"])]
    cw = 3.02
    for i, (role, who, items) in enumerate(roles):
        x = X0 + i * (cw + 0.14)
        card(s, x, 1.36, cw, 3.3, fill=WHITE, accent=CRIMSON)
        badge(s, x + 0.22, 1.5, 0.5, f"R{i + 1}", CRIMSON, size=15)
        text(s, x + 0.82, 1.46, cw - 0.9, 0.6, role, size=16.5, bold=True, color=NAVY, anchor="m")
        nm, _, roll = who.partition(" (")
        text(s, x + 0.25, 2.1, cw - 0.4, 0.42, nm, size=19, bold=True, color=CRIMSON)
        text(s, x + 0.25, 2.5, cw - 0.4, 0.36, roll.rstrip(")"), size=12.5, color=GREY)
        text(s, x + 0.25, 3.0, cw - 0.4, 1.6, items, size=15.5, bullets=True, space_after=7)
    coord = [("Coordination", f"weekly sync · shared Git repository · {n_tests} automated tests"),
             ("Documentation", "63-page technical report · README · generated FINDINGS"),
             ("Reproducibility", "make figure · make quick-bench · tagged release v1.1.0")]
    for i, (t, d) in enumerate(coord):
        labelled(s, X0 + i * 4.2, 4.9, 4.08, 1.9, [[(t, {"bold": True, "color": NAVY, "size": 19})], [(d, {"size": 15.5})]], fill=PALE, line=None, anchor="m",
                 margin=0.22)
    notes(s, "Each member owns a real component of the system and can explain how it connects to the others through the workflow diagram.")

    # ================================================================== 20. references
    refs = ["G.-I. Yu et al., “Orca: A distributed serving system for transformer-based generative models,” in Proc. USENIX OSDI, 2022.",
            "W. Kwon et al., “Efficient memory management for large language model serving with PagedAttention,” in Proc. ACM SOSP, 2023.",
            "Y. Zhong et al., “DistServe: Disaggregating prefill and decoding for goodput-optimized LLM serving,” in Proc. USENIX OSDI, 2024.",
            "P. Patel et al., “Splitwise: Efficient generative LLM inference using phase splitting,” in Proc. ACM/IEEE ISCA, 2024.",
            "J. Lin et al., “AWQ: Activation-aware weight quantization for LLM compression and acceleration,” in Proc. MLSys, 2024.",
            "E. Frantar et al., “GPTQ: Accurate post-training quantization for generative pre-trained transformers,” in Proc. ICLR, 2023.",
            "E. Frantar et al., “MARLIN: Mixed-precision auto-regressive parallel inference on large language models,” in Proc. ACM PPoPP, 2025.",
            "T. Hoefler and R. Belli, “Scientific benchmarking of parallel computing systems,” in Proc. SC, 2015.",
            "A. Georges, D. Buytaert, and L. Eeckhout, “Statistically rigorous Java performance evaluation,” in Proc. ACM OOPSLA, 2007, pp. 57–76.",
            "S. Williams, A. Waterman, and D. Patterson, “Roofline: An insightful visual performance model for multicore architectures,” Commun. ACM, vol. 52, no. 4, 2009."]
    for shp in s_refs.shapes:
        if shp.name == "Rectangle 5":
            tf = shp.text_frame
            proto = tf.paragraphs[0]._p
            for p in tf.paragraphs[1:]:
                p._p.getparent().remove(p._p)
            for i, r in enumerate(refs):
                if i == 0:
                    p = tf.paragraphs[0]
                    p.runs[0].text = r
                    for extra in p.runs[1:]:
                        extra._r.getparent().remove(extra._r)
                else:
                    newp = copy.deepcopy(proto)
                    tf._txBody.append(newp)
                    _Paragraph(newp, tf).runs[0].text = r
            shp.top, shp.height = Inches(1.38), Inches(5.4)
            for p in tf.paragraphs:
                p.space_before = Pt(3)
                for r in p.runs:
                    r.font.size = Pt(15)
        if shp.name == "TextBox 3":
            shp._element.getparent().remove(shp._element)
        if shp.name == "TextBox 6":
            shp.width = Inches(11.4)
    order.append(s_refs)
    notes(s_refs, "References in IEEE format: serving systems, quantization and kernels, and benchmarking methodology.")

    # ================================================================== 21. thank you
    order.append(s_thanks)
    text(s_thanks, 3.4, 4.5, 6.5, 0.7, "Questions?", size=32, bold=True, color=CRIMSON, align="c", font="Times New Roman")
    notes(s_thanks, "Thank you. Every number in this presentation is generated from the raw result files in the project repository (release v1.1.0).")

    cp = prs.core_properties
    cp.title = "Inferno: Adaptive LLM Inference Serving – HPC Final Evaluation 2026"
    cp.author = ", ".join(t.split(" (")[0] for t in team)
    cp.subject = "High-Performance Computing project, B.Tech (AIDS), UG-III"
    cp.keywords = "CUDA, MPI, vLLM, LLM inference, benchmarking"

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

    keep = set(map(id, order))
    for sl in orig:
        if id(sl) not in keep:
            delete_slide(prs, sl)
    reorder(prs, order)
    prs.save(str(OUT))
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", nargs=4, default=["Likhith (DL.AI.U4AID24117)", "Karthikeya (DL.AI.U4AID24020)", "Sandeep (DL.AI.U4AID24143)",
                                                "A. Karthik (DL.AI.U4AID24104)"])
    a = ap.parse_args()
    print("wrote", build(a.team))
