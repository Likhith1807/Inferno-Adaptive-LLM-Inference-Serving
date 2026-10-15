"""Diagrams for the technical report, drawn with matplotlib from the actual design parameters and computed values."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

BLUE, AQUA, ORANGE, INK, GREY, RED = "#1f6fd0", "#17916a", "#d9622b", "#14171c", "#8a9099", "#c0392b"
INK2, RULE = "#4a4f57", "#d5d8de"
LIGHT = {"blue": "#e6f0fc", "aqua": "#e3f4ee", "orange": "#fbece4", "grey": "#f2f4f7", "red": "#f9e6e3"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _box(ax, x, y, w, h, text, fc="grey", ec=GREY, fs=8.4, bold=False, tc=INK, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=LIGHT.get(fc, fc), ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.25)


def _arrow(ax, p, q, color=INK, lw=1.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(path)


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    ax.set_xlim(0, 20.8)
    ax.set_ylim(0, 11.8)
    ax.axis("off")
    _box(ax, 0.2, 8.6, 4.4, 2.6, "Workload generator\nPoisson arrivals\nbursty schedule\nlognormal / bimodal lengths\nseeded per repetition", "blue", BLUE, bold=False, fs=7.0)
    _box(ax, 5.6, 8.9, 3.6, 2.0, "Open-loop client\nstreams /v1/completions\nTTFT, TPOT, e2e", "blue", BLUE, fs=7.2)
    _arrow(ax, (4.6, 9.9), (5.6, 9.9))
    # router layer
    ax.text(12.7, 11.55, "Routing layer (one of)", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    _box(ax, 10.0, 9.7, 5.4, 1.5, "HTTP router\nleast-outstanding baseline", "aqua", AQUA)
    _box(ax, 10.0, 7.9, 5.4, 1.5, "MPI dispatcher (rank 0) + workers\nisend / iprobe, per-hop timestamps", "aqua", AQUA, fs=7.4)
    _box(ax, 10.0, 6.1, 5.4, 1.5, "Policy router\nPinned | LeastLoaded | AdaptiveTier", "orange", ORANGE, fs=7.2)
    _arrow(ax, (9.2, 9.9), (10.0, 10.4))
    _arrow(ax, (9.2, 9.7), (10.0, 8.7))
    _arrow(ax, (9.2, 9.5), (10.0, 6.9))
    # pool
    ax.add_patch(Rectangle((16.3, 3.6), 4.3, 7.6, fc="#fafbfc", ec=INK, lw=1.4))
    ax.text(18.45, 10.9, "ONE 8 GB GPU", ha="center", fontsize=8.6, fontweight="bold")
    _box(ax, 16.6, 7.6, 3.7, 2.9, "Replica tier 0 (fast)\nvLLM, AWQ int4 (Marlin)\nmax_num_seqs 16\nprefill chunk 512", "blue", BLUE, fs=7.7)
    _box(ax, 16.6, 4.0, 3.7, 3.2, "Replica tier 1 (bulk)\nvLLM, AWQ int4 (Marlin)\nmax_num_seqs 64\nprefill chunk 4096", "orange", ORANGE, fs=7.7)
    for y in (10.4, 8.7, 6.9):
        _arrow(ax, (15.4, y), (16.6, 9.0 if y > 8 else 5.6))
    # measurement
    _box(ax, 5.6, 5.6, 3.6, 1.9, "Power sampler\nnvidia-smi @ 200 ms\nenergy, tokens per joule", "grey", GREY, fs=7.2)
    _box(ax, 5.6, 3.2, 3.6, 1.9, "Per-request logs\nJSONL + summary.csv\nhardware.json", "grey", GREY)
    _arrow(ax, (18.45, 3.6), (18.45, 2.2), color=GREY)
    ax.plot([18.45, 9.2], [2.2, 2.2], color=GREY, lw=1.1)
    _arrow(ax, (9.2, 2.2), (9.2, 3.5), color=GREY)
    _arrow(ax, (7.4, 5.6), (7.4, 5.1), color=GREY)
    # analysis
    _box(ax, 0.2, 3.2, 4.6, 1.9, "Analysis\npaired tests, Holm, effect sizes\nbootstrap CIs, verdict rules", "aqua", AQUA, fs=7.4)
    _arrow(ax, (5.6, 4.15), (4.8, 4.15))
    _box(ax, 0.2, 0.4, 4.6, 1.9, "Generated outputs\nREADME blocks, FINDINGS.md\nshowcase site, this PDF", "grey", GREY)
    _arrow(ax, (2.5, 3.2), (2.5, 2.3))
    # kernel + calibration
    _box(ax, 5.6, 0.4, 5.0, 1.9, "Standalone CUDA kernel benchmark\nCPU reference gate, rotating weights,\n5 invocations, Nsight scripts", "orange", ORANGE, fs=7.4)
    _box(ax, 11.4, 0.4, 4.3, 1.9, "Capacity calibration\nrates = fractions of\nmeasured capacity", "blue", BLUE, fs=7.9)
    _arrow(ax, (13.5, 2.3), (13.5, 6.1), color=BLUE, ls="--")
    ax.text(13.62, 3.9, "sets load", fontsize=7.4, color=BLUE)
    return _save(fig, out)


def fig_phases_roofline(out):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.1, 5.6, "One request, two phases", fontsize=9.6, fontweight="bold")
    _box(ax, 0.2, 3.2, 3.4, 1.7, "PREFILL\nall prompt tokens in one pass\nweights reused across tokens\ncompute-bound", "blue", BLUE, fs=7.9)
    for i in range(6):
        _box(ax, 4.4 + i * 0.9, 3.2, 0.78, 1.7, "T", "orange", ORANGE, fs=8)
    ax.text(4.4, 5.05, "DECODE: one token per step, autoregressive", fontsize=7.6, color=ORANGE)
    ax.text(4.4, 2.75, "every step re-reads all weights (and the KV cache)", fontsize=7.4, color=INK)
    ax.text(0.2, 2.0, "At batch size 1 a decode matvec does ~1 multiply-add per weight\nread from DRAM, so runtime is set by bytes moved, not FLOPs.", fontsize=7.9, va="top")
    ax.text(0.2, 0.55, "Batching raises arithmetic intensity (more tokens per weight read),\nwhich is why serving engines batch continuously.", fontsize=7.9, va="top", color=INK)
    ax = axes[1]
    ai = np.logspace(-1, 3, 200)
    bw = 256e9
    peak = 60e12
    perf = np.minimum(peak, bw * ai)
    ax.loglog(ai, perf / 1e12, color=INK, lw=2)
    ax.set_xlabel("Arithmetic intensity (FLOP per DRAM byte)")
    ax.set_ylabel("Attainable performance (TFLOP/s)")
    ax.set_title("Roofline (RTX 4060 Laptop bandwidth 256 GB/s; compute roof illustrative)", fontsize=8.8, loc="left")
    pts = [("fp16 GEMV\n(batch 1)", 1.0, BLUE), ("int4 GEMV\n(batch 1)", 2 / 0.53125, AQUA), ("fp16, batch 32", 32.0, GREY), ("int4, batch 32", 32 * 2 / 0.53125, GREY)]
    for name, x, c in pts:
        y = min(peak, bw * x) / 1e12
        ax.plot([x], [y], "o", color=c, ms=8, mec="white", mew=1.4, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, -20 if x < 10 else 7), fontsize=7.6, color=c)
    ax.axvspan(0.1, peak / bw, color=LIGHT["orange"], alpha=0.6, zorder=0)
    ax.text(0.13, 30, "memory-bound region", fontsize=8, color=ORANGE)
    ax.annotate("", xy=(3.76, 0.35), xytext=(1.0, 0.35), arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.3))
    ax.text(1.05, 0.18, "3.76x fewer bytes", fontsize=7.6, color=AQUA)
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.03, 100)
    fig.tight_layout()
    return _save(fig, out)


def fig_layout(out):
    fig, ax = plt.subplots(figsize=(10.2, 3.7))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.text(0, 10.3, "Weight layout and the work assignment of one lane (asymmetric, group-wise int4, G = 128)", fontsize=9.6, fontweight="bold")
    # row of words
    for w in range(8):
        ax.add_patch(Rectangle((0.5 + w * 3.7, 6.4), 3.6, 1.6, fc=LIGHT["blue"], ec=BLUE, lw=1))
        ax.text(0.5 + w * 3.7 + 1.8, 7.75, f"uint32 word {w}", ha="center", fontsize=6.9, color=BLUE)
        for j in range(8):
            ax.add_patch(Rectangle((0.5 + w * 3.7 + j * 0.45, 6.4), 0.45, 1.0, fc="white", ec=GREY, lw=0.5))
            ax.text(0.5 + w * 3.7 + j * 0.45 + 0.225, 6.9, str(j), ha="center", va="center", fontsize=5.6, color=GREY)
    ax.text(0.5, 8.75, "row n of W: K/8 words; nibble j of word w holds k = 8w + j (little-endian)", fontsize=8, color=INK)
    ax.annotate("", xy=(0.5, 5.6), xytext=(0.5 + 8 * 3.7, 5.6), arrowprops=dict(arrowstyle="<->", color=AQUA, lw=1.2))
    ax.text(0.5 + 4 * 3.7, 5.05, "one 128-bit load (uint4) = 4 words = 32 weights = k0 .. k0+31, one lane", ha="center", fontsize=8, color=AQUA)
    ax.add_patch(Rectangle((0.5, 6.35), 4 * 3.7 - 0.1, 1.7, fc="none", ec=AQUA, lw=2))
    ax.add_patch(Rectangle((0.5, 2.6), 32 * 0.9, 1.4, fc=LIGHT["orange"], ec=ORANGE, lw=1))
    ax.text(0.9, 3.3, "group g: 128 consecutive k share one (scale s, zero z), stored as fp16 in [N][K/128] arrays", fontsize=8, va="center")
    ax.text(0.5, 1.5, "y[n] = sum over groups g of  s(n,g) * ( sum_k q(n,k) x(k)  -  z(n,g) * sum_k x(k) )      (k over the 128 weights of group g)", fontsize=8.2, color=INK)
    ax.text(0.5, 0.6, "A lane's 32 weights lie inside one group, so it applies s and z once per 32 weights (kernel v4).", fontsize=8, color=INK)
    return _save(fig, out)


def fig_access_pattern(out, K=1536):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), sharey=True)
    lanes = np.arange(32)
    stride = K // 2  # bytes per row of packed weights
    a0 = lanes * stride
    a1 = lanes * 4
    for ax, addr, title, col, note in ((axes[0], a0, "v0: one thread per row", RED, f"32 lanes touch 32 different rows,\n{stride} B apart: 32 separate\n128-byte segments per load"),
                                       (axes[1], a1, "v1: one warp per row", AQUA, "32 lanes read 32 consecutive words:\none 128-byte segment per load")):
        ax.scatter(lanes, addr, color=col, s=26, zorder=3)
        ax.set_title(title, fontsize=9.2, loc="left")
        ax.set_xlabel("lane (thread within a warp)")
        ax.grid(alpha=0.25)
        ax.text(0.04, 0.95, note, transform=ax.transAxes, va="top", fontsize=8, color=col)
    axes[0].set_ylabel("byte address of the lane's first load")
    axes[0].set_yscale("symlog", linthresh=200)
    fig.tight_layout()
    return _save(fig, out)


def bank_of(word_index: np.ndarray) -> np.ndarray:
    return word_index % 32


def fig_bank_conflicts(out):
    lanes = np.arange(32)
    layouts = [("v2 half activations, unpadded\nlane stride 64 B = 16 words", 16 * lanes),
               ("(float, unpadded, for comparison)\nlane stride 128 B = 32 words", 32 * lanes),
               ("v3 half, +1 pad word per 32 halves\nlane stride 68 B = 17 words", 17 * lanes),
               ("v4 float, +1 pad float per 32\nlane stride 132 B = 33 words", 33 * lanes)]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.1), sharey=True)
    for ax, (title, words) in zip(axes, layouts):
        banks = bank_of(words)
        counts = np.bincount(banks, minlength=32)
        worst = counts.max()
        ax.scatter(lanes, banks, s=18, color=RED if worst > 1 else AQUA, zorder=3)
        ax.set_title(title, fontsize=7.6, loc="left")
        ax.set_xlabel("lane", fontsize=8)
        ax.text(0.05, 0.93, f"{worst}-way" if worst > 1 else "conflict-free", transform=ax.transAxes, fontsize=8.6, fontweight="bold",
                color=RED if worst > 1 else AQUA, va="top")
        ax.grid(alpha=0.25)
        ax.set_ylim(-1, 33)
    axes[0].set_ylabel("shared-memory bank (word index mod 32)")
    fig.tight_layout()
    return _save(fig, out)


def fig_controller(out):
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    xs = [1.0, 7.3, 13.6]
    names = [("Level 0: calm", "all requests -> tier 0\n(low-latency decode tier)", "blue"),
             ("Level 1: pressured", "prefill-heavy (>= 512 tokens) -> last tier\nrest stay on tier 0", "orange"),
             ("Level 2: overloaded", "least-loaded across all tiers\n(utilization = in-flight / capacity)", "red")]
    for x, (t, d, c) in zip(xs, names):
        _box(ax, x, 3.6, 5.4, 3.2, f"{t}\n\n{d}", c, {"blue": BLUE, "orange": ORANGE, "red": RED}[c], fs=8.3)
    for i in range(2):
        _arrow(ax, (xs[i] + 5.4, 6.0), (xs[i + 1], 6.0), color=INK)
        _arrow(ax, (xs[i + 1], 4.4), (xs[i] + 5.4, 4.4), color=GREY, ls="--")
    ax.text(6.35, 6.35, "escalate", ha="center", fontsize=8, color=INK)
    ax.text(12.65, 6.35, "escalate", ha="center", fontsize=8, color=INK)
    ax.text(6.35, 3.85, "de-escalate", ha="center", fontsize=8, color=GREY)
    ax.text(12.65, 3.85, "de-escalate", ha="center", fontsize=8, color=GREY)
    ax.text(0.6, 2.75, "Escalate (once per 1 s tick) when tier-0 utilization >= 0.70 OR recent TTFT p95 > 0.5 s.", fontsize=8.3)
    ax.text(0.6, 2.05, "De-escalate one level only after 5 consecutive calm ticks (pool utilization <= 0.25 AND TTFT p95 <= 0.25 s): hysteresis.", fontsize=8.3)
    ax.text(0.6, 1.35, "Hard spill rule (all levels): if the chosen tier's utilization >= 0.90, send to the least-loaded tier.", fontsize=8.3)
    ax.text(0.6, 0.65, "Every tick is logged with a reason string; all thresholds were fixed before the campaign and not tuned on its data.", fontsize=8.3, color=INK)
    ax.text(0.6, 8.1, "Adaptive tier policy: a three-level, hysteresis-based controller (no learning)", fontsize=9.6, fontweight="bold")
    return _save(fig, out)


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 6)
    ax.axis("off")
    steps = [("1 Calibrate", "saturate a\nreference replica;\nmeasure capacity", "blue"), ("2 Derive load", "rates = fractions\nof capacity\n(0.6x / 0.35x-1.5x)", "blue"),
             ("3 Generate", "seeded streams,\nseed = 42 + rep\n(same for all policies)", "aqua"), ("4 Randomize", "policy order\nshuffled per\nrepetition", "aqua"),
             ("5 Execute", "open-loop replay\n+ power sampling", "orange"), ("6 Summarize", "TTFT/TPOT/p99,\ngoodput, energy", "orange"),
             ("7 Test", "paired t + Wilcoxon,\nHolm, d_z, bootstrap", "grey")]
    w, gap = 2.65, 0.32
    for i, (t, d, c) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        _box(ax, x, 1.3, w, 3.6, f"{t}\n\n{d}", c, {"blue": BLUE, "aqua": AQUA, "orange": ORANGE, "grey": GREY}[c], fs=7.6)
        if i < len(steps) - 1:
            _arrow(ax, (x + w, 3.1), (x + w + gap, 3.1))
    ax.text(0.15, 5.5, "Measurement pipeline (identical for every experiment)", fontsize=9.6, fontweight="bold")
    ax.text(0.15, 0.55, "Failures are data: unfinished requests, errors and drain time are recorded, never silently dropped.", fontsize=8.2)
    return _save(fig, out)


def fig_paired(out, reps=8, policies=("fixed-fast", "fixed-bulk", "static-lb", "adaptive"), seed=7):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.set_xlim(-1.6, len(policies) + 0.2)
    ax.set_ylim(-0.6, reps + 0.6)
    ax.invert_yaxis()
    ax.axis("off")
    ax.text(-1.55, -0.5, "repetition (seed)", fontsize=8, color=GREY)
    for j, p in enumerate(policies):
        ax.text(j + 0.5, -0.5, p, ha="center", fontsize=8.2, fontweight="bold", color=INK)
    for r in range(reps):
        ax.text(-1.2, r + 0.5, f"rep {r}   seed {42 + r}", fontsize=7.6, va="center", color=INK)
        order = rng.permutation(len(policies))
        for j in range(len(policies)):
            ax.add_patch(Rectangle((j + 0.04, r + 0.08), 0.92, 0.84, fc=LIGHT["blue"] if r % 2 == 0 else LIGHT["aqua"], ec="white"))
            ax.text(j + 0.5, r + 0.5, f"run {int(np.where(order == j)[0][0]) + 1}", ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(0, reps + 0.55, "Each row replays one identical seeded request stream against every policy (pairs); the number is that policy's random position within the repetition.",
            fontsize=7.4, color=INK2)
    return _save(fig, out)


def fig_timeslice(out):
    fig, ax = plt.subplots(figsize=(9.8, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 9)
    ax.axis("off")
    rows = [("1 replica", [(0, 100, BLUE)]), ("2 replicas, one idle", [(0, 100, ORANGE)])]
    for i, (name, segs) in enumerate(rows):
        y = 6.6 - i * 2.4
        ax.text(-1, y + 0.7, name, ha="right", va="center", fontsize=8.4)
        for a, b, c in segs:
            ax.add_patch(Rectangle((a, y), b - a, 1.4, fc=c, ec="white"))
    y = 1.8
    ax.text(-1, y + 0.7, "2 replicas, both active", ha="right", va="center", fontsize=8.4)
    x = 0
    k = 0
    while x < 100:
        c = BLUE if k % 2 == 0 else AQUA
        ax.add_patch(Rectangle((x, y), 8.2, 1.4, fc=c, ec="white"))
        ax.add_patch(Rectangle((x + 8.2, y), 1.1, 1.4, fc="#dfe3e8", ec="white"))
        x += 9.3
        k += 1
    ax.text(50, 0.5, "illustrative: kernels of two processes take turns (time slicing) with switch gaps; each replica also runs smaller batches", ha="center", fontsize=7.8, color=INK2)
    ax.text(50, 8.6, "GPU time, schematic", ha="center", fontsize=9, fontweight="bold")
    return _save(fig, out)


def fig_mpi_path(out):
    fig, ax = plt.subplots(figsize=(10.2, 3.9))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis("off")
    cols = {"Client / rank 0\n(dispatcher)": 2.0, "MPI\n(shared memory)": 7.0, "Rank r\n(replica worker)": 12.0, "vLLM replica\n(HTTP)": 17.6}
    for name, x in cols.items():
        ax.text(x, 7.55, name, ha="center", fontsize=8.4, fontweight="bold")
        ax.plot([x, x], [0.3, 7.0], color="#c9cdd4", lw=1.2, ls="--")
    def msg(x0, x1, y, text, c=INK, dy=0.22):
        _arrow(ax, (x0, y), (x1, y), color=c)
        ax.text((x0 + x1) / 2, y + dy, text, ha="center", fontsize=7.5, color=c)
    msg(2.0, 12.0, 6.2, "isend(job)   [t_dispatch]", BLUE)
    ax.text(12.15, 5.75, "iprobe poll every 200 us\n[t_worker_recv]", fontsize=7.4, color=BLUE)
    msg(12.0, 17.6, 4.9, "POST /v1/completions (stream)", INK)
    msg(17.6, 12.0, 3.9, "SSE tokens (first token, ..., done)", INK)
    msg(12.0, 2.0, 2.7, "isend(result)   [t_worker_send]", AQUA)
    ax.text(2.15, 2.25, "iprobe drain [t_dispatcher_recv]", fontsize=7.4, color=AQUA)
    ax.annotate("", xy=(0.7, 6.2), xytext=(0.7, 2.7), arrowprops=dict(arrowstyle="<->", color=GREY))
    ax.text(0.55, 4.45, "request lifetime", rotation=90, ha="right", va="center", fontsize=7.5, color=GREY)
    ax.text(10.0, 0.45, "dispatch overhead = t_worker_recv - t_dispatch;  result overhead = t_dispatcher_recv - t_worker_send  (one shared monotonic clock on one host)",
            ha="center", fontsize=7.6, color=INK)
    return _save(fig, out)


def fig_open_closed(out):
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 3.7), sharex=True)
    rng = np.random.default_rng(3)
    arr = np.cumsum(rng.exponential(1.0, 14))
    for ax, title in zip(axes, ("Open loop (used here): arrivals follow the schedule regardless of completions", "Closed loop: next request waits for the previous response")):
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=8.6, loc="left")
    for t in arr:
        axes[0].plot([t, t], [0.15, 0.7], color=BLUE, lw=1.6)
    axes[0].text(arr[-1] + 0.3, 0.4, "queue can build\nwhen the server is slow", fontsize=7.6, color=RED)
    t = 0.6
    for i in range(6):
        d = 1.0 + 0.7 * (i % 3)
        axes[1].add_patch(Rectangle((t, 0.2), d, 0.5, fc=LIGHT["aqua"], ec=AQUA))
        t += d + 0.25
    axes[1].text(t + 0.1, 0.4, "offered load silently drops\nas latency rises: queueing hides", fontsize=7.6, color=RED)
    axes[1].set_xlabel("time")
    fig.tight_layout()
    return _save(fig, out)


def build_all(outdir):
    outdir = Path(outdir)
    return {
        "architecture": fig_architecture(outdir / "architecture.png"),
        "roofline": fig_phases_roofline(outdir / "roofline.png"),
        "layout": fig_layout(outdir / "layout.png"),
        "access": fig_access_pattern(outdir / "access.png"),
        "banks": fig_bank_conflicts(outdir / "banks.png"),
        "controller": fig_controller(outdir / "controller.png"),
        "pipeline": fig_pipeline(outdir / "pipeline.png"),
        "paired": fig_paired(outdir / "paired.png"),
        "timeslice": fig_timeslice(outdir / "timeslice.png"),
        "mpipath": fig_mpi_path(outdir / "mpipath.png"),
        "openclosed": fig_open_closed(outdir / "openclosed.png"),
    }


if __name__ == "__main__":
    import sys

    for k, v in build_all(sys.argv[1] if len(sys.argv) > 1 else "build/figs").items():
        print(k, v)
