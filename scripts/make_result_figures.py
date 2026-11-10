#!/usr/bin/env python
"""
Result figures for the layer matrix and the Marlin correctness probe, from saved data only (no GPU).

  docs/figures/layer_matrix.png      us per call vs batch, one panel per Qwen2.5-1.5B layer, four implementations
  docs/figures/custom_vs_marlin.png  custom/Marlin time ratio per (layer, batch): where the custom kernel earns its place
  docs/figures/marlin_fault.png      Marlin calls with wrong results (of 40 identical calls) by batch size and layer

    python scripts/make_result_figures.py      (also run by `make figure`)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
# categorical slots 1-4 of the validated default palette, fixed per implementation
IMPL = {"custom": ("#2a78d6", "o", "custom kernel"), "marlin": ("#eb6834", "s", "Marlin (vLLM default)"),
        "awq": ("#1baf7a", "^", "classic AWQ"), "fp16": ("#eda100", "D", "fp16 cuBLAS")}
SHAPE_TITLE = {"qkv": "qkv", "o_proj": "o_proj", "gate_up": "gate_up", "down": "down"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def layer_matrix(lm, faulty):
    shapes = list(lm["shapes"])
    B = lm["batches"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), dpi=170)
    axes = axes.ravel()
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes, shapes):
        s = lm["shapes"][name]
        _style(ax)
        for impl, (col, mk, _) in IMPL.items():
            xs = [b for b in B if s["cells"].get(f"{impl}/B{b}")]
            ys = [s["cells"][f"{impl}/B{b}"]["us_median"] for b in xs]
            ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=5, zorder=3, label=IMPL[impl][2])
            if impl == "marlin":                     # hollow marks where Marlin's output was wrong
                bad = [(x, y) for x, y in zip(xs, ys) if x in faulty]
                if bad:
                    ax.scatter(*zip(*bad), s=70, facecolors=SURFACE, edgecolors=col, linewidths=1.6, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(B)
        ax.set_xticklabels([str(b) for b in B])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{SHAPE_TITLE[name]}  ({s['N']} x {s['K']})", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("batch (tokens per step)", fontsize=8, color=MUTED)
    for ax in axes[::2]:
        ax.set_ylabel("us per call (median)", fontsize=8, color=MUTED)
    h, l_ = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=IMPL["marlin"][0], marker="o", ls="", mfc=SURFACE, ms=7, mew=1.6))
    l_.append("Marlin, batch with wrong results (see fault figure)")
    leg = fig.legend(h, l_, loc="upper left", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.0, 1.06))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("Marlin stays flat to batch 16; the custom kernel's time grows with batch",
                 x=0.0, y=1.11, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.0, -0.035, f"{lm['gpu']}, vLLM {lm['vllm']}. Same AWQ weights for every implementation; CUDA graph, weights "
             f"rotated over >= {lm['rotate_mb']:.0f} MB; median of {lm['trials']} trials.\n"
             "Custom kernel instantiated up to batch 16. "
             "Source: docs/results/layer_matrix/layer_matrix.json", fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, "layer_matrix.png")


def ratio_heatmap(lm):
    shapes = list(lm["shapes"])
    B = [b for b in lm["batches"] if any(lm["shapes"][n]["cells"].get(f"custom/B{b}") for n in shapes)]
    M = np.array([[lm["shapes"][n]["cells"][f"custom/B{b}"]["us_median"] / lm["shapes"][n]["cells"][f"marlin/B{b}"]["us_median"]
                   for b in B] for n in shapes])
    marlin_b1 = {n: lm["shapes"][n]["cells"]["marlin/B1"]["us_median"] for n in shapes}
    tot = sum(marlin_b1.values())
    # diverging: blue = custom faster, neutral grey at parity, orange = custom slower (log scale around 1)
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#e8e7e2", "#f2a07d", "#b2461b"])
    L = np.log2(M)
    lim = max(abs(L.min()), abs(L.max()))
    fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(L, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(shapes)):
        for j in range(len(B)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(L[i, j]) > 0.6 * lim else INK, fontweight="bold" if v < 1 else "normal")
    ax.set_xticks(range(len(B)))
    ax.set_xticklabels([f"B={b}" for b in B], fontsize=8, color=INK)
    ax.set_yticks(range(len(shapes)))
    ax.set_yticklabels([f"{n} ({100 * marlin_b1[n] / tot:.0f}%)" for n in shapes], fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(B), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(shapes), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Custom kernel time / Marlin time (below 1 = custom faster)", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK)
    fig.text(0.0, -0.02, "Blue: custom faster; grey: parity; orange: Marlin faster. (%) = share of quantized-GEMM\n"
             "time per decode step at batch 1. Source: docs/results/layer_matrix/layer_matrix.json",
             fontsize=6.8, color=MUTED)
    fig.tight_layout()
    _save(fig, "custom_vs_marlin.png")


def fault(mf):
    shapes = mf["shapes"]
    B = mf["batches"]
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    hatches = ["", "//", ".."]
    w = 0.26
    for i, s in enumerate(shapes):
        ys = [next(r["bad"] for r in s["rows"] if r["B"] == b and r["variant"] == "plain") for b in B]
        xs = np.arange(len(B)) + (i - 1) * (w + 0.02)
        ax.bar(xs, ys, width=w, color=cols[i], hatch=hatches[i], edgecolor=SURFACE, linewidth=0, zorder=2,
               label=f"{s['name']} ({s['N']} x {s['K']})")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.6, str(y), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(B)))
    ax.set_xticklabels([f"B={b}" for b in B], color=INK)
    ax.set_ylim(0, 42)
    ax.set_ylabel("calls with wrong output (of 40)", fontsize=8, color=MUTED)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    ok = [b for b in B if b not in mf["faulty_batches"]]
    ax.set_title(f"Marlin returns wrong results at batch {', '.join(map(str, mf['faulty_batches']))}; "
                 f"never at {', '.join(map(str, ok))}",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.0, -0.03, f"{mf['gpu']}, vLLM 0.11.0 gptq_marlin_gemm, AWQ uint4, 40 identical calls per bar (shared "
             "workspace, as vLLM runs it); wrong = relative max error > 1e-2 vs dequantize + fp32 matmul. "
             f"Worst error {mf['worst_rel_err_faulty']:.2f}. Observed on one machine only. "
             "Source: docs/results/marlin_fault/summary.json", fontsize=6.6, color=MUTED, wrap=True)
    fig.tight_layout()
    _save(fig, "marlin_fault.png")


def main() -> None:
    lm_p, mf_p = R / "layer_matrix" / "layer_matrix.json", R / "marlin_fault" / "summary.json"
    mf = json.loads(mf_p.read_text()) if mf_p.exists() else None
    if lm_p.exists():
        lm = json.loads(lm_p.read_text())
        layer_matrix(lm, set(mf["faulty_batches"]) if mf else set())
        ratio_heatmap(lm)
    if mf:
        fault(mf)


if __name__ == "__main__":
    main()
