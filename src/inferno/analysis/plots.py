"""
Publication-quality figures for the report / README / social posts.

Conventions (from the data-viz method used for this repo):
  * quantization level is the *entity*, so it keeps a fixed categorical hue:
    fp16 = blue, int8 = orange, int4 = aqua (slots 1-3, validated for scatter);
  * the adaptive policy is drawn in ink (near-black), not a series hue, so the
    comparison the whole project is about is unmistakable;
  * one y-axis per chart, thin marks, recessive grid, direct labels, error bars
    are 95% CIs over repetitions;
  * light and dark variants share the same hues (dark stepped for the dark surface).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from .aggregate import ADAPTIVE_LABEL
from .stats import compute_ci

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", fp16="#2a78d6", int8="#eb6834", int4="#1baf7a",
                  ladder=["#a9caf3", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0a2b57"], neutral="#b3b1aa"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", fp16="#3987e5", int8="#d95926", int4="#199e70",
                 ladder=["#256abf", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6", "#d3e4f9", "#e8f1fc"], neutral="#5c5b56"),
}
_FONT = ["Segoe UI", "DejaVu Sans", "Arial"]


def _apply(theme: str):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": _FONT, "font.size": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "xtick.color": t["muted"],
        "ytick.color": t["muted"], "text.color": t["ink"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
    })
    return t


def _quant_of(config: str) -> str:
    return config.split("-")[0] if config != ADAPTIVE_LABEL else "adaptive"


def _colors(configs, t) -> dict:
    """Stable color per config. Adaptive = ink. Configs named '<quant>-...' keep that quantization hue;
    any other fixed policy takes the next free categorical slot (fp16/int8/int4 hues = slots 1-3)."""
    slots = ["fp16", "int8", "int4"]
    out, free = {}, [q for q in slots]
    named = {c: _quant_of(c) for c in configs if _quant_of(c) in slots}
    free = [q for q in slots if q not in named.values()]
    for c in sorted(configs):
        if c == ADAPTIVE_LABEL:
            out[c] = t["ink"]
        elif c in named:
            out[c] = t[named[c]]
        else:
            out[c] = t[free.pop(0)] if free else t["neutral"]
    return out


def _bs_of(config: str) -> int | None:
    return int(config.split("bs")[-1]) if "bs" in config else None


def _mean_ci(series: pd.Series) -> tuple[float, float]:
    vals = series.dropna().tolist()
    if len(vals) >= 2:
        m, lo, hi = compute_ci(vals)
        return m, (hi - lo) / 2
    return (vals[0] if vals else float("nan")), 0.0


def _save(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(df: pd.DataFrame, pattern: str, save_path: str | None = None, theme: str = "light",
                metric: str = "e2e_p99", title: str | None = None) -> None:
    """Throughput (x) vs p99 latency (y) per config for one load pattern; up-left is not the goal,
    down-right is: lower latency at higher throughput. Error bars are 95% CIs."""
    t = _apply(theme)
    sub = df[df["pattern"] == pattern]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    colors = _colors(sub["config"].unique(), t)
    for config, g in sub.groupby("config"):
        x, xe = _mean_ci(g["throughput_tok_per_s"])
        y, ye = _mean_ci(g[metric])
        if config == ADAPTIVE_LABEL:
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="D", ms=11, color=t["ink"], ecolor=t["ink"],
                        elinewidth=1.4, capsize=3, mec=t["surface"], mew=2, zorder=5)
            ax.annotate("adaptive", (x, y), textcoords="offset points", xytext=(10, 8),
                        color=t["ink"], fontsize=10, fontweight="bold")
        else:
            bs = _bs_of(config)
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=5 + 1.5 * np.log2(max(bs or 16, 1)),
                        color=colors[config], ecolor=colors[config], elinewidth=1, capsize=2,
                        mec=t["surface"], mew=2, zorder=3)
            ax.annotate(f"bs{bs}" if bs else config, (x, y), textcoords="offset points", xytext=(8, -12),
                        color=t["ink2"], fontsize=8.5)
    handles = [plt.Line2D([], [], marker="D" if c == ADAPTIVE_LABEL else "o", ls="", color=colors[c], label=c,
                          mec=t["surface"], mew=1.5) for c in sorted(colors)]
    ax.legend(handles=handles, loc="upper right", labelcolor=t["ink2"], fontsize=9)
    ax.set_xlabel("Throughput (output tokens/s)  → higher is better")
    ax.set_ylabel("p99 end-to-end latency (s)  ↓ lower is better")
    ax.set_title(title or f"Latency vs throughput, {pattern} load", loc="left", fontsize=12,
                 fontweight="bold", color=t["ink"], pad=24)
    ax.text(0, 1.02, "Points: mean of repetitions; bars: 95% CI",
            transform=ax.transAxes, fontsize=8.5, color=t["muted"], va="bottom")
    _save(fig, save_path)


def plot_p99_by_pattern(df: pd.DataFrame, save_path: str | None = None, theme: str = "light",
                        metric: str = "e2e_p99") -> None:
    """Small multiples, one panel per load pattern: horizontal bars of mean p99 (95% CI whiskers),
    sorted best to worst so the strongest fixed config and the adaptive policy are directly comparable."""
    t = _apply(theme)
    patterns = sorted(df["pattern"].unique())
    colors = _colors(df["config"].unique(), t)
    fig, axes = plt.subplots(1, len(patterns), figsize=(4.2 * len(patterns), 4.0), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, pattern in zip(axes, patterns):
        rows = []
        for config, g in df[df["pattern"] == pattern].groupby("config"):
            m, e = _mean_ci(g[metric])
            rows.append((config, m, e))
        rows.sort(key=lambda r: r[1])
        for i, (config, m, e) in enumerate(rows):
            ax.barh(i, m, height=0.62, color=colors[config], zorder=3)
            ax.errorbar(m, i, xerr=e, fmt="none", ecolor=t["ink2"], elinewidth=1, capsize=2.5, zorder=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=t["ink2"])
        ax.invert_yaxis()
        ax.set_title(pattern, loc="left", fontsize=11, fontweight="bold", color=t["ink"])
        ax.set_xlabel("p99 latency (s), lower is better")
        ax.grid(axis="y", visible=False)
    fig.suptitle("p99 end-to-end latency by configuration and load pattern (mean ± 95% CI)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)


def plot_kernel_speedup(bench_json: str | Path, save_path: str | None = None, theme: str = "light") -> None:
    """Custom-kernel results: speedup over cuBLAS fp16 GEMV per layer shape. Time-based, so kernels that
    move different byte counts are compared fairly. The optimization ladder v0..v3 is one ordinal hue."""
    t = _apply(theme)
    data = json.loads(Path(bench_json).read_text())
    order = ["cublas_fp16", "unfused_naive", "int4_v0_naive", "int4_v1_warp", "int4_v2_vec", "int4_v3_pad", "int4_v4_alg", "int4_v5_reuse", "int4_v6_lane16"]
    colors = {"cublas_fp16": t["neutral"], "unfused_naive": t["neutral"], **dict(zip(order[2:], t["ladder"]))}
    shapes = data["shapes"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.092
    x = np.arange(len(shapes))
    for j, name in enumerate(order):
        ys = []
        for sh in shapes:
            k = {k["kernel"]: k for k in sh["kernels"]}
            ys.append(k["cublas_fp16"]["us_median"] / k[name]["us_median"])
        ax.bar(x + (j - (len(order) - 1) / 2) * width, ys, width * 0.86, color=colors[name], zorder=3,
               hatch="//" if name == "unfused_naive" else None, edgecolor=t["surface"], linewidth=0.8,
               label=name)
    ax.axhline(1.0, color=t["ink2"], lw=1, zorder=4)
    # Byte-ratio ceiling: fp16 bytes / int4 bytes (int4 also reads fp16 scales and zeros), median over shapes.
    ceils = sorted((s["fp16_bytes"] + 2 * s["K"] + 2 * s["N"]) / (s["int4_bytes"] + 2 * s["K"] + 2 * s["N"])
                   for s in shapes)
    ceiling = ceils[len(ceils) // 2]
    ax.axhline(ceiling, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(len(shapes) - 0.5, ceiling + 0.04, f"byte-ratio ceiling {ceiling:.2f}× (int4 also reads scales and zeros)",
            ha="right", fontsize=8.5, color=t["ink2"])
    ax.set_ylim(0, ceiling * 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels([s["name"].split("(")[0].strip() + f"\n{s['N']}×{s['K']}" for s in shapes],
                       fontsize=8.5)
    ax.set_ylabel("Speedup vs cuBLAS fp16 GEMV (×)")
    ax.legend(ncol=3, loc="upper left", fontsize=8.5, labelcolor=t["ink2"])
    ax.set_title("Fused int4 GEMV: speedup over fp16 cuBLAS (batch-1 decode)", loc="left", fontsize=12,
                 fontweight="bold", color=t["ink"])
    _save(fig, save_path)


def plot_kernel_bandwidth(bench_json: str | Path, save_path: str | None = None, theme: str = "light") -> None:
    """Achieved DRAM bandwidth (% of theoretical peak) for the int4 kernel ladder: how close each
    optimization stage gets to the memory-bound roofline."""
    t = _apply(theme)
    data = json.loads(Path(bench_json).read_text())
    names = ["int4_v0_naive", "int4_v1_warp", "int4_v2_vec", "int4_v3_pad", "int4_v4_alg", "int4_v5_reuse", "int4_v6_lane16"]
    shapes = data["shapes"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.118
    x = np.arange(len(shapes))
    for j, name in enumerate(names):
        ys = [{k["kernel"]: k for k in sh["kernels"]}[name]["pct_peak"] for sh in shapes]
        ax.bar(x + (j - (len(names) - 1) / 2) * width, ys, width * 0.86, color=t["ladder"][j], zorder=3,
               edgecolor=t["surface"], linewidth=0.8, label=name)
    ax.axhline(100, color=t["ink2"], lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(len(shapes) - 0.5, 101.5, f"theoretical peak {data['peak_dram_gbps']:.0f} GB/s", ha="right",
            fontsize=8.5, color=t["ink2"])
    ax.set_xticks(x)
    ax.set_xticklabels([s["name"].split("(")[0].strip() for s in shapes], fontsize=8.5)
    ax.set_ylabel("Achieved DRAM bandwidth (% of peak)")
    ax.set_ylim(0, 115)
    ax.legend(ncol=7, loc="upper left", fontsize=7.4, labelcolor=t["ink2"])
    ax.set_title("How close each optimization stage gets to the memory roofline", loc="left",
                 fontsize=12, fontweight="bold", color=t["ink"])
    _save(fig, save_path)


def plot_scaling(rows: list[dict], save_path: str | None = None, theme: str = "light") -> None:
    """Replica scaling: aggregate throughput vs replica count for each router, against ideal linear."""
    t = _apply(theme)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    routers = sorted({r["router"] for r in rows})
    palette = {"mpi": t["fp16"], "http": t["int8"]}
    for router in routers:
        pts = sorted((r["replicas"], r["throughput"], r.get("ci", 0.0)) for r in rows if r["router"] == router)
        xs, ys, es = zip(*pts)
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=7, lw=1.8, color=palette.get(router, t["int4"]),
                    mec=t["surface"], mew=2, capsize=3, label=f"{router} router")
    base = [r for r in rows if r["replicas"] == 1]
    if base:
        b = np.mean([r["throughput"] for r in base])
        ns = sorted({r["replicas"] for r in rows})
        ax.plot(ns, [b * n for n in ns], ls=(0, (4, 3)), color=t["muted"], lw=1.2, label="ideal linear")
    ax.set_xlabel("Replicas")
    ax.set_ylabel("Aggregate throughput (tokens/s)")
    ax.set_xticks(sorted({r["replicas"] for r in rows}))
    ax.legend(loc="upper left", labelcolor=t["ink2"])
    ax.set_title("Replica scaling", loc="left", fontsize=12, fontweight="bold", color=t["ink"])
    _save(fig, save_path)


def plot_contention(cells: list[dict], save_path: str | None = None, theme: str = "light") -> None:
    """p99 latency by load level for the three GPU-sharing conditions (grouped bars, 95% CI whiskers)."""
    t = _apply(theme)
    order = ["single", "split-idle", "split-active"]
    colors = {"single": t["fp16"], "split-idle": t["int8"], "split-active": t["int4"]}
    patterns = sorted({c["pattern"] for c in cells})
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    width = 0.24
    for j, cond in enumerate(order):
        ys, es = [], []
        for p in patterns:
            c = next((c for c in cells if c["pattern"] == p and c["config"] == cond), None)
            ys.append(c["e2e_p99"] if c else float("nan"))
            es.append(c["e2e_p99_ci"] if c else 0.0)
        xs = np.arange(len(patterns)) + (j - 1) * width
        ax.bar(xs, ys, width * 0.88, color=colors[cond], zorder=3, label=cond, edgecolor=t["surface"], linewidth=0.8)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=t["ink2"], elinewidth=1, capsize=3, zorder=4)
        for x, y, e in zip(xs, ys, es):
            ax.text(x, y + e + 0.2, f"{y:.1f}", ha="center", fontsize=8.5, color=t["ink2"])
    ax.set_xticks(np.arange(len(patterns)))
    ax.set_xticklabels([p.replace("rho", "load = ") + " x capacity" for p in patterns])
    ax.set_ylabel("p99 end-to-end latency (s), lower is better")
    ax.legend(loc="upper left", labelcolor=t["ink2"])
    ax.set_title("A second active replica on the same GPU doubles p99", loc="left", fontsize=12,
                 fontweight="bold", color=t["ink"])
    ax.grid(axis="x", visible=False)
    _save(fig, save_path)


def plot_sweep(df: pd.DataFrame, pattern: str = "steady", save_path: str | None = None, theme: str = "light") -> None:
    """Batching x quantization characterization: four small multiples against the batch cap (max_num_seqs), one
    line per quantization level (fixed hues), 95% CI whiskers. Completion rate shows configurations whose capacity is
    below the offered load (unfinished requests are counted as failures by the drain timeout)."""
    t = _apply(theme)
    sub = df[df["pattern"] == pattern].copy()
    sub["quant"] = sub["config"].map(_quant_of)
    sub["bs"] = sub["config"].map(_bs_of)
    sub["completed"] = sub["n_ok"] / sub["n_requests"]
    panels = [("throughput_tok_per_s", "Throughput (tokens/s)"), ("tokens_per_joule", "Energy efficiency (tokens/J)"),
              ("e2e_p50", "p50 latency of completed requests (s)"), ("completed", "Requests completed (fraction)")]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), panels):
        for q in ("fp16", "int8", "int4"):
            g = sub[sub["quant"] == q]
            if g.empty:
                continue
            pts = sorted((bs, *_mean_ci(gg[metric])) for bs, gg in g.groupby("bs"))
            xs, ys, es = zip(*pts)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=6, lw=1.8, color=t[q], ecolor=t[q], capsize=2.5,
                        mec=t["surface"], mew=1.5, label=q)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub["bs"].dropna().unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Batch cap (max concurrent sequences)")
    axes[0, 0].legend(title="quantization", loc="upper left", labelcolor=t["ink2"], fontsize=9, title_fontsize=9)
    fig.suptitle(f"Batching x quantization sweep, {pattern} load (mean ± 95% CI)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, save_path)


# Backwards-compatible aliases for the original scaffold names.
def plot_latency_throughput_curve(df: pd.DataFrame, save_path: str | None = None) -> None:
    plot_pareto(df, df["pattern"].iloc[0], save_path)


def plot_percentile_comparison(df: pd.DataFrame, save_path: str | None = None) -> None:
    plot_p99_by_pattern(df, save_path)
