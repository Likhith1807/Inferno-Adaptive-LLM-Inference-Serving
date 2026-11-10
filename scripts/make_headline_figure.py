#!/usr/bin/env python
"""
Regenerate the headline figure from saved data only (no GPU):

  p99 end-to-end latency for one server, two servers with the second idle, and two servers both serving, on the
  same GPU at equal total load (docs/results/contention/contention_summary.json; mean +- 95% CI over paired reps).

    python scripts/make_headline_figure.py          ->  docs/figures/headline_{light,dark}.png
    make figure
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "results" / "contention" / "contention_summary.json"
OUT = ROOT / "docs" / "figures"
CONFIGS = [("single", "one server"), ("split-idle", "two servers, second idle"), ("split-active", "two servers, both serving")]
THEMES = {  # categorical slots 1-3 of the validated default palette, stepped per mode
    "light": {"surface": "#fcfcfb", "ink": "#1f1f1e", "muted": "#6b6a64", "grid": "#e6e5e0",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#33332f",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}


def main() -> None:
    d = json.loads(SRC.read_text())
    cells = {(c["pattern"], c["config"]): c for c in d["cells"]}
    patterns = sorted({c["pattern"] for c in d["cells"]})
    order = ["single", "split-idle", "split-active"]
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        w, gap = 0.26, 0.02
        for i, (cfg, label) in enumerate(CONFIGS):
            xs = [p_i + (i - 1) * (w + gap) for p_i in range(len(patterns))]
            ys = [cells[(p, cfg)]["e2e_p99"] for p in patterns]
            es = [cells[(p, cfg)]["e2e_p99_ci"] for p in patterns]
            ax.bar(xs, ys, width=w, color=t["series"][order.index(cfg)], label=label, zorder=2)
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=t["ink"], elinewidth=1, capsize=3, zorder=3)
            for x, y, e in zip(xs, ys, es):
                ax.text(x, y + e + 0.2, f"{y:.2f} s", ha="center", va="bottom", fontsize=8, color=t["ink"])
        n = min(c["n"] for c in d["cells"])
        ax.set_xticks(range(len(patterns)))
        ax.set_xticklabels([f"{float(p.replace('rho', '')):.0%} of calibrated capacity" for p in patterns], color=t["ink"])
        ax.set_ylabel("p99 end-to-end latency (s)", color=t["muted"])
        ax.tick_params(colors=t["muted"], length=0)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("A second replica costs nothing idle and doubles p99 once it serves",
                     loc="left", color=t["ink"], fontsize=11, fontweight="bold")
        fig.text(0.01, 0.01, f"Same GPU, equal total load, mean ± 95% CI over {n} paired repetitions. "
                 "Source: docs/results/contention/contention_summary.json", fontsize=7, color=t["muted"])
        leg = ax.legend(frameon=False, loc="upper left", fontsize=8)
        for txt in leg.get_texts():
            txt.set_color(t["ink"])
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p = OUT / f"headline_{mode}.png"
        fig.savefig(p, facecolor=t["surface"])
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
