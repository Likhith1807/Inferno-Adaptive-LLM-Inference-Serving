"""
Aggregate the end-to-end kernel matrix (results/e2e_kernel/<name>_rep<r>.json) into a summary, a markdown table and a
figure, and check that the custom kernel produced the same text as the stock kernels.

Statistics: every number is the mean of per-repetition medians over fresh processes (n = number of repetitions,
default 3), with a t-based 95% CI. With n = 3 the CI is wide and no p-value is computed; the comparison that matters
is the paired ratio to the reference configuration (same repetition, same thermal epoch), reported as mean and range.

    python -m inferno.experiments.report_e2e results/e2e_kernel docs/results/e2e_kernel
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats

LABEL = {"fp16": "fp16 (no quantization)", "awq_marlin": "AWQ-int4, vLLM Marlin (default)", "awq_classic": "AWQ-int4, vLLM classic kernel",
         "inferno_b8": "AWQ-int4, custom kernel (<= 8 tokens)", "inferno_b2": "AWQ-int4, custom kernel (<= 2 tokens)"}
ORDER = ["fp16", "awq_classic", "awq_marlin", "inferno_b2", "inferno_b8"]
REFERENCE = "awq_marlin"


def load(src: Path) -> dict:
    runs: dict = {}
    for p in sorted(src.glob("*_rep*.json")):
        d = json.loads(p.read_text())
        name, rep = p.stem.rsplit("_rep", 1)
        runs.setdefault(name, {})[int(rep)] = d
    return runs


def mean_ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1))


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def summarize(runs: dict) -> dict:
    conc = sorted({int(c) for r in runs.values() for d in r.values() for c in d["results"]}, key=int)
    out = {"concurrency": conc, "configs": {}, "reference": REFERENCE}
    for name in ORDER:
        if name not in runs:
            continue
        reps = runs[name]
        entry = {"label": LABEL[name], "n_reps": len(reps), "by_concurrency": {}}
        for c in conc:
            vals = [reps[r]["results"][str(c)]["tok_per_s_median"] for r in sorted(reps)]
            m, h = mean_ci(vals)
            entry["by_concurrency"][str(c)] = {"mean": m, "ci95": h, "per_rep": vals}
        first = reps[min(reps)]
        entry["kernel_calls"] = first.get("kernel_calls")
        entry["layer_check"] = first.get("layer_check")
        entry["startup_sec"] = float(np.median([d["startup_sec"] for d in reps.values()]))
        out["configs"][name] = entry
    ref = out["configs"].get(REFERENCE)
    if ref:
        for name, e in out["configs"].items():
            for c in conc:
                a, b = e["by_concurrency"][str(c)]["per_rep"], ref["by_concurrency"][str(c)]["per_rep"]
                n = min(len(a), len(b))
                ratios = [a[i] / b[i] for i in range(n)]
                e["by_concurrency"][str(c)]["ratio_to_reference"] = {"mean": float(np.mean(ratios)), "min": float(min(ratios)),
                                                                     "max": float(max(ratios))}
    # text agreement of the custom kernel with the stock kernels (greedy decoding, first prompt)
    agree = {}
    for name in ("inferno_b8", "inferno_b2"):
        for other in ("awq_classic", "awq_marlin"):
            if name in runs and other in runs:
                a = runs[name][min(runs[name])]["first_prompt_token_ids"]
                b = runs[other][min(runs[other])]["first_prompt_token_ids"]
                agree[f"{name} vs {other}"] = {"tokens": len(a), "identical_prefix": common_prefix(a, b), "identical_total": int(a == b)}
    if "awq_classic" in runs and "awq_marlin" in runs:
        a = runs["awq_classic"][min(runs["awq_classic"])]["first_prompt_token_ids"]
        b = runs["awq_marlin"][min(runs["awq_marlin"])]["first_prompt_token_ids"]
        agree["awq_classic vs awq_marlin (stock vs stock)"] = {"tokens": len(a), "identical_prefix": common_prefix(a, b), "identical_total": int(a == b)}
    out["text_agreement"] = agree
    return out


def markdown(s: dict) -> str:
    conc = s["concurrency"]
    L = ["| configuration | " + " | ".join(f"{c} seq" for c in conc) + " |", "|---|" + "---|" * len(conc)]
    for name, e in s["configs"].items():
        cells = []
        for c in conc:
            b = e["by_concurrency"][str(c)]
            r = b.get("ratio_to_reference")
            cells.append(f"{b['mean']:.0f} &plusmn; {b['ci95']:.0f}" + (f" ({r['mean']:.2f}x)" if r and name != s["reference"] else ""))
        L.append(f"| {e['label']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(f"Output tokens/s (mean of {next(iter(s['configs'].values()))['n_reps']} process repetitions, +/- 95% CI); parentheses: paired ratio to {LABEL[s['reference']]}.")
    return "\n".join(L)


def figure(s: dict, path: Path, theme: str = "light") -> None:
    import matplotlib.pyplot as plt

    from ..analysis import plots

    t = plots._apply(theme)
    conc = s["concurrency"]
    names = list(s["configs"])
    colors = {"fp16": t["fp16"], "awq_classic": t["int8"], "awq_marlin": t["int4"], "inferno_b2": t["ladder"][3], "inferno_b8": t["ladder"][4]}
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w = 0.8 / len(names)
    x = np.arange(len(conc))
    for j, n in enumerate(names):
        e = s["configs"][n]["by_concurrency"]
        ax.bar(x + (j - (len(names) - 1) / 2) * w, [e[str(c)]["mean"] for c in conc], w * 0.88,
               yerr=[e[str(c)]["ci95"] for c in conc], color=colors.get(n, t["neutral"]), zorder=3,
               error_kw=dict(ecolor=t["ink2"], lw=0.9, capsize=2), label=LABEL[n])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} concurrent" if c > 1 else "1 sequence" for c in conc])
    ax.set_ylabel("Decode throughput (output tokens/s)")
    ax.set_title("End-to-end decode in vLLM 0.11 (CUDA graphs), Qwen2.5-1.5B on an RTX 4060 Laptop", loc="left", fontsize=11,
                 fontweight="bold", color=t["ink"])
    ax.legend(loc="upper left", fontsize=8.2, labelcolor=t["ink2"])
    plots._save(fig, str(path))


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/e2e_kernel")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/results/e2e_kernel")
    dst.mkdir(parents=True, exist_ok=True)
    runs = load(src)
    if not runs:
        raise SystemExit(f"no *_rep*.json files in {src}")
    for p in src.glob("*_rep*.json"):
        shutil.copy(p, dst / p.name)
    s = summarize(runs)
    (dst / "summary.json").write_text(json.dumps(s, indent=2))
    (dst / "summary.md").write_text(markdown(s) + "\n")
    figure(s, dst / "e2e_light.png", "light")
    figure(s, dst / "e2e_dark.png", "dark")
    print(markdown(s))
    print(json.dumps(s["text_agreement"], indent=2))


if __name__ == "__main__":
    main()
