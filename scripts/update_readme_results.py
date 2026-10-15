#!/usr/bin/env python
"""
Fill the README's RESULTS block from measured data only.

    python scripts/update_readme_results.py --run results/<campaign_dir> [--kernel results/kernel_bench.json]

Reads <run>/report/ (run `make report RUN=...` first, or this script builds it), copies the figures into
docs/figures/, and rewrites everything between the RESULTS markers. Refuses to write anything if the inputs
are missing, so the README can never show numbers that were not produced by a run.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inferno.analysis import plots  # noqa: E402
from inferno.experiments.report import build_report  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def render_block(run: Path, kernel_json: Path | None, fig_dir: Path, readme_dir: Path) -> str:
    report = run / "report"
    if not (report / "summary.json").exists():
        build_report(run)
    summary = json.loads((report / "summary.json").read_text())
    h1 = summary.get("hypothesis_h1", [])
    if not h1:
        raise SystemExit("No adaptive-vs-fixed comparison in this run; nothing to report for H1.")

    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"Measured on `{run.name}` (mean ± 95% CI, paired by repetition). Raw data and code: this repo.", "",
             "| Load pattern | Best fixed baseline | Adaptive p99 change | Holm p | Throughput vs best fixed | Verdict |",
             "|---|---|---|---|---|---|"]
    for r in h1:
        lines.append(f"| {r['pattern']} | {r['best_fixed']} | {r['pct_change']:+.1f}% | {r['p_holm']:.3g} | "
                     f"{r['throughput_ratio']:.2f}x | {r['verdict']} |")
    lines.append("")

    for src in sorted((report / "figures").glob("p99_by_pattern_light.png")):
        shutil.copy(src, fig_dir / src.name)
        lines += [f"![p99 latency by policy and load pattern]({(fig_dir / src.name).relative_to(readme_dir).as_posix()})", ""]

    if kernel_json and kernel_json.exists():
        for name, fn in (("kernel_speedup", plots.plot_kernel_speedup), ("kernel_bandwidth", plots.plot_kernel_bandwidth)):
            fn(kernel_json, fig_dir / f"{name}.png")
        data = json.loads(kernel_json.read_text())
        ok = all(k["correct"] for sh in data["shapes"] for k in sh["kernels"])
        lines += [f"**CUDA kernel** on {data['device']} (theoretical DRAM peak {data['peak_dram_gbps']:.0f} GB/s; "
                  f"all kernels {'match' if ok else 'DO NOT ALL MATCH'} the CPU reference):", "",
                  f"![kernel speedup]({(fig_dir / 'kernel_speedup.png').relative_to(readme_dir).as_posix()})",
                  f"![kernel bandwidth]({(fig_dir / 'kernel_bandwidth.png').relative_to(readme_dir).as_posix()})", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--kernel")
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--figures", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README is missing the {START} / {END} markers")
    block = render_block(Path(args.run), Path(args.kernel) if args.kernel else None,
                         Path(args.figures), readme.parent)
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")
    print(f"Updated {readme}")


if __name__ == "__main__":
    main()
