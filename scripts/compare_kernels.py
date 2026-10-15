#!/usr/bin/env python
"""
Merge the standalone kernel benchmark with kernel times measured IN SITU inside vLLM decode, per Qwen layer role.

    python scripts/compare_kernels.py

Three int4 paths are timed under identical conditions (same model, same process setup, CUDA-graph decode, torch
profiler): vLLM's Marlin kernel, vLLM's classic AWQ kernel, and the custom kernel patched into the AWQ layers. The
fair comparison is in situ against in situ. The standalone column (custom kernel in the isolated benchmark) is shown
for reference.

Ratios are time ratios: "custom / X" below 1 means the custom kernel is faster than X. The table states which way it
went, whichever way that is.

History: an earlier version of this comparison used a Marlin timing (about 2x too slow) that could not be reproduced.
Re-measuring all three paths in the same session, in both eager and CUDA-graph mode, gave the numbers used here; the
earlier Marlin file is not used.
"""
import argparse
import json
from pathlib import Path

ROLE_TO_SHAPE_PREFIX = {"qkv": "qwen1.5B_qkv", "o_proj": "qwen1.5B_o_proj", "gate_up": "qwen1.5B_gate_up",
                        "down": "qwen1.5B_down"}
CUSTOM_KERNELS = ("int4_v3_pad", "int4_v4_alg", "int4_v5_reuse", "int4_v6_lane16")
DEFAULT_LIBRARIES = ["vLLM Marlin=results/insitu/awq_marlin_graphs.json", "vLLM AWQ=results/insitu/awq_graphs.json"]
DEFAULT_CUSTOM_INSITU = "results/insitu/awq_inferno_graphs.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results/kernel_bench.json")
    ap.add_argument("--library", action="append", help='"Label=path" (repeatable)')
    ap.add_argument("--custom-insitu", default=DEFAULT_CUSTOM_INSITU)
    ap.add_argument("--out", default="results/kernel_comparison.md")
    args = ap.parse_args()

    bench = json.loads(Path(args.bench).read_text())
    libs = {}
    for spec in args.library or DEFAULT_LIBRARIES:
        label, path = spec.split("=", 1)
        if Path(path).exists():
            data = json.loads(Path(path).read_text())
            if data.get("roles"):
                libs[label] = data
            else:
                print(f"skipping {label}: {data.get('warning')}")
    if not libs:
        raise SystemExit("No library timings with per-role attribution available.")
    custom_in = json.loads(Path(args.custom_insitu).read_text()) if Path(args.custom_insitu).exists() else None
    if custom_in and not custom_in.get("roles"):
        custom_in = None

    header = ["layer", "shape", "cuBLAS fp16 (standalone) us", "custom standalone us", "best variant"]
    if custom_in:
        header.append("custom in situ us")
    header += [f"{l} in situ us" for l in libs]
    header += [f"custom / {l}" for l in libs] + ["custom standalone vs cuBLAS", "custom %peak BW"]
    mode = next(iter(libs.values())).get("mode", "eager")
    lines = [f"Device: {bench['device']} (theoretical DRAM peak {bench['peak_dram_gbps']:.0f} GB/s, measured streaming read "
             f"{bench.get('stream_read_gbps', float('nan')):.0f} GB/s), {bench.get('n_runs', 1)} standalone benchmark run(s). "
             f"In-situ times: torch profiler on vLLM decode ({mode} mode, Qwen2.5-1.5B AWQ, "
             f"{next(iter(libs.values()))['layers']} layers), median over decode steps, then median across "
             f"{next(iter(libs.values())).get('n_process_reps', 1)} fresh processes.", "",
             "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    verdict = {l: {"faster": 0, "on par": 0, "slower": 0} for l in libs}
    n_rows = 0
    for role, prefix in ROLE_TO_SHAPE_PREFIX.items():
        shape = next((s for s in bench["shapes"] if s["name"].startswith(prefix)), None)
        if shape is None:
            continue
        k = {x["kernel"]: x for x in shape["kernels"]}
        cands = [k[n] for n in CUSTOM_KERNELS if n in k and k[n]["correct"]]
        best = min(cands, key=lambda x: x["us_median"])
        # ratios use the in-situ custom time when available (like against like), else the standalone best
        custom_us = custom_in["roles"][role]["us_median"] if custom_in else best["us_median"]
        lib_us = {l: d["roles"][role]["us_median"] for l, d in libs.items() if role in d["roles"]}
        n_rows += 1
        cells = [role, f"{shape['N']}x{shape['K']}", f"{k['cublas_fp16']['us_median']:.2f}",
                 f"{best['us_median']:.2f}", best["kernel"].replace("int4_", "")]
        if custom_in:
            cells.append(f"{custom_us:.2f}")
        cells += [f"{lib_us[l]:.2f}" if l in lib_us else "n/a" for l in libs]
        for l in libs:
            if l in lib_us:
                ratio = custom_us / lib_us[l]
                verdict[l]["faster" if ratio < 0.98 else "slower" if ratio > 1.02 else "on par"] += 1
                cells.append(f"{ratio:.2f}x")
            else:
                cells.append("n/a")
        cells += [f"{k['cublas_fp16']['us_median'] / best['us_median']:.2f}x faster", f"{best['pct_peak']:.1f}%"]
        lines.append("| " + " | ".join(cells) + " |")
    basis = "in situ" if custom_in else "standalone"
    lines += [""] + [f"Custom kernel ({basis}) versus {l}: faster on {v['faster']}, on par on {v['on par']}, slower on {v['slower']} of {n_rows} layer shapes (on par = within 2%)." for l, v in verdict.items()]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
