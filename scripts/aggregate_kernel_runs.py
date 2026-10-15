#!/usr/bin/env python
"""
Merge repeated kernel-benchmark invocations into one result with run-to-run spread.

A single invocation reports the median of many back-to-back batches, but on a laptop GPU whole invocations can
differ (thermal state, clocks). Repeating the invocation and taking the median of run medians is the honest
number; the spread across runs is reported so a reader can see how much to trust it.

    bash scripts/repeat_kernel_bench.sh 5          # writes results/kernel_bench_runs/run_*.json, then calls this
    python scripts/aggregate_kernel_runs.py results/kernel_bench_runs results/kernel_bench.json

Output has the same schema as a single run plus, per kernel: n_runs, run_min_us, run_max_us, run_spread_pct.
"""
import json
import statistics
import sys
from pathlib import Path


def aggregate(run_files: list[Path]) -> dict:
    runs = [json.loads(p.read_text()) for p in run_files]
    if not runs:
        raise SystemExit("no run files found")
    out = {k: v for k, v in runs[0].items() if k != "shapes"}
    out["n_runs"] = len(runs)
    if all("stream_read_gbps" in r for r in runs):
        out["stream_read_gbps"] = statistics.median(r["stream_read_gbps"] for r in runs)
    out["shapes"] = []
    for si, shape in enumerate(runs[0]["shapes"]):
        merged = {k: v for k, v in shape.items() if k != "kernels"}
        merged["kernels"] = []
        for ki, kern in enumerate(shape["kernels"]):
            same = [r["shapes"][si]["kernels"][ki] for r in runs]
            assert all(s["kernel"] == kern["kernel"] for s in same), "kernel order differs between runs"
            med = [s["us_median"] for s in same]
            m = statistics.median(med)
            peak = out["peak_dram_gbps"]
            bytes_moved = kern["gbps"] * 1e9 * kern["us_median"] * 1e-6     # bytes per launch (unchanged across runs)
            gbps = bytes_moved / (m * 1e-6) / 1e9
            merged["kernels"].append({
                "kernel": kern["kernel"], "us_median": m,
                "us_p10": statistics.median(s["us_p10"] for s in same),
                "us_p90": statistics.median(s["us_p90"] for s in same),
                "gbps": gbps, "pct_peak": 100 * gbps / peak,
                "max_rel_err": max(s["max_rel_err"] for s in same),
                "correct": all(s["correct"] for s in same),
                "n_runs": len(same), "run_min_us": min(med), "run_max_us": max(med),
                "run_spread_pct": 100 * (max(med) - min(med)) / m,
                "runs_within_2pct": sum(1 for v in med if abs(v - m) <= 0.02 * m),
            })
        out["shapes"].append(merged)
    return out


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/kernel_bench_runs")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "results/kernel_bench.json")
    result = aggregate(sorted(src.glob("run_*.json")))
    dst.write_text(json.dumps(result, indent=2))
    worst = max((k["run_spread_pct"], sh["name"].split("(")[0].strip(), k["kernel"])
                for sh in result["shapes"] for k in sh["kernels"])
    print(f"aggregated {result['n_runs']} runs -> {dst}; largest run-to-run spread {worst[0]:.1f}% ({worst[2]} on {worst[1]})")


if __name__ == "__main__":
    main()
