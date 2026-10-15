#!/usr/bin/env python
"""
Summarize the per-kernel Nsight Compute CSVs into one Markdown table, so each optimization stage is tied to
the counter that explains it (bandwidth %, bank conflicts, occupancy, instruction count).

    python scripts/summarize_ncu.py [results/ncu]

Reads only what ncu measured; missing metrics show as "n/a". Column meaning:
  time_us          kernel duration
  dram_%peak       DRAM throughput as % of peak (the memory-bound figure of merit)
  bank_conf_ratio  shared-load bank conflicts per shared-load wavefront (v2 should be high, v3 near zero)
  warps_active_%   achieved occupancy
  inst_M           instructions executed (v4 should drop vs v3: 1 cvt + 1 FMA per weight)
"""
import csv
import sys
from pathlib import Path

METRICS = {
    "gpu__time_duration.sum": "time_us",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_%peak",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warps_active_%",
    "smsp__inst_executed.sum": "inst_M",
    "launch__registers_per_thread": "regs/thread",
}
CONF, WAVE = ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
              "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum")


def to_float(x: str):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return None


def read_csv(path: Path) -> tuple[dict, dict]:
    """ncu --page raw --csv: a header row of metric names, a units row, then value rows.
    Returns (values by metric, units by metric)."""
    rows = list(csv.reader(path.read_text().splitlines()))
    header_i = next((i for i, r in enumerate(rows) if "Kernel Name" in r), None)
    if header_i is None or header_i + 2 >= len(rows):
        return {}, {}
    header = rows[header_i]
    return dict(zip(header, rows[header_i + 2])), dict(zip(header, rows[header_i + 1]))


_TO_US = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6}


def fmt(v, digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}"


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ncu")
    files = sorted(d.glob("*_shape*.csv"))
    if not files:
        raise SystemExit(f"no CSVs in {d}; run `make -C src/inferno/kernels profile` first")
    lines = ["| kernel | shape | time_us | dram_%peak | bank_conf_ratio | warps_active_% | inst_M | regs/thread |",
             "|---|---|---|---|---|---|---|---|"]
    for f in files:
        rec, units = read_csv(f)
        kernel, shape = f.stem.rsplit("_shape", 1)
        vals = {label: to_float(rec.get(metric, "")) for metric, label in METRICS.items()}
        conf, wave = to_float(rec.get(CONF, "")), to_float(rec.get(WAVE, ""))
        ratio = conf / wave if conf is not None and wave else None
        inst = vals["inst_M"] / 1e6 if vals["inst_M"] is not None else None
        scale = _TO_US.get(units.get("gpu__time_duration.sum", ""), None)
        t_us = vals["time_us"] * scale if vals["time_us"] is not None and scale else None
        lines.append(f"| {kernel} | {shape} | {fmt(t_us, 2)} | {fmt(vals['dram_%peak'])} | {fmt(ratio, 2)} | "
                     f"{fmt(vals['warps_active_%'])} | {fmt(inst, 2)} | {fmt(vals['regs/thread'], 0)} |")
    out = d / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
