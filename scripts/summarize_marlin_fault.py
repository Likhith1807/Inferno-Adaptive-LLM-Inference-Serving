#!/usr/bin/env python
"""
Summarize the Marlin batch-size correctness probe into docs/results/marlin_fault/{summary.json,report.md}.

Inputs (written by the probes, raw in results/marlin_fault/):
  kernel_probe.txt     scripts/probe_marlin_flaky.sh: 40 identical calls per (shape, batch, variant)
  e2e_*.json           scripts/probe_marlin_e2e.sh: real-checkpoint greedy decode, batched vs batch-1 text (optional)

    python scripts/summarize_marlin_fault.py
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "results" / "marlin_fault", ROOT / "docs" / "results" / "marlin_fault"
HEAD = re.compile(r"N=(\d+) K=(\d+) alloc=(\S+) gpu=(.+?) SMs=(\d+)")
ROW = re.compile(r"B=\s*(\d+) (\w+)\s+bad=\s*(\d+)/(\d+) worst_rel_err=([\d.e+-]+|nan)")
SHAPE_NAMES = {(1536, 8960): "down", (17920, 1536): "gate_up", (2048, 1536): "qkv", (1536, 1536): "o_proj"}


def parse(txt: str) -> dict:
    out, cur = {"gpu": None, "shapes": []}, None
    for line in txt.splitlines():
        m = HEAD.search(line)
        if m:
            N, K = int(m[1]), int(m[2])
            cur = {"N": N, "K": K, "name": SHAPE_NAMES.get((N, K), f"{N}x{K}"), "rows": []}
            out["gpu"], out["sms"] = m[4].strip(), int(m[5])
            out["shapes"].append(cur)
            continue
        m = ROW.search(line)
        if m and cur is not None:
            cur["rows"].append({"B": int(m[1]), "variant": m[2], "bad": int(m[3]), "calls": int(m[4]),
                                "worst_rel_err": float(m[5])})
    return out


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    probe = parse((SRC / "kernel_probe.txt").read_text())
    shutil.copy(SRC / "kernel_probe.txt", DST / "kernel_probe.txt")
    e2e = {}
    for p in sorted(SRC.glob("e2e_*.json")):
        shutil.copy(p, DST / p.name)
        e2e[p.stem] = json.loads(p.read_text())
    batches = sorted({r["B"] for s in probe["shapes"] for r in s["rows"]})
    faulty = sorted({r["B"] for s in probe["shapes"] for r in s["rows"] if r["bad"] > 0})
    clean = [b for b in batches if b not in faulty]
    worst = max((r["worst_rel_err"] for s in probe["shapes"] for r in s["rows"] if r["bad"] > 0), default=0.0)
    ok_err = max((r["worst_rel_err"] for s in probe["shapes"] for r in s["rows"] if r["bad"] == 0), default=0.0)
    summary = {"gpu": probe["gpu"], "sms": probe["sms"], "batches": batches, "faulty_batches": faulty,
               "clean_batches": clean, "worst_rel_err_faulty": worst, "worst_rel_err_clean": ok_err,
               "shapes": probe["shapes"], "e2e": e2e,
               "status": "observed on one machine (WSL2, laptop GPU); not yet reproduced on other hardware"}
    (DST / "summary.json").write_text(json.dumps(summary, indent=2))

    L = ["# Marlin int4 GEMM: batch-size-dependent wrong results (generated)\n",
         f"GPU {probe['gpu']} ({probe['sms']} SMs), vLLM 0.11.0 `gptq_marlin_gemm` with AWQ uint4 weights (group 128), "
         "called exactly as `apply_awq_marlin_linear` does. Each row: 40 calls with identical inputs, compared with "
         "`awq_dequantize` + fp32 matmul; a call is bad if its relative max error exceeds 1e-2. Variants: `plain` "
         "back-to-back with vLLM's shared workspace; `dirty` NaN-filled allocations between calls; `fresh_ws` a newly "
         "zeroed workspace per call; `sync` a device synchronize between calls.\n",
         f"**Status: {summary['status']}.** Not reported upstream until reproduced.\n",
         "## Bad calls out of 40 (worst relative error)\n"]
    variants = sorted({r["variant"] for s in probe["shapes"] for r in s["rows"]}, key=["plain", "dirty", "fresh_ws", "sync"].index)
    for s in probe["shapes"]:
        L.append(f"### {s['name']} (N={s['N']}, K={s['K']})\n")
        L.append("| B | " + " | ".join(variants) + " |")
        L.append("|---|" + "---|" * len(variants))
        for b in batches:
            cells = []
            for v in variants:
                r = next((r for r in s["rows"] if r["B"] == b and r["variant"] == v), None)
                cells.append("" if r is None else f"{r['bad']}/{r['calls']} ({r['worst_rel_err']:.1e})")
            L.append(f"| {b} | " + " | ".join(cells) + " |")
        L.append("")
    L.append(f"Batch sizes with any bad call: {faulty}. Always correct: {clean} (worst error {ok_err:.1e}). "
             f"Worst faulty error {worst:.2f}. The custom kernel, classic AWQ and fp16 are deterministic and correct on "
             "the same inputs (`docs/results/layer_matrix/report.md`). A fresh zeroed workspace does not remove the "
             "fault, so stale lock state in the shared workspace is not the explanation.\n")
    if e2e:
        L.append("## Real checkpoint (Qwen2.5-1.5B-Instruct-AWQ, greedy): batched copies identical to batch-1 text\n")
        L.append("| run | " + " | ".join(f"B={b}" for b in next(iter(e2e.values()))["batches"]) + " |")
        L.append("|---|" + "---|" * len(next(iter(e2e.values()))["batches"]))
        for k, v in e2e.items():
            L.append(f"| {k} | " + " | ".join(f"{c['exact_match']}/{c['n']}" for c in v["batches"].values()) + " |")
        L.append("\nBatch-dependent numerics make a few late divergences normal for any kernel; compare Marlin "
                 "against classic AWQ at the same batch size.\n")
    else:
        L.append("## Real checkpoint\n\nNot run yet (`scripts/probe_marlin_e2e.sh`).\n")
    L.append("## Reproduce\n\n```bash\nbash scripts/probe_marlin_flaky.sh > results/marlin_fault/kernel_probe.txt\n"
             "bash scripts/probe_marlin_e2e.sh\npython scripts/summarize_marlin_fault.py\n```\n")
    (DST / "report.md").write_text("\n".join(L))
    print("wrote", DST / "report.md", "| faulty batches:", faulty)


if __name__ == "__main__":
    main()
