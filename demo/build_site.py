#!/usr/bin/env python
"""
Build the self-contained showcase site (docs/demo/index.html) from measured result files.

    python demo/build_site.py                 # writes docs/demo/index.html
    python demo/build_site.py --serve         # also serves it on http://127.0.0.1:8765

Inputs are the committed files under docs/results/ (plus, when present, the raw campaign request logs under
results/ to regenerate the replay time series). Nothing is typed in by hand: every number on the page comes from
the DATA object assembled here, and the page derives its sentences from that object.
"""
import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inferno.analysis.stats import compute_ci  # noqa: E402
from inferno.demo_data import build_timeseries  # noqa: E402

R = ROOT / "docs" / "results"
PATTERNS = ["steady", "bursty", "mixed_length"]


def jload(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def mean_ci(vals):
    v = [x for x in vals if pd.notna(x)]
    if len(v) >= 2:
        m, lo, hi = compute_ci(v)
        return float(m), float((hi - lo) / 2)
    return (float(v[0]) if v else None), 0.0


def kernel_block():
    b = jload("kernel_bench.json")
    if not b:
        return None
    shapes = []
    for s in b["shapes"]:
        ceil = (s["fp16_bytes"] + 2 * s["K"] + 2 * s["N"]) / (s["int4_bytes"] + 2 * s["K"] + 2 * s["N"])
        shapes.append({"name": s["name"].split("(")[0].strip(), "N": s["N"], "K": s["K"], "bytes": s["int4_bytes"], "ceil": ceil,
                       "kernels": [{"k": k["kernel"], "us": k["us_median"], "pct": k["pct_peak"], "gbps": k["gbps"],
                                    "spread": k.get("run_spread_pct", 0.0), "ok": k["correct"]} for k in s["kernels"]]})
    lib = {}
    for label, f in (("marlin", "insitu/awq_marlin_graphs.json"), ("awq", "insitu/awq_graphs.json"), ("inferno", "insitu/awq_inferno_graphs.json")):
        d = jload(f)
        if d and d.get("roles"):
            lib[label] = {role: v["us_median"] for role, v in d["roles"].items()}
    maxerr = max(k["max_rel_err"] for s in b["shapes"] for k in s["kernels"])
    return {"device": b["device"], "peak": b["peak_dram_gbps"], "stream": b.get("stream_read_gbps"), "runs": b.get("n_runs", 1),
            "shapes": shapes, "lib": lib, "maxerr": maxerr}


def e2e_block():
    s = jload("e2e_kernel/summary.json")
    if not s:
        return None
    return {"concurrency": s["concurrency"], "reference": s["reference"], "agreement": s.get("text_agreement", {}),
            "configs": {n: {"label": e["label"], "n": e["n_reps"], "by": {c: {"mean": v["mean"], "ci": v["ci95"], "ratio": (v.get("ratio_to_reference") or {}).get("mean")}
                                                                                for c, v in e["by_concurrency"].items()},
                            "layer_check": e.get("layer_check")} for n, e in s["configs"].items()}}


def headroom_block():
    return jload("campaign/headroom.json")


def campaign_block():
    p = R / "campaign" / "summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    cells = {}
    for (pat, cfg), g in df.groupby(["pattern", "config"]):
        row = {}
        for m, key in (("e2e_p99", "p99"), ("e2e_p50", "p50"), ("throughput_tok_per_s", "thr"),
                       ("tokens_per_joule", "tpj"), ("goodput_req_per_s", "good")):
            mean, ci = mean_ci(g[m])
            row[key], row[key + "_ci"] = mean, ci
        row["n"] = int(len(g))
        cells.setdefault(pat, {})[cfg] = row
    summ = jload("campaign/summary.json") or {}
    h1 = [{k: r[k] for k in ("pattern", "best_fixed", "n_pairs", "treatment_mean", "fixed_mean", "pct_change",
                             "p_holm", "cohens_dz", "throughput_ratio", "verdict")} for r in summ.get("hypothesis_h1", [])]
    return {"patterns": PATTERNS, "policies": ["fixed-fast", "fixed-bulk", "static-lb", "adaptive"], "cells": cells,
            "h1": h1}


def contention_block():
    c = jload("contention/contention_summary.json")
    if not c:
        return None
    cells = [{k: x[k] for k in ("pattern", "config", "n", "e2e_p99", "e2e_p99_ci", "throughput_tok_per_s",
                                "tokens_per_joule")} for x in c["cells"]]
    comps = [{k: x[k] for k in ("pattern", "comparison", "metric", "ratio_a_over_b", "n_pairs")} for x in c["comparisons"]]
    return {"cells": cells, "comps": comps}


def sweep_block():
    p = R / "sweep" / "summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["done"] = df["n_ok"] / df["n_requests"]
    out = []
    for cfg, g in df.groupby("config"):
        q, bs = cfg.split("-bs")
        row = {"config": cfg, "quant": q, "bs": int(bs), "cold": float(g["cold_start_sec"].iloc[0])}
        for m, key in (("throughput_tok_per_s", "thr"), ("tokens_per_joule", "tpj"), ("e2e_p50", "p50"), ("done", "done")):
            mean, ci = mean_ci(g[m])
            row[key], row[key + "_ci"] = mean, ci
        out.append(row)
    return out


def mpi_block():
    m = jload("mpi_microbench.json")
    s = jload("scaling/scaling_summary.json")
    if not m and not s:
        return None
    out = {}
    if m:
        out["pingpong"], out["dispatch"], out["cores"] = m["pingpong"], m["dispatch"], m["cores"]
    if s:
        out["scaling"] = {"cells": s["cells"], "mv": s["mpi_vs_http"], "overhead": s.get("overhead", [])}
    return out


def replay_block():
    out = {}
    cache = R / "demo"
    cache.mkdir(parents=True, exist_ok=True)
    runs = sorted((ROOT / "results").glob("campaign_*"))
    for pat in PATTERNS:
        f = cache / f"timeseries_{pat}_rep0.json"
        if runs and (runs[-1] / "requests").exists():
            ts = build_timeseries(runs[-1], pat, 0)
            f.write_text(json.dumps(ts))
        elif f.exists():
            ts = json.loads(f.read_text())
        else:
            continue
        out[pat] = ts
    return out or None


def build() -> Path:
    hw = (jload("campaign/hardware.json") or {})
    gpu = (hw.get("gpus") or [{}])[0]
    cal = {k: jload(f) for k, f in (("marlin", "calibration.json"), ("awq", "calibration_awq_kernel.json"))}
    data = {
        "hw": {"gpu": gpu.get("name", "GPU"), "vllm": hw.get("packages", {}).get("vllm"),
               "torch": hw.get("packages", {}).get("torch"), "mem_mib": gpu.get("memory.total")},
        "kernel": kernel_block(), "e2e": e2e_block(), "headroom": headroom_block(), "campaign": campaign_block(), "contention": contention_block(),
        "sweep": sweep_block(), "mpi": mpi_block(), "replay": replay_block(),
        "calibration": {k: (v["measured"]["steady"]["tokens_per_s"] if v else None) for k, v in cal.items()},
    }
    tpl = (ROOT / "demo" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "docs" / "demo" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    out = build()
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    if args.serve:
        print(f"Serving http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
        ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(out.parent))).serve_forever()


if __name__ == "__main__":
    main()
