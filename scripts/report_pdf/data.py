"""Loads the measured result files and turns them into table rows for the report. No number is typed in by hand."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "docs" / "results"


def jl(rel):
    p = RES / rel
    return json.loads(p.read_text()) if p.exists() else None


def f(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def ci_str(m, h, d=2):
    return f"{f(m, d)} &plusmn; {f(h, d)}"


def mean_ci(vals):
    import numpy as np
    from scipy import stats

    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) < 2:
        return (float(v[0]) if len(v) else float("nan")), 0.0
    m = v.mean()
    h = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)
    return float(m), float(h)


def hardware():
    hw = jl("campaign/hardware.json") or {}
    return hw


def kernel():
    return jl("kernel_bench.json")


LADDER = ("int4_v0_naive", "int4_v1_warp", "int4_v2_vec", "int4_v3_pad", "int4_v4_alg", "int4_v5_reuse", "int4_v6_lane16")
CUSTOM = LADDER[3:]          # candidates for "best custom kernel" per shape


def best_custom(k):
    return min((k[n] for n in CUSTOM), key=lambda x: x["us_median"])


def ceiling(s):
    return (s["fp16_bytes"] + 2 * s["K"] + 2 * s["N"]) / (s["int4_bytes"] + 2 * s["K"] + 2 * s["N"])


def kernel_rows():
    """Two tables' worth of data in one list: [times..., best, best/cuBLAS, ceiling, % of ceiling, %peak, %stream, agree]."""
    b = kernel()
    stream = b.get("stream_read_gbps")
    rows = [["layer", "N x K", "cuBLAS fp16", "unfused", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "best", "best / cuBLAS", "ceiling",
             "% of ceiling", "%peak BW", "% of streaming BW", "runs agreeing"]]
    for s in b["shapes"]:
        k = {x["kernel"]: x for x in s["kernels"]}
        best = best_custom(k)
        ceil = ceiling(s)
        sp = k["cublas_fp16"]["us_median"] / best["us_median"]
        rows.append([s["name"].split("(")[0].strip().replace("_", " "), f"{s['N']}x{s['K']}",
                     f(k["cublas_fp16"]["us_median"], 1), f(k["unfused_naive"]["us_median"], 1),
                     *[f(k[n]["us_median"], 1) for n in LADDER],
                     best["kernel"].replace("int4_", ""), f"{sp:.2f}x", f"{ceil:.2f}x", f"{100 * sp / ceil:.0f}%", f"{best['pct_peak']:.0f}%",
                     f"{100 * best['gbps'] / stream:.0f}%" if stream else "n/a",
                     f"{best.get('runs_within_2pct', b.get('n_runs', 1))}/{b.get('n_runs', 1)}"])
    return rows


def kernel_spread_rows():
    b = kernel()
    names = ("cublas_fp16", "int4_v1_warp", "int4_v2_vec", "int4_v3_pad", "int4_v4_alg", "int4_v5_reuse", "int4_v6_lane16")
    rows = [["layer", "cuBLAS", "v1", "v2", "v3", "v4", "v5", "v6", "correct in all runs"]]
    for s in b["shapes"]:
        k = {x["kernel"]: x for x in s["kernels"]}
        rows.append([s["name"].split("(")[0].strip().replace("_", " "),
                     *[f"{k[n]['run_spread_pct']:.0f}% ({k[n].get('runs_within_2pct', '?')}/5)" for n in names],
                     "yes" if all(x["correct"] for x in s["kernels"]) else "NO"])
    return rows


ROLES = {"qkv": "qwen1.5B_qkv", "o_proj": "qwen1.5B_o_proj", "gate_up": "qwen1.5B_gate_up", "down": "qwen1.5B_down"}


def insitu(q, mode="graphs"):
    d = jl(f"insitu/{q}_{mode}.json")
    return {r: v["us_median"] for r, v in d["roles"].items()} if d and d.get("roles") else None


def library_rows():
    """In-situ (CUDA-graph decode) kernel times of the three int4 paths, plus the custom kernel measured standalone."""
    b = kernel()
    marlin, awq, cust = insitu("awq_marlin"), insitu("awq"), insitu("awq_inferno")
    rows = [["layer", "shape", "custom standalone (us)", "custom in situ (us)", "vLLM Marlin (us)", "vLLM AWQ (us)",
             "custom / Marlin", "custom / AWQ"]]
    for role, prefix in ROLES.items():
        s = next(x for x in b["shapes"] if x["name"].startswith(prefix))
        k = {x["kernel"]: x for x in s["kernels"]}
        best = best_custom(k)
        c = cust[role]
        rows.append([role, f"{s['N']}x{s['K']}", f(best["us_median"], 1), f(c, 1), f(marlin[role], 1), f(awq[role], 1),
                     f"{c / marlin[role]:.2f}x", f"{c / awq[role]:.2f}x"])
    return rows


def insitu_spread():
    """Largest process-to-process spread (max-min over median, %) of any in-situ per-role time among the processes NOT flagged
    as GPU-wide slow, and the path/role/mode it occurred on."""
    import glob
    import statistics
    worst = (0.0, "", "")
    for q in ("awq_marlin", "awq", "awq_inferno"):
        for mode in ("graphs", "eager"):
            agg = jl(f"insitu/{q}_{mode}.json") or {}
            slow = set(agg.get("slow_process_reps", []))
            reps = []
            for p in sorted(glob.glob(str(RES / "insitu" / f"{q}_{mode}_rep*.json"))):
                r = int(Path(p).stem.rsplit("_rep", 1)[1])
                d = json.loads(Path(p).read_text())
                if r not in slow and d.get("roles"):
                    reps.append(d["roles"])
            for role in ROLES:
                v = [r[role]["us_median"] for r in reps if role in r]
                if len(v) >= 2:
                    sp = 100 * (max(v) - min(v)) / statistics.median(v)
                    if sp > worst[0]:
                        worst = (sp, q, f"{role} {mode}")
    return worst


def insitu_slow():
    """(number of in-situ profiling processes flagged as GPU-wide slow, total processes)."""
    slow = total = 0
    for q in ("awq_marlin", "awq", "awq_inferno"):
        for mode in ("graphs", "eager"):
            d = jl(f"insitu/{q}_{mode}.json")
            if d:
                slow += len(d.get("slow_process_reps", []))
                total += d.get("n_process_reps", 1)
    return slow, total


def insitu_mode_rows():
    """Marlin, classic AWQ and the custom kernel under both engine modes (the check that exposed the invalid old Marlin timing)."""
    names = {"awq_marlin": "vLLM Marlin", "awq": "vLLM classic AWQ", "awq_inferno": "custom kernel (v6)"}
    rows = [["path", "mode", "qkv", "o_proj", "gate_up", "down", "sum per layer"]]
    for mode in ("graphs", "eager"):
        for q in ("awq_marlin", "awq", "awq_inferno"):
            v = insitu(q, mode)
            if v:
                rows.append([names[q], mode, *[f(v[r], 1) for r in ROLES], f(sum(v.values()), 1)])
    return rows


def batch_runs():
    import glob
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(RES / "kernel_batch_runs" / "run_*.json")))]


def batch_rows(shape_idx):
    import statistics
    runs = batch_runs()
    rows = [["B", "cuBLAS fp16 (us)", "custom, 16 lanes (us)", "speedup", "byte-ratio ceiling"]]
    for bi, bb in enumerate(runs[0]["shapes"][shape_idx]["batches"]):
        cub = statistics.median(r["shapes"][shape_idx]["batches"][bi]["cublas_us"] for r in runs)
        l16 = statistics.median(r["shapes"][shape_idx]["batches"][bi]["int4_l16_us"] for r in runs)
        rows.append([str(bb["B"]), f(cub, 1), f(l16, 1), f"{cub / l16:.2f}x", f"{bb['ceiling']:.2f}x"])
    return rows, runs[0]["shapes"][shape_idx]["name"].split("(")[0].strip()


def flush_rows():
    import glob
    import statistics
    fl = {}
    for p in sorted(glob.glob(str(RES / "flush_sweep" / "*.json"))):
        d = json.loads(Path(p).read_text())
        stem = Path(p).stem
        cub = stem.startswith("cub_")
        mb = int(stem.replace("cub_", "").split("_")[0].replace("mb", ""))
        fl.setdefault(mb, {"v5": [], "cublas": []})["cublas" if cub else "v5"].append(d["shapes"][0]["kernels"][0]["us_median"])
    rows = [["rotation footprint", "v5, llama7B_mlp_up (us; 3 runs)", "cuBLAS fp16 (us; median of 3)"]]
    for mb in sorted(fl):
        rows.append([f"{mb} MB", ", ".join(f(v, 1) for v in fl[mb]["v5"]), f(statistics.median(fl[mb]["cublas"]), 1)])
    return rows


def e2e():
    return jl("e2e_kernel/summary.json")


def e2e_rows():
    s = e2e()
    conc = s["concurrency"]
    rows = [["configuration", *[f"{c} seq" if c == 1 else f"{c} seqs" for c in conc]]]
    for name, e in s["configs"].items():
        cells = []
        for c in conc:
            b = e["by_concurrency"][str(c)]
            r = b.get("ratio_to_reference")
            cells.append(f"{b['mean']:.0f} &plusmn; {b['ci95']:.0f}" + (f" ({r['mean']:.2f}x)" if r and name != s["reference"] else ""))
        rows.append([e["label"], *cells])
    return rows


def headroom():
    return jl("campaign/headroom.json")


def calibration():
    return jl("calibration.json"), jl("calibration_awq_kernel.json")


def campaign_df():
    return pd.read_csv(RES / "campaign" / "summary.csv")


def campaign_tables():
    df = campaign_df()
    out = {}
    for pat in ("steady", "bursty", "mixed_length"):
        rows = [["policy", "n", "p50 (s)", "p95 (s)", "p99 (s)", "TTFT p99 (s)", "throughput (tok/s)", "goodput (req/s)", "tokens/J"]]
        for pol in ("fixed-fast", "fixed-bulk", "static-lb", "adaptive"):
            g = df[(df.pattern == pat) & (df.config == pol)]
            cells = [pol, str(len(g))]
            for m, d in (("e2e_p50", 2), ("e2e_p95", 2), ("e2e_p99", 2), ("ttft_p99", 2), ("throughput_tok_per_s", 0), ("goodput_req_per_s", 2), ("tokens_per_joule", 1)):
                cells.append(ci_str(*mean_ci(g[m]), d))
            rows.append(cells)
        out[pat] = rows
    return out


def h1_rows():
    s = jl("campaign/summary.json")
    rows = [["load pattern", "best fixed", "pairs", "adaptive p99 (s)", "best-fixed p99 (s)", "change", "Holm p", "d<sub>z</sub>", "throughput ratio"]]
    for r in s["hypothesis_h1"]:
        rows.append([r["pattern"].replace("_", " "), r["best_fixed"], str(r["n_pairs"]), f(r["treatment_mean"]), f(r["fixed_mean"]),
                     f"{r['pct_change']:+.0f}%", f"{r['p_holm']:.3f}", f"{r['cohens_dz']:.1f}", f"{r['throughput_ratio']:.2f}x"])
    return rows, s["hypothesis_h1"]


def contention_rows():
    c = jl("contention/contention_summary.json")
    rows = [["load", "condition", "n", "p99 (s)", "throughput (tok/s)", "tokens/J"]]
    order = {"single": 0, "split-idle": 1, "split-active": 2}
    for x in sorted(c["cells"], key=lambda z: (z["pattern"], order[z["config"]])):
        rows.append([x["pattern"].replace("rho", "rho = "), x["config"], str(x["n"]), ci_str(x["e2e_p99"], x["e2e_p99_ci"]),
                     f(x["throughput_tok_per_s"], 0), f(x["tokens_per_joule"], 1)])
    comp = [["load", "comparison", "metric", "ratio", "pairs"]]
    for x in c["comparisons"]:
        comp.append([x["pattern"].replace("rho", "rho = "), x["comparison"], x["metric"], f"{x['ratio_a_over_b']:.3f}", str(x["n_pairs"])])
    return rows, comp


def mpi():
    return jl("mpi_microbench.json")


def scaling():
    return jl("scaling/scaling_summary.json")


def sweep_rows():
    df = pd.read_csv(RES / "sweep" / "summary.csv")
    df["done"] = df["n_ok"] / df["n_requests"]
    rows = [["config", "reps", "throughput (tok/s)", "tokens/J", "p50 (s)", "completed", "cold start (s)"]]
    order = [f"{q}-bs{b}" for q in ("fp16", "int8", "int4") for b in (1, 8, 32, 64)]
    for cfg in order:
        g = df[df.config == cfg]
        rows.append([cfg, str(len(g)), ci_str(*mean_ci(g["throughput_tok_per_s"]), 0), f(g["tokens_per_joule"].mean(), 1),
                     f(g["e2e_p50"].mean(), 1) if g["e2e_p50"].notna().any() else "n/a", f"{100 * g['done'].mean():.0f}%",
                     f(g["cold_start_sec"].iloc[0], 0)])
    return rows


def facts() -> dict:
    """Headline numbers used in the prose; all derived from the result files."""
    out = {}
    b = kernel()
    best = None
    for s in b["shapes"]:
        k = {x["kernel"]: x for x in s["kernels"]}
        bk = best_custom(k)
        sp = k["cublas_fp16"]["us_median"] / bk["us_median"]
        ceil = ceiling(s)
        if best is None or bk["pct_peak"] > best["pct"]:
            best = {"name": s["name"].split("(")[0].strip(), "speedup": sp, "ceil": ceil, "pct": bk["pct_peak"], "gbps": bk["gbps"],
                    "us": bk["us_median"], "cub_us": k["cublas_fp16"]["us_median"], "cub_pct": k["cublas_fp16"]["pct_peak"],
                    "variant": bk["kernel"], "N": s["N"], "K": s["K"]}
    out["kern"] = best
    out["peak_gbps"], out["l2_mb"], out["device"] = b["peak_dram_gbps"], b["l2_mb"], b["device"]
    out["stream_gbps"] = b.get("stream_read_gbps")
    out["kernel_runs"] = b.get("n_runs", 1)
    rows, h1 = h1_rows()
    out["h1"] = h1
    c, ca = calibration()
    out["cal"] = {"steady_tok": c["measured"]["steady"]["tokens_per_s"], "steady_rps": c["measured"]["steady"]["requests_per_s"],
                  "mixed_tok": c["measured"]["mixed_length"]["tokens_per_s"], "mixed_rps": c["measured"]["mixed_length"]["requests_per_s"],
                  "awq_tok": ca["measured"]["steady"]["tokens_per_s"], "rates": c["rates"],
                  "steady_mean_out": c["measured"]["steady"]["mean_output_tokens"], "mixed_mean_out": c["measured"]["mixed_length"]["mean_output_tokens"]}
    out["marlin_ratio"] = out["cal"]["steady_tok"] / out["cal"]["awq_tok"]
    cont = jl("contention/contention_summary.json")
    comps = {(x["pattern"], x["comparison"], x["metric"]): x for x in cont["comparisons"]}
    out["cont_active"] = [comps[(p, "split-active vs split-idle", "e2e_p99")]["ratio_a_over_b"] for p in ("rho0.3", "rho0.6")]
    out["cont_idle"] = [comps[(p, "split-idle vs single", "e2e_p99")]["ratio_a_over_b"] for p in ("rho0.3", "rho0.6")]
    m = mpi()
    out["mpi_pp"] = {r["bytes"]: r for r in m["pingpong"]}
    out["mpi_disp"] = {r["workers"]: r for r in m["dispatch"]}
    out["cores"] = m["cores"]
    sc = scaling()
    out["scal_cells"] = {(x["router"], x["replicas"]): x for x in sc["cells"]}
    out["scal_mv"] = [x for x in sc["mpi_vs_http"] if x["metric"] == "throughput_tok_per_s"]
    out["scal_over"] = sc["overhead"]
    sw = pd.read_csv(RES / "sweep" / "summary.csv")
    cold = sw.groupby("config")["cold_start_sec"].first()
    out["cold"] = float(cold[cold < 100].median())
    ee = e2e()
    if ee:
        cf = ee["configs"]
        out["e2e"] = {n: {str(c): cf[n]["by_concurrency"][str(c)]["mean"] for c in ee["concurrency"]} for n in cf}
        out["e2e_ratio_b8"] = {str(c): cf["inferno_b8"]["by_concurrency"][str(c)].get("ratio_to_reference") for c in ee["concurrency"]} if "inferno_b8" in cf else {}
        out["e2e_layer_check"] = next((c["layer_check"] for c in cf.values() if c.get("layer_check")), None)
        out["e2e_agreement"] = ee.get("text_agreement", {})
    hr = headroom()
    out["headroom"] = hr
    ins = {q: insitu(q) for q in ("awq_marlin", "awq", "awq_inferno")}
    out["insitu"] = ins
    return out
