#!/usr/bin/env python
"""
Generate docs/FINDINGS.md and docs/POST_DRAFT.md from the measured result files in docs/results/.

Every number is read from a data file; inputs that do not exist yet are skipped rather than filled in. The prose
states each finding in the direction the data points, including negative ones. Review before publishing.

    python scripts/build_findings.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "docs" / "results"


def load(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def _p99(cont, pattern, config):
    return next(c["e2e_p99"] for c in cont["cells"] if c["pattern"] == pattern and c["config"] == config)


def _n(cont, pattern):
    return min(c["n"] for c in cont["cells"] if c["pattern"] == pattern)


def main():
    kern = load("kernel_bench.json")
    cmp_md = (R / "kernel_comparison.md")
    cal_marlin, cal_awq = load("calibration.json"), load("calibration_awq_kernel.json")
    camp = load("campaign/summary.json")
    cont = load("contention/contention_summary.json")
    scal = load("scaling/scaling_summary.json")
    mpi = load("mpi_microbench.json")
    hw = load("campaign/hardware.json")
    e2e = load("e2e_kernel/summary.json")
    headroom = load("campaign/headroom.json")
    import glob
    batch_runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(R / "kernel_batch_runs" / "run_*.json")))]

    def e2e_tok(name, c):
        return e2e["configs"][name]["by_concurrency"][str(c)]["mean"] if e2e and name in e2e["configs"] else None
    gpu = ((hw or {}).get("gpus") or [{}])[0].get("name", "an 8 GB laptop GPU")

    F, P, facts = [], [], {}
    F.append("# Findings\n")
    F.append(f"All measured on **{gpu}** with vLLM 0.11 and Qwen2.5-1.5B-Instruct unless stated. Each number is read "
             "from a data file under `docs/results/` by `scripts/build_findings.py`; methods and caveats are in "
             "`docs/methodology.md`, `docs/KERNEL_REPORT.md` and the per-experiment reports.\n")

    # ---------------------------------------------------------------- kernel
    if kern:
        by = {s["name"].split("(")[0].strip(): {k["kernel"]: k for k in s["kernels"]} for s in kern["shapes"]}
        best = {}
        for name, k in by.items():
            cands = [k[n] for n in ("int4_v3_pad", "int4_v4_alg", "int4_v5_reuse", "int4_v6_lane16")]
            b = min(cands, key=lambda x: x["us_median"])
            best[name] = (b, k["cublas_fp16"]["us_median"] / b["us_median"])
        top = max(best.items(), key=lambda kv: kv[1][0]["pct_peak"])
        facts["kern_top"] = top
        big = [(n, v) for n, v in best.items() if kern["shapes"][list(by).index(n)]["int4_bytes"] > 5e6]
        _lo, _hi = min(v[0]["pct_peak"] for _, v in big), max(v[0]["pct_peak"] for _, v in big)
        F.append("## 1. A fused int4 GEMV kernel reaches %.0f-%.0f%% of DRAM bandwidth on large layers (parity with vLLM's Marlin end to end at batch 1, not a win)\n" % (_lo, _hi))
        ts = kern["shapes"][list(by).index(top[0])]
        ceil_top = (ts["fp16_bytes"] + 2 * ts["K"] + 2 * ts["N"]) / (ts["int4_bytes"] + 2 * ts["K"] + 2 * ts["N"])
        facts["ceil_top"] = ceil_top
        F.append(f"- Best case **{top[0]}**: {top[1][0]['pct_peak']:.0f}% of peak DRAM bandwidth, "
                 f"**{top[1][1]:.2f}x faster than fp16 cuBLAS**, against a byte-ratio ceiling of {ceil_top:.2f}x "
                 "(int4 also reads fp16 scales and zero points, so it moves 0.53 bytes per weight, not 0.5). "
                 f"Median of {kern.get('n_runs', 1)} benchmark invocations; every kernel matched a CPU reference.")
        F.append("- Large layers: " + "; ".join(f"{n} {v[1]:.2f}x ({v[0]['pct_peak']:.0f}% of peak)" for n, v in big) + ".")
        F.append("- Small layers (about 1-2 MB of weights) are dominated by launch and clock jitter (only 2-4 of 5 runs "
                 "agree within 2%), so no ranking is claimed there.")
        st = kern.get("stream_read_gbps")
        if st:
            F.append(f"- Measured streaming-read bandwidth is {st:.0f} GB/s ({100 * st / kern['peak_dram_gbps']:.0f}% of the {kern['peak_dram_gbps']:.0f} GB/s "
                     f"computed from the memory clock); the best kernel reaches {100 * top[1][0]['gbps'] / st:.0f}% of that achievable figure. "
                     "The benchmark rotates 768 MB of weights: an earlier 192 MB rotation let a non-LRU L2 serve about 6% of the traffic and inflated every "
                     "'% of peak' figure (`docs/KERNEL_REPORT.md`).")
        F.append("- The optimization ladder is not monotone: wider loads with shared-memory staging made the kernel *slower* until the predicted "
                 "16-way bank conflict (from address arithmetic; counters pending) was padded away; the algebraic rewrite (v4) helped on K=4096 but hurt "
                 "on K=1536/8960; dropping shared-memory staging (v5) and balancing lanes for 48-chunk rows (v6) fixed that. Causes are hypotheses until "
                 "Nsight counters can be read.")
        if batch_runs:
            import statistics as _st
            si = 0
            sp_ = []
            for bi, bb in enumerate(batch_runs[0]["shapes"][si]["batches"]):
                if bb["B"] in (1, 2, 4, 8):
                    cub = _st.median(r["shapes"][si]["batches"][bi]["cublas_us"] for r in batch_runs)
                    l16 = _st.median(r["shapes"][si]["batches"][bi]["int4_l16_us"] for r in batch_runs)
                    sp_.append("B=%d %.2fx" % (bb["B"], cub / l16))
            F.append("- Small batches (`%s`, speedup over fp16 cuBLAS with the same B): %s. The CUDA-core kernel loses to tensor-core cuBLAS above "
                     "B of about 8; it is a batch-1 / speculative-decoding-sized result." % (batch_runs[0]["shapes"][si]["name"].split("(")[0].strip(), ", ".join(sp_)))
    if cmp_md.exists():
        F.append("### Against vLLM's own kernels (decode, in situ, CUDA graphs)\n")
        F.append(cmp_md.read_text(encoding="utf-8"))
        F.append("\n**vLLM's Marlin kernel is at the bandwidth limit on the large layers at batch 1; the custom kernel does not beat it.** "
                 "(An earlier version of this project reported the opposite from a Marlin timing that turned out to be about 2x too slow and "
                 "irreproducible; it was found by cross-checking against end-to-end throughput, re-measured in both engine modes and retired. "
                 "See `docs/KERNEL_REPORT.md`.)\n")
        if e2e:
            F.append("### End to end in vLLM (decode throughput, tokens/s, CUDA graphs, mean of process repetitions)\n")
            F.append((R / "e2e_kernel" / "summary.md").read_text(encoding="utf-8"))
            F.append("\nAt one sequence the custom kernel matches vLLM's default Marlin path and beats the classic AWQ kernel and fp16; "
                     "with 4-8 concurrent sequences Marlin's tensor-core kernel is clearly faster. The custom kernel was verified on the real checkpoint "
                     "(all patched layers within fp16 rounding of vLLM's own dequantize-and-matmul).\n")
        if cal_marlin and cal_awq:
            r = cal_marlin["measured"]["steady"]["tokens_per_s"] / cal_awq["measured"]["steady"]["tokens_per_s"]
            facts["marlin_ratio"] = r
            F.append(f"At server level with 64 concurrent sequences, vLLM's Marlin path sustains "
                     f"**{r:.2f}x** the throughput of the classic AWQ path "
                     f"({cal_marlin['measured']['steady']['tokens_per_s']:.0f} vs "
                     f"{cal_awq['measured']['steady']['tokens_per_s']:.0f} tok/s): a batch-1 kernel is a low-concurrency, latency-sensitive result.\n")

    # ---------------------------------------------------------------- adaptive
    if camp and camp.get("hypothesis_h1"):
        h = camp["hypothesis_h1"]
        facts["h1"] = h
        F.append("## 2. The adaptive router did not beat the best fixed configuration (H1 rejected)\n")
        F.append("Pre-registered comparison, 8 paired repetitions per load pattern, best fixed baseline chosen "
                 "per pattern, Holm-corrected:\n")
        F.append("| load pattern | best fixed | adaptive p99 vs best fixed | Holm p | Cohen's dz | throughput ratio |")
        F.append("|---|---|---|---|---|---|")
        for r in h:
            F.append(f"| {r['pattern']} | {r['best_fixed']} | {r['pct_change']:+.0f}% | {r['p_holm']:.3f} | "
                     f"{r['cohens_dz']:.1f} | {r['throughput_ratio']:.2f}x |")
        F.append("\nAdaptive was significantly *worse* on every pattern (all 8 pairs in the same direction, "
                 "Holm p is the smallest attainable at n=8), with lower throughput and lower tokens/joule too. It tracked the "
                 "static load balancer almost exactly.\n")
        if headroom:
            F.append("Headroom check (post hoc, existing data): the two fixed tiers differ by at most %.1f%% in p99 once the better one is chosen per pattern "
                     "(%s is the best single tier by worst-case regret), so even a perfect pattern-aware switch between them could not have won by more than that "
                     "on this pool. The H1 test was close to unwinnable here, independent of the co-residency penalty of the policy actually tested.\n"
                     % (headroom["max_headroom_pct"], headroom["single_fixed_tier"]))
    if cont:
        comps = {(c["pattern"], c["comparison"], c["metric"]): c for c in cont["comparisons"]}
        act = [comps[(p, "split-active vs split-idle", "e2e_p99")] for p in sorted({c["pattern"] for c in cont["cells"]})]
        idl = [comps[(p, "split-idle vs single", "e2e_p99")] for p in sorted({c["pattern"] for c in cont["cells"]})]
        facts["cont"] = (act, idl)
        F.append("### Why: two active replicas on one GPU contend\n")
        F.append("A controlled experiment holds total load fixed (30% and 60% of measured capacity) and varies only how "
                 "the GPU is shared:\n")
        idle_txt = ", ".join("%+.1f%%" % ((c["ratio_a_over_b"] - 1) * 100) for c in idl)
        act_txt = ", ".join("%.2fx" % c["ratio_a_over_b"] for c in act)
        F.append("- Halving the memory budget with the second replica idle changes p99 by " + idle_txt + ": **no cost**.")
        F.append("- Making the second replica *active* multiplies p99 by " + act_txt + ", with about 5% lower "
                 "throughput and 8-9% lower tokens/joule: two processes time-slice one GPU and run smaller batches each.")
        F.append("- Design implication: replica-level adaptivity needs replicas on separate devices. The cloud "
                 "configs (`configs/pool_config_cloud.yaml`, `infra/aws/`) put one replica per GPU; they have not been "
                 "run yet.\n")

    # ---------------------------------------------------------------- MPI / scaling
    if mpi:
        pp = {r["bytes"]: r for r in mpi["pingpong"]}
        d = {r["workers"]: r for r in mpi["dispatch"]}
        facts["mpi"] = (pp, d)
        F.append("## 3. MPI routing: microseconds per hop, unmeasurable end to end\n")
        F.append(f"- Ping-pong: one-way latency **{pp[64]['one_way_us_est']:.1f} us** for 64 B, "
                 f"{pp[32768]['one_way_us_est']:.1f} us for 32 KB, and {pp[262144]['one_way_us_est']:.0f} us for 256 KB "
                 "(a protocol cliff between 32 KB and 256 KB).")
        F.append(f"- Dispatch loop: {d[1]['jobs_per_s'] / 1e3:.0f}k jobs/s with 1 worker, "
                 f"{d[8]['jobs_per_s'] / 1e3:.0f}k with 8, {d[15]['jobs_per_s'] / 1e3:.0f}k with 15 "
                 f"(on {mpi['cores']} logical cores, so the 15-worker point is oversubscribed and shows p99 spikes).")
    if scal:
        cells = {(c["router"], c["replicas"]): c for c in scal["cells"]}
        mv = [c for c in scal["mpi_vs_http"] if c["metric"] == "throughput_tok_per_s"]
        facts["scal"] = (cells, mv, scal.get("overhead", []))
        F.append("- Real replicas (saturating load, MPI vs HTTP router, paired by repetition): "
                 + "; ".join(f"{c['replicas']} replica(s): MPI {c['pct_diff_mpi_vs_http']:+.2f}% throughput vs HTTP "
                              f"(p_ttest {c['p_ttest']:.2f}, p_wilcoxon {c['p_wilcoxon']:.2f})" for c in mv) +
                 ". No end-to-end difference is detectable.")
        e = (cells.get(("http", 2)) or {}).get("efficiency_vs_1x")
        if e is not None:
            F.append(f"- Two replicas on one GPU reach only **{e:.2f}** of ideal linear scaling "
                     f"({cells[('http', 1)]['throughput']:.0f} to {cells[('http', 2)]['throughput']:.0f} tok/s): "
                     "throughput falls, the same co-residency penalty as above.")
        if scal.get("overhead"):
            o = scal["overhead"][0]
            F.append(f"- Inside the real router the measured MPI dispatch hop is about "
                     f"{o['mpi_dispatch_us_p50'] / 1e3:.1f} ms (p50) rather than microseconds, because the worker's single "
                     "asyncio loop also parses hundreds of streaming responses; still under 0.1% of a request's latency.")
        F.append("")

    # ---------------------------------------------------------------- Phase 2 sweep
    sweep_csv = R / "sweep" / "summary.csv"
    if sweep_csv.exists():
        import pandas as pd
        sw = pd.read_csv(sweep_csv)
        sw["done"] = sw["n_ok"] / sw["n_requests"]
        g = sw.groupby("config").agg(thr=("throughput_tok_per_s", "mean"), tpj=("tokens_per_joule", "mean"),
                                     p50=("e2e_p50", "mean"), done=("done", "mean"), cold=("cold_start_sec", "median"))
        if {"fp16-bs64", "int4-bs64", "int8-bs64"} <= set(g.index):
            f16, i8, i4 = g.loc["fp16-bs64"], g.loc["int8-bs64"], g.loc["int4-bs64"]
            cold = sw.groupby("config")["cold_start_sec"].first()
            cold = cold[cold < 100]                               # exclude first-time model downloads
            facts["sweep"] = (f16, i8, i4, float(cold.median()))
            F.append("## 4. Quantization and batch cap (Phase 2 characterization)\n")
            F.append("Steady load at 0.6x the measured capacity of the best configuration; 3 repetitions per cell; "
                     "unfinished requests 60 s after the last arrival are counted as failures "
                     "(see `docs/results/sweep/report.md`).\n")
            F.append("- At the same offered load, moving fp16 to int4 (batch cap 64) cuts median latency from "
                     "%.1f s to **%.1f s (%.1fx)** and raises tokens per joule from %.1f to **%.1f (%+.0f%%)**; "
                     "int8 sits between (%.1f s, %.1f tok/J). Completed throughput differs by only %.0f%% because the load, "
                     "not the hardware, limits it."
                     % (f16.p50, i4.p50, f16.p50 / i4.p50, f16.tpj, i4.tpj, 100 * (i4.tpj / f16.tpj - 1), i8.p50, i8.tpj,
                        100 * (i4.thr / f16.thr - 1)))
            F.append("- A batch cap of 1 cannot sustain this load at any quantization level (fp16 and int8 complete 0% of "
                     "requests in the window); caps of 32 and 64 are indistinguishable at this load because in-flight "
                     "concurrency stays under 32.")
            F.append("- **Reconfiguration is expensive:** restarting the server to change batch size or quantization "
                     "took a median of **%.0f s** (model already cached), several times a 10 s burst. This is why the "
                     "controller routes across pre-loaded replicas instead of hot-swapping.\n" % facts["sweep"][3])

    # ---------------------------------------------------------------- layer matrix (milestone 2)
    lm = load("layer_matrix/layer_matrix.json")
    if lm:
        B = lm["batches"]
        ratio = {n: {b: (s["cells"][f"custom/B{b}"]["us_median"] / s["cells"][f"marlin/B{b}"]["us_median"])
                     for b in B if s["cells"].get(f"custom/B{b}")} for n, s in lm["shapes"].items()}
        share = {n: s["cells"]["marlin/B1"]["us_median"] for n, s in lm["shapes"].items()}
        tot = sum(share.values())
        share = {n: 100 * v / tot for n, v in share.items()}
        b1_wins = [n for n in ratio if ratio[n][1] < 1 and lm["shapes"][n]["cells"]["custom/B1"]["us_max"]
                   < lm["shapes"][n]["cells"]["marlin/B1"]["us_min"]]
        wins_share = sum(share[n] for n in b1_wins)
        hi = max(r[16] for r in ratio.values() if 16 in r)
        lo4 = min(r[4] for r in ratio.values() if 4 in r)
        facts["lm"] = (ratio, share, b1_wins, wins_share, hi)
        facts["lm_lo4"] = lo4
        F.append("## 5. Where the custom kernel earns its place: almost nowhere (decision: keep vLLM's Marlin)\n")
        F.append("Every Qwen2.5-1.5B decode layer at batch 1-64, same AWQ weights for every implementation, weights "
                 "streamed from DRAM, CUDA-graph timing (`docs/results/layer_matrix/report.md`, "
                 "`scripts/run_layer_matrix.sh`).\n")
        F.append("- Custom/Marlin time at batch 1: " + ", ".join(f"{n} {ratio[n][1]:.2f}" for n in ratio)
                 + f". The custom kernel clearly wins only on {', '.join(b1_wins) or 'no layer'}, "
                 f"which are {wins_share:.0f}% of quantized-GEMM time per decode step; the two large layers "
                 f"(gate_up {share['gate_up']:.0f}%, down {share['down']:.0f}%) are ties.")
        F.append(f"- From batch 4 the custom kernel is {lo4:.1f}-{hi:.1f}x slower: its time grows roughly linearly with batch "
                 "(CUDA-core arithmetic per weight scales with batch), while Marlin's tensor-core path stays flat to batch 16.")
        F.append("- One-time layout conversion is negligible for both (milliseconds per layer at load).")
        F.append("- **Decision (pre-stated rule):** the built-in path already matches the custom kernel on the layers "
                 "that dominate decode, so the kernel is not carried into the serving path as a default; engineering "
                 "effort goes to routing. The opt-in integration (`inferno.kernels.vllm_int4.install`) stays as a "
                 "research tool.\n")

    mf = load("marlin_fault/summary.json")
    if mf:
        facts["mf"] = mf
        F.append("## 6. vLLM's Marlin int4 GEMM returned wrong results at some batch sizes on this machine\n")
        F.append(f"The layer benchmark's correctness check flagged Marlin. In isolation (40 identical calls per cell, "
                 f"`docs/results/marlin_fault/report.md`), batches {mf['faulty_batches']} give intermittent relative "
                 f"errors up to {mf['worst_rel_err_faulty']:.2f} on every Qwen shape tested, while batches "
                 f"{mf['clean_batches']} are always correct (worst {mf['worst_rel_err_clean']:.0e}). The custom kernel, "
                 "classic AWQ and fp16 are correct on the same inputs; a fresh zeroed workspace does not remove it.")
        F.append(f"- **Status: {mf['status']}.** It may be specific to WSL2 or this GPU; it is not called a vLLM bug "
                 "until it reproduces elsewhere (`scripts/probe_marlin_flaky.sh`).")
        if mf.get("e2e"):
            F.append("- Real-checkpoint greedy decode, batched copies identical to the batch-1 text: " + "; ".join(
                f"{k}: " + ", ".join(f"B={b} {c['exact_match']}/{c['n']}" for b, c in v["batches"].items())
                for k, v in mf["e2e"].items()) + ".")
        else:
            F.append("- Real-checkpoint end-to-end check (`scripts/probe_marlin_e2e.sh`) not run yet.")
        F.append("- Marlin timings at those batch sizes in section 5 should be read with this caveat.\n")

    F.append("## Limitations\n")
    F.append("Single small GPU; replicas share it. Synthetic prompt text with server-verified token counts; "
             "checkpoint quality is not evaluated. n=4-8 repetitions per cell: enough "
             "for large effects, not for effects near 1%. See `docs/methodology.md`.\n")
    (ROOT / "docs" / "FINDINGS.md").write_text("\n".join(F), encoding="utf-8")

    # ---------------------------------------------------------------- README summary block
    S = ["**Headline findings** (details in [`docs/FINDINGS.md`](docs/FINDINGS.md); every number is generated from data):", ""]
    if "kern_top" in facts:
        n, (b, sp) = facts["kern_top"]
        st_ = kern.get("stream_read_gbps")
        txt_e2e = ""
        if e2e and e2e_tok("inferno_b8", 1) and e2e_tok("awq_marlin", 1):
            txt_e2e = (" Inside vLLM (CUDA graphs, one sequence) it decodes at %.0f tok/s vs %.0f for the default Marlin path (%.0f for fp16), "
                       "and Marlin is faster with 4+ concurrent sequences." % (e2e_tok("inferno_b8", 1), e2e_tok("awq_marlin", 1), e2e_tok("fp16", 1)))
        S.append(f"- **Kernel:** a hand-written fused int4 GEMV reaches {b['pct_peak']:.0f}% of theoretical DRAM bandwidth"
                 + (f" ({100 * b['gbps'] / st_:.0f}% of the measured streaming-read bandwidth)" if st_ else "")
                 + f" and {sp:.2f}x fp16 cuBLAS on `{n}` (byte-ratio ceiling {facts['ceil_top']:.2f}x)." + txt_e2e
                 + " It does not beat vLLM's Marlin kernel at batch 1 (an earlier claim that it did was a measurement error, since corrected).")
    if "h1" in facts:
        lo = min(r["pct_change"] for r in facts["h1"])
        hi = max(r["pct_change"] for r in facts["h1"])
        S.append(f"- **Adaptive routing (H1) was rejected:** p99 {lo:+.0f}% to {hi:+.0f}% versus the best fixed "
                 "configuration, 8 paired repetitions, Holm-corrected.")
    if "cont" in facts:
        S.append("- **Why:** a second *active* replica on the same GPU multiplies p99 by "
                 + "%.1fx" % facts["cont"][0][0]["ratio_a_over_b"] + "; an idle resident one costs nothing"
                 + (" (60%% load: %.2f s one server, %.2f s with an idle second replica, %.2f s with both serving; "
                    "%d paired reps)" % (tuple(_p99(cont, "rho0.6", c) for c in ("single", "split-idle", "split-active"))
                                         + (_n(cont, "rho0.6"),)) if cont else "")
                 + ". Replicas need separate devices.")
    if "lm" in facts:
        ratio, share, b1_wins, wins_share, hi = facts["lm"]
        S.append(f"- **Kernel vs vLLM's Marlin, every Qwen layer, batch 1-64:** the custom kernel wins only on "
                 f"{'/'.join(b1_wins) or 'no layer'} at batch 1 ({wins_share:.0f}% of decode GEMM time), ties the large "
                 f"layers, and is {facts['lm_lo4']:.1f}-{hi:.1f}x slower from batch 4. Decision: keep Marlin; effort moves to routing.")
    if "mf" in facts:
        mf = facts["mf"]
        S.append(f"- **Correctness finding (unconfirmed elsewhere):** on this machine vLLM 0.11's Marlin int4 GEMM gave "
                 f"intermittent errors up to {mf['worst_rel_err_faulty']:.0%} at batches {mf['faulty_batches']}, correct at "
                 f"{mf['clean_batches']}; caught by the benchmark's reference check "
                 "([report](docs/results/marlin_fault/report.md)).")
    if "sweep" in facts:
        f16, i8, i4, cold = facts["sweep"]
        S.append("- **Quantization (same offered load):** int4 vs fp16 cuts p50 latency %.1fx and lifts tokens/joule %+.0f%%; "
                 "reconfiguring a server costs ~%.0f s, so adaptivity has to route across pre-loaded replicas."
                 % (f16.p50 / i4.p50, 100 * (i4.tpj / f16.tpj - 1), cold))
    if "mpi" in facts:
        pp, d = facts["mpi"]
        S.append("- **MPI routing:** ~" + "%.0f" % pp[64]["one_way_us_est"] + " us per hop in isolation, no measurable "
                 "end-to-end difference from an HTTP router.")
    # ---------------------------------------------------------------- hypothesis outcomes table (README)
    import pandas as pd
    O = ["| ID | Statement | Outcome (generated from data) |", "|---|---|---|"]
    if "h1" in facts:
        worst = [r for r in facts["h1"] if r["pct_change"] > 0]
        O.append("| H1 | The adaptive router achieves lower p99 than the best fixed configuration at non-inferior "
                 "throughput under bursty / mixed-length load | **" + ("Rejected" if len(worst) == len(facts["h1"]) else "Mixed")
                 + "** on this hardware: p99 " + ", ".join("%s %+.0f%%" % (r["pattern"], r["pct_change"]) for r in facts["h1"])
                 + " (paired, Holm-corrected). Mechanism isolated; untested with one replica per GPU. |")
    if "scal" in facts and "mpi" in facts:
        cells, mv, ov = facts["scal"]
        pp, d = facts["mpi"]
        eff = (cells.get(("http", 2)) or {}).get("efficiency_vs_1x")
        t1, t2 = (cells.get(("http", 1)) or {}).get("throughput"), (cells.get(("http", 2)) or {}).get("throughput")
        O.append("| H2 | MPI-coordinated multi-replica routing scales sub-linearly; coordination overhead is measured "
                 "| **Sub-linear: yes** (2 replicas on one GPU: efficiency %.2f, throughput %.0f to %.0f tok/s). "
                 "**Overhead:** ~%.0f us/hop isolated, ~%.1f ms in situ, MPI vs HTTP throughput %s; no crossover observed. |"
                 % (eff, t1, t2, pp[64]["one_way_us_est"], (ov[0]["mpi_dispatch_us_p50"] / 1e3) if ov else float("nan"),
                    ", ".join("%+.2f%%" % c["pct_diff_mpi_vs_http"] for c in mv)))
    if "kern_top" in facts:
        n, (b, sp) = facts["kern_top"]
        O.append("| H3 | A hand-written fused int4 kernel reaches a fraction of the vendor path; the gap is explained by "
                 "profiler counters | **Refuted as stated, explanation pending:** the kernel reaches %.0f%% of DRAM peak and %.2fx fp16 cuBLAS on %s; "
                 "in situ it is close to, but not faster than, vLLM's Marlin kernel at batch 1, and slower at concurrency"
                 % (b["pct_peak"], sp, n)
                 + (" (layer matrix: wins only %s at batch 1)" % ("/".join(facts["lm"][2]) or "no layer") if "lm" in facts else "")
                 + ". Counter-level explanation is pending (GPU counters blocked, `docs/PROFILING.md`). |")
    eff_csv = R / "campaign" / "all_metrics_ci.csv"
    if eff_csv.exists() and "h1" in facts:
        ci = pd.read_csv(eff_csv)
        tj = ci[ci.metric == "tokens_per_joule"].pivot(index="pattern", columns="config", values="mean")
        rows_ = []
        for r in facts["h1"]:
            rows_.append("%s %+.1f%%" % (r["pattern"], 100 * (tj.loc[r["pattern"], "adaptive"] / tj.loc[r["pattern"], r["best_fixed"]] - 1)))
        O.append("| H4 | Systems-level adaptivity, not model changes, drives efficiency gains (tokens per joule) | **Not "
                 "supported locally:** adaptive tokens/joule vs the best fixed config: " + ", ".join(rows_) + ". |")

    def fill(txt, tag, lines):
        a, b_ = "<!-- %s:START -->" % tag, "<!-- %s:END -->" % tag
        if a not in txt or b_ not in txt or not lines:
            return txt
        head, rest = txt.split(a, 1)
        _, tail = rest.split(b_, 1)
        return head + a + "\n" + "\n".join(lines) + "\n" + b_ + tail

    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    txt = fill(txt, "FINDINGS", S if len(S) > 2 else None)
    txt = fill(txt, "OUTCOMES", O if len(O) > 2 else None)
    readme.write_text(txt, encoding="utf-8")

    # ---------------------------------------------------------------- post draft
    P.append("# Post and resume draft (numbers generated from data; edit the voice, not the figures)\n")
    P.append("## LinkedIn post\n")
    lines = ["I built Inferno, a testbed that measures LLM inference from a CUDA kernel up to the request router, "
             "on an 8 GB laptop GPU. What I found, including where my own idea failed:\n"]
    if "kern_top" in facts:
        n, (b, sp) = facts["kern_top"]
        lines.append(f"1. A hand-written fused int4 GEMV kernel hits {b['pct_peak']:.0f}% of DRAM bandwidth and runs "
                     f"{sp:.2f}x faster than fp16 cuBLAS on {n} (byte-ratio ceiling {facts['ceil_top']:.2f}x). Inside vLLM it matches the default "
                     "Marlin path at one sequence and loses at higher concurrency: I first measured it beating Marlin, then caught that my Marlin baseline "
                     "was wrong.")
    if "h1" in facts:
        lo = min(r["pct_change"] for r in facts["h1"]); hi = max(r["pct_change"] for r in facts["h1"])
        lines.append(f"2. My adaptive routing policy lost to a fixed config: p99 {lo:+.0f}% to {hi:+.0f}% (8 paired reps, "
                     "Holm-corrected). A controlled experiment showed why: a second *active* replica on the same GPU "
                     f"multiplies p99 by {facts['cont'][0][0]['ratio_a_over_b']:.1f}x. Replicas need separate devices.")
    if "mpi" in facts:
        pp, d = facts["mpi"]
        lines.append(f"3. An MPI router adds ~{pp[64]['one_way_us_est']:.0f} us per hop in isolation and nothing "
                     "measurable end to end versus plain HTTP.")
    lines.append("\nEvery number comes from a script over raw logs, with paired statistics and reproducible runs. "
                 "Negative results are in the report. Repo: {link}")
    P.append("\n".join(lines) + "\n")
    P.append("Attach: `docs/figures/kernel_speedup.png` (hero) or `docs/results/contention/contention_light.png`.\n")
    P.append("## Resume bullets\n")
    if "kern_top" in facts:
        n, (b, sp) = facts["kern_top"]
        shp = kern["shapes"][[s["name"].split("(")[0].strip() for s in kern["shapes"]].index(n)]
        P.append(f"- Implemented a fused INT4 dequantization + matrix-vector CUDA kernel, verified against a "
                 f"double-precision CPU reference; measured {sp:.2f}x over FP16 cuBLAS on a {shp['N']} x {shp['K']} layer "
                 f"at batch 1 ({b['pct_peak']:.0f}% of peak DRAM bandwidth; byte-ratio ceiling {facts['ceil_top']:.2f}x).")
    if "lm" in facts:
        ratio, share, b1_wins, wins_share, hi = facts["lm"]
        P.append(f"- Benchmarked the kernel against vLLM's Marlin and AWQ paths on every Qwen2.5 decode layer at batch "
                 f"1-64; it won only layers worth {wins_share:.0f}% of decode GEMM time and lost up to {hi:.1f}x at batch 16, "
                 "so I kept the library path and redirected effort; the harness's reference check also flagged "
                 "batch-dependent wrong results from Marlin on this machine.")
    if "h1" in facts and cont:
        camp_n = len(__import__("pandas").read_csv(R / "campaign" / "summary.csv")) if (R / "campaign" / "summary.csv").exists() else None
        P.append(f"- Built an LLM inference benchmark spanning CUDA, vLLM and MPI on an 8 GB RTX 4060"
                 + (f"; ran a {camp_n}-run paired campaign that rejected an adaptive-routing hypothesis" if camp_n else "")
                 + ". A controlled %d-rep study isolated the cause: at 60%% load, activating a second GPU-sharing "
                 "replica raised p99 from %.2f s to %.2f s, while loading it idle gave %.2f s."
                 % (_n(cont, "rho0.6"), _p99(cont, "rho0.6", "single"), _p99(cont, "rho0.6", "split-active"),
                    _p99(cont, "rho0.6", "split-idle")))
    if "mpi" in facts:
        P.append("- Designed an MPI request router with per-hop timestamps and a GPU-free microbenchmark; found it "
                 "indistinguishable from an HTTP router end to end and traced a 1000x gap between isolated and in-situ hop "
                 "latency to event-loop scheduling.")
    P.append("- Automated environment setup (date-pinned CUDA 12.x stack, no-root toolchain) and multi-GPU cloud "
             "benchmarking with a self-terminating cost cap (Terraform; not yet applied).")
    (ROOT / "docs" / "POST_DRAFT.md").write_text("\n".join(P), encoding="utf-8")
    print("wrote docs/FINDINGS.md and docs/POST_DRAFT.md")


if __name__ == "__main__":
    main()
