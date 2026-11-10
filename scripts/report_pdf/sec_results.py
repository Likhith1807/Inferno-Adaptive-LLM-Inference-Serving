"""Chapters 8-12: implementation and verification, results, discussion, limitations and future work, conclusion."""
from __future__ import annotations

import json
from pathlib import Path

from . import data
from .refs import VERIFIED, cite
from .sec_lit import _Out
from .style import BUL, CALLOUT, FIG, H, NUM, P, TABLE

ROOT = data.ROOT
RES = data.RES
FIGS = ROOT / "docs" / "figures"


def _count_tests() -> tuple[int, int]:
    n_files = n_funcs = 0
    for p in sorted((ROOT / "tests").glob("test_*.py")):  # excludes __init__.py
        n_files += 1
        n_funcs += sum(1 for ln in p.read_text(encoding="utf8").splitlines() if ln.startswith("def test_"))
    return n_files, n_funcs


def _loc(sub: str, pattern: str = "*.py") -> tuple[int, int]:
    files = [p for p in (ROOT / sub).rglob(pattern) if "__pycache__" not in p.parts and p.name != "__init__.py"]
    return len(files), sum(len(p.read_text(encoding="utf8", errors="ignore").splitlines()) for p in files)


def _ladder(name_prefix: str):
    b = data.kernel()
    s = next(x for x in b["shapes"] if x["name"].startswith(name_prefix))
    k = {x["kernel"]: x["us_median"] for x in s["kernels"]}
    return s, k


def _fig(path, caption, width=1.0, max_h=None):
    return FIG(path, caption, width, max_h) if Path(path).exists() else []


def _json(rel):
    p = RES / rel
    return json.loads(p.read_text()) if p.exists() else None


def layer_facts():
    """Derived numbers for the layer-matrix and Marlin-probe sections (None if the data is absent)."""
    lm, mf = _json("layer_matrix/layer_matrix.json"), _json("marlin_fault/summary.json")
    if not lm:
        return None
    shapes = list(lm["shapes"])
    B = lm["batches"]
    cell = lambda n, i, b: lm["shapes"][n]["cells"].get(f"{i}/B{b}")  # noqa: E731
    ratio = {n: {b: cell(n, "custom", b)["us_median"] / cell(n, "marlin", b)["us_median"] for b in B if cell(n, "custom", b)}
             for n in shapes}
    t1 = {n: cell(n, "marlin", 1)["us_median"] for n in shapes}
    share = {n: 100 * v / sum(t1.values()) for n, v in t1.items()}
    wins = [n for n in shapes if cell(n, "custom", 1)["us_max"] < cell(n, "marlin", 1)["us_min"]]
    return {"lm": lm, "mf": mf, "shapes": shapes, "B": B, "ratio": ratio, "share": share, "wins": wins,
            "wins_share": sum(share[n] for n in wins), "max_ratio": max(max(r.values()) for r in ratio.values()),
            "cell": cell}


def _layer_matrix_section():
    f = layer_facts()
    if not f:
        return []
    lm, mf, shapes, B, ratio, share, cell = f["lm"], f["mf"], f["shapes"], f["B"], f["ratio"], f["share"], f["cell"]
    out = _Out(H(2, "Where the custom kernel earns its place: every layer, batch 1 to 64"))
    out.append(P(
        f"The in-situ comparison above covers batch 1 and a few concurrency levels. To decide whether the kernel belongs in the serving path at all, every "
        f"Qwen2.5-1.5B decode layer was timed at batch sizes {', '.join(map(str, B))} against vLLM&rsquo;s Marlin path, the classic AWQ path and fp16 cuBLAS, "
        f"each called exactly as vLLM calls it, on the <i>same</i> AutoAWQ weights converted once into each layout. Every cell is one CUDA graph holding one call per "
        f"weight copy, with copies rotated over at least {lm['rotate_mb']:.0f}&nbsp;MB (24&times; L2) so each call streams its weights from DRAM, replayed "
        f"{lm['reps']} times per trial over {lm['trials']} trials; launch overhead is excluded for every implementation alike. "
        "A cell counts as a win only when the winner&rsquo;s slowest trial beats the runner-up&rsquo;s fastest. The decision rule was fixed before the run: carry the kernel "
        "into vLLM only if it consistently wins on layers that account for meaningful decode time."))
    rows = [["Layer (N &times; K)", "share"] + [f"B={b}" for b in B]]
    for n in shapes:
        s = lm["shapes"][n]
        rows.append([f"{n} ({s['N']}&times;{s['K']})", f"{share[n]:.0f}%"]
                    + [("<b>%.2f</b>" % ratio[n][b] if ratio[n][b] < 1 else "%.2f" % ratio[n][b]) if b in ratio[n] else "n/a" for b in B])
    out.append(TABLE(rows, [0.22, 0.08] + [0.70 / len(B)] * len(B),
                     "Custom-kernel time divided by Marlin time (below 1, in bold, means the custom kernel is faster). Share: fraction of quantized-GEMM time per "
                     "decode step at batch 1. n/a: the custom kernel is instantiated only up to batch 16; the deployed path falls back above that.",
                     font=7.4, align=["LEFT"] + ["RIGHT"] * (len(B) + 1)))
    out += _fig(FIGS / "custom_vs_marlin.png", "Where the custom kernel earns its place: custom/Marlin time ratio by layer and batch size.", 0.9)
    out += _fig(FIGS / "layer_matrix.png", "Microseconds per call by batch size for each Qwen2.5-1.5B decode layer and implementation. Hollow markers: batch sizes at "
                "which Marlin&rsquo;s output was wrong in the correctness probe (next section).", 1.0)
    conv = {i: max(lm["shapes"][n]["conversion_ms"][i] for n in shapes) for i in ("custom", "marlin")}
    hi16 = max(r[16] for r in ratio.values() if 16 in r)
    out.append(P(
        f"At batch 1 the custom kernel clearly wins only on {' and '.join(f['wins']) or 'no layer'} ({f['wins_share']:.0f}% of decode GEMM time) and ties on the two large "
        f"layers that carry {share['gate_up'] + share['down']:.0f}% of it. From batch 4 it is {min(r[4] for r in ratio.values()):.1f} to {max(r[4] for r in ratio.values()):.1f}&times; "
        f"slower, rising to {hi16:.1f}&times; at batch 16: its time grows almost linearly with the batch (each weight is multiplied on CUDA cores once per token), while "
        "Marlin&rsquo;s tensor-core path stays flat to batch 16 because weight traffic, not arithmetic, sets its time. The one-time layout conversion is negligible for both "
        f"(at most {conv['custom']:.0f}&nbsp;ms per layer for the custom layout and {conv['marlin']:.0f}&nbsp;ms for Marlin&rsquo;s, paid once at load). "
        f"Every implementation matched the dequantize-and-matmul reference (custom kernel worst {max(lm['shapes'][n]['max_rel_err']['custom'] for n in shapes):.0e}) "
        "except one Marlin cell, which led to the next section."))
    out.append(CALLOUT("Decision", "The built-in Marlin path already matches the custom kernel on the layers that dominate decode and is far faster at every batch above 2. "
                       "Following the pre-stated rule, the kernel is not made a serving default and the paired end-to-end serving test of a modified engine is not run; "
                       "the opt-in switch (<font name='Mono'>vllm_int4.install()</font>) remains as a research tool, and engineering effort moves to the routing experiment."))

    if mf:
        out += H(2, "Correctness probe: Marlin returned wrong results at some batch sizes")
        out.append(P(
            f"The reference check above flagged Marlin on one layer. Marlin was then called 40 times with identical inputs per cell, in four variants: back-to-back with "
            f"vLLM&rsquo;s shared workspace (as vLLM runs it), with NaN-filled allocations between calls, with a freshly zeroed workspace for every call, and with a device "
            f"synchronize between calls. Batches {', '.join(map(str, mf['faulty_batches']))} produced intermittent wrong results on every layer tested, with relative max errors "
            f"up to {mf['worst_rel_err_faulty']:.2f}; batches {', '.join(map(str, mf['clean_batches']))} were always correct (worst {mf['worst_rel_err_clean']:.0e}). "
            "The custom kernel, the classic AWQ kernel and fp16 are deterministic and correct on the same inputs. A freshly zeroed workspace does not remove the fault, "
            "so stale lock state in the shared workspace is not the explanation; the cause was not established."))
        rows = [["Layer", "variant"] + [f"B={b}" for b in mf["batches"]]]
        for s in mf["shapes"]:
            for v in ("plain", "fresh_ws"):
                rows.append([s["name"], v] + [str(next((r["bad"] for r in s["rows"] if r["B"] == b and r["variant"] == v), "")) for b in mf["batches"]])
        out.append(TABLE(rows, [0.14, 0.12] + [0.74 / len(mf["batches"])] * len(mf["batches"]),
                         "Marlin calls with wrong output (relative max error above 1e-2) out of 40 identical calls. plain: shared workspace as in vLLM; fresh_ws: newly zeroed "
                         "workspace per call. All variants: docs/results/marlin_fault/report.md.", font=7.4, align=["LEFT", "LEFT"] + ["RIGHT"] * len(mf["batches"])))
        out += _fig(FIGS / "marlin_fault.png", "Marlin calls with wrong output by batch size and layer (shared workspace, 40 calls per bar).", 0.92)
        e2e = mf.get("e2e") or {}
        out.append(P(
            f"<b>Status: {mf['status']}.</b> vLLM pads decode batches to captured sizes that include 32 and 64, so if the fault is general it would corrupt real serving "
            "at moderate concurrency; it may equally be specific to WSL2, this driver (566.07) or this laptop GPU. "
            + ("The real-checkpoint check is reported in docs/results/marlin_fault/report.md. " if e2e else
               "The real-checkpoint check (greedy decode of the AWQ model at batch 1 versus 32/64, against the classic AWQ path) was written but could not run on the "
               "development machine during this revision, and no second machine was available. ")
            + "It is therefore not reported upstream, and the Marlin timings at the affected batch sizes in the previous section carry this caveat. "
              "<font name='Mono'>make marlin-probe</font> and <font name='Mono'>make quick-bench</font> reproduce it; docs/REPRODUCE.md asks independent testers to report either outcome."))
    return out


# ============================================================================================ chapter 8
def chapter_implementation():
    out = _Out(H(1, "Implementation and instrument verification"))
    out.append(P(
        "A result is only as reliable as the instrument that produced it. This chapter summarizes the implementation and, more importantly, "
        "how each instrument was verified before its output was used."))
    out += H(2, "Code base")
    rows = [["Component", "Path", "Files", "Lines"]]
    for label, sub, pat in [
        ("CUDA kernel and benchmark", "src/inferno/kernels", "*.cu*"),
        ("Python package", "src/inferno", "*.py"),
        ("Tests", "tests", "*.py"),
        ("Scripts", "scripts", "*.py"),
        ("Showcase site", "demo", "*.*"),
    ]:
        nf, nl = _loc(sub, pat)
        rows.append([label, f"<font name='Mono'>{sub}</font>", str(nf), f"{nl:,}"])
    out.append(TABLE(rows, [0.32, 0.36, 0.13, 0.19], "Size of the code base (counted when this report was built).", font=8.4, align=["LEFT", "LEFT", "RIGHT", "RIGHT"]))
    out.append(P(
        "The <font name='Mono'>.cu</font>/<font name='Mono'>.cuh</font> count includes the kernel header, the standalone benchmark and an earlier naive "
        "dequantization kernel that predates this study. The <font name='Mono'>scripts</font> line includes this report&rsquo;s own build modules."))
    out += H(2, "Reproducibility of the environment")
    out.append(BUL([
        "<b>Date-pinned dependency resolution.</b> An unpinned <font name='Mono'>vllm</font> install resolved to a PyTorch build for CUDA 13, which "
        "the installed driver (CUDA 12.7) cannot run. Resolution is now bounded by an exclude-newer date, and the versions used are recorded in "
        "<font name='Mono'>hardware.json</font> next to every run.",
        "<b>Environment doctor.</b> A check script confirms GPU visibility, CUDA toolchain, Open MPI, and package versions before any experiment.",
        "<b>Storage.</b> The WSL virtual disk was relocated to a larger drive after the system drive filled and produced read-only-filesystem errors mid-run.",
        "<b>Recorded provenance.</b> Every run directory stores the hardware and package snapshot and the exact configuration used.",
    ]))
    out += H(2, "Test suite")
    nfiles, nfuncs = _count_tests()
    out.append(P(
        f"The repository contains {nfuncs} test functions in {nfiles} files (the last full run in WSL2 collected 81 test cases, because some are parametrized, and all passed), "
        f"including a test that builds and runs the CUDA benchmark; on the Windows host the GPU test is skipped, because it needs a Linux binary. "
        f"Tests are grouped by what they protect:"))
    out.append(TABLE([
        ["Area", "What the tests check"],
        ["Metrics", "Derived latencies; TPOT undefined for single-token outputs; warm-up and error exclusion; goodput counts only requests meeting both SLO bounds; throughput includes drain time; empty run does not crash."],
        ["Workload", "Increasing arrival times; positive lengths; more requests during burst windows; reproducibility under the same seed; bimodal mixed-length profile; empirical-trace sampling."],
        ["Client", "Streamed tokens and TTFT recorded; fallback to chunk counting; HTTP errors recorded, not raised; open-loop pacing; prompt length; drain timeout cancels stragglers and records failures."],
        ["Routing policies, controller and live router", "Pinned policy uses only its replica; adaptive escalates under a burst and uses both tiers; static balancer spreads a burst; escalation under overload and de-escalation under light load; conservative cold-start default."],
        ["MPI routing", "Least-outstanding balancing and tie-breaking; idle replica preferred; overhead split in microseconds; HTTP baseline balances across replicas."],
        ["Statistics and aggregation", "Confidence intervals; paired test detects a real difference and reports none when there is none; best fixed ignores runs with errors; a clear win, a null result and a regression are each reported as such; Holm adjustment monotone and bounded."],
        ["Calibration and reporting", "Capacity counts only completions inside the window; derived rates are fractions of capacity with bursts above it; mean output length reflects the mixture; generated report contains tables, figures and the H<sub>1</sub> verdict and states a null result plainly."],
        ["Power and hardware", "Trapezoidal energy integration; too few samples handled; live sampling; multi-GPU ticks summed; hardware snapshot has the core fields."],
        ["Kernel", "Builds the benchmark and runs it against the CPU reference (Linux only)."],
    ], [0.24, 0.76], "What the test suite covers.", font=8))
    out.append(CALLOUT("What the tests do not do",
                       "They verify the software against expected behaviour; they do not validate that the experiment measures what it claims. That "
                       "second job is done by the protocol in Chapter 7 (correctness gate, calibration, paired design)."))
    out += H(2, "Instrument verification steps performed")
    out.append(TABLE([
        ["Instrument", "Verification"],
        ["Kernel", "Double-precision CPU reference on every shape in every benchmark invocation; correctness flag stored per kernel; timing withheld on failure."],
        ["Kernel timer", "L2-defeating weight rotation; back-to-back launches; run-to-run spread reported beside every time."],
        ["Latency client", "Mock-server tests; range check on recorded latencies (this found the negative TTFT)."],
        ["Throughput metric", "Regression test with a run whose arrivals stop early and completions trail; the metric must not equal the offered rate."],
        ["Power sampling", "Unit tests on parsing and integration; number of samples per run stored and available for inspection."],
        ["MPI timestamps", "All ranks on one host share the monotonic clock; a unit test checks that the overhead is split into dispatch and result components in microseconds."],
        ["Report generation", "Tests build reports and check the verdict text for a win, a null result and a regression."],
    ], [0.22, 0.78], "How each measuring instrument was verified.", font=8))
    return out


# ============================================================================================ chapter 9
def chapter_results():
    F = data.facts()
    k = F["kern"]
    out = _Out(H(1, "Results"))
    out.append(P(
        "All tables in this chapter are generated from the raw files under <font name='Mono'>docs/results</font>. Values are means with 95% "
        "confidence intervals unless stated. The chapter reports outcomes; interpretation is in Chapter 10."))

    # ------------------------------------------------------------------ 9.1 kernel
    out += H(2, "Kernel study")
    out += H(3, "Correctness")
    b = data.kernel()
    worst = max(x.get("max_rel_err", 0) for s in b["shapes"] for x in s["kernels"])
    allok = all(x["correct"] for s in b["shapes"] for x in s["kernels"])
    out.append(P(
        f"All kernels on all {len(b['shapes'])} shapes matched the CPU reference in every one of {F['kernel_runs']} benchmark invocations "
        f"({'no failures' if allok else 'FAILURES PRESENT'}); the worst relative error recorded is {worst:.2e}, which is consistent with fp16 output rounding."))
    out += H(3, "Measurement corrections")
    out.append(P(
        f"Two reference measurements accompany every result. A read-only streaming kernel over 512&nbsp;MB reaches {F['stream_gbps']:.0f}&nbsp;GB/s, "
        f"{100 * F['stream_gbps'] / F['peak_gbps']:.1f}% of the {F['peak_gbps']:.0f}&nbsp;GB/s computed from the reported memory clock; results are given against both. "
        f"And the weight-rotation footprint of the benchmark was corrected. The first version rotated through 192&nbsp;MB and reported a bandwidth above the theoretical peak "
        f"for one kernel, which prompted a sweep: a larger footprint slowed the same kernel by about 6% while leaving cuBLAS unchanged, so the smaller footprint had let the L2 "
        f"serve part of the traffic. All kernel results in this report use 768&nbsp;MB."))
    out.append(TABLE(data.flush_rows(), [0.25, 0.4, 0.35], "Effect of the weight-rotation footprint on the v5 kernel (Llama-7B MLP shape) and on cuBLAS.", font=8,
                     align=["LEFT", "RIGHT", "RIGHT"]))
    out += H(3, "Timing by version")
    rows = data.kernel_rows()
    t1 = [[r[0], r[1], *r[2:11]] for r in rows]
    t1[0] = ["layer", "N x K", "cuBLAS fp16", "unfused", "v0", "v1", "v2", "v3", "v4", "v5", "v6"]
    out.append(TABLE(t1, [0.15, 0.135, 0.075, 0.075, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07], "Median kernel time in microseconds over 5 invocations (lower is better).", font=7,
                     align=["LEFT"] + ["RIGHT"] * 10))
    t2 = [[r[0], *r[11:18]] for r in rows]
    t2[0] = ["layer", "best", "best / cuBLAS", "byte-ratio ceiling", "% of ceiling", "% of theoretical BW", "% of streaming BW", "runs within 2%"]
    out.append(TABLE(t2, [0.2, 0.1, 0.11, 0.11, 0.1, 0.12, 0.12, 0.14], "Best custom kernel per shape versus fp16 cuBLAS, the byte-ratio ceiling and the two bandwidth references.", font=7.4,
                     align=["LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    out.append(TABLE(data.kernel_spread_rows(), [0.16, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.14], "Run-to-run spread ((max-min)/median of the run medians) and, in parentheses, how many of the 5 runs lie within 2% of the median.", font=6.6,
                     align=["LEFT"] + ["RIGHT"] * 7 + ["LEFT"]))
    out += _fig(FIGS / "kernel_speedup.png", "Speedup over fp16 cuBLAS by kernel version and layer; the dashed line is the per-shape byte-ratio ceiling.", 0.95)
    out += _fig(FIGS / "kernel_bandwidth.png", "Achieved DRAM bandwidth as a fraction of the theoretical peak.", 0.95)
    out.append(P(
        f"The best result is <b>{k['name'].replace('_', ' ')}</b> ({k['N']}&times;{k['K']}): {k['us']:.1f}&nbsp;&micro;s, {k['gbps']:.0f}&nbsp;GB/s, "
        f"{k['pct']:.1f}% of the theoretical and {100 * k['gbps'] / F['stream_gbps']:.0f}% of the streaming-read bandwidth, and {k['speedup']:.2f}&times; faster than cuBLAS fp16 "
        f"({k['cub_us']:.1f}&nbsp;&micro;s, {k['cub_pct']:.0f}% of peak), against a byte-ratio ceiling of {k['ceil']:.2f}&times;. The two small layers (1 to 2&nbsp;MB of weights, about 12 to "
        f"15&nbsp;&micro;s) are dominated by launch and clock jitter, and no ranking is claimed for them. On the large layers at least four of five runs agree within 2%."))
    out += H(3, "What each optimization stage bought")
    stable = []
    for sh in data.kernel()["shapes"]:
        kk = {x["kernel"]: x for x in sh["kernels"]}
        if sh["int4_bytes"] > 5e6 and all(kk[n].get("runs_within_2pct", 5) >= 4 for n in data.CUSTOM + ("cublas_fp16",)):
            stable.append(sh["name"].split("(")[0].strip())
    rows = [["shape", "v0 to v1 (coalescing)", "v1 to v2 (wide loads + smem)", "v2 to v3 (padding)", "v3 to v4 (algebraic)", "v4 to v5 (register reuse)", "v5 to v6 (16 lanes)"]]
    for pre in stable:
        s_, t = _ladder(pre)

        def ch(a_, c_):
            r = t[a_] / t[c_]
            return f"{r:.2f}&times; faster" if r >= 1 else f"{1 / r:.2f}&times; slower"
        rows.append([pre.replace("_", " "), ch("int4_v0_naive", "int4_v1_warp"), ch("int4_v1_warp", "int4_v2_vec"), ch("int4_v2_vec", "int4_v3_pad"),
                     ch("int4_v3_pad", "int4_v4_alg"), ch("int4_v4_alg", "int4_v5_reuse"), ch("int4_v5_reuse", "int4_v6_lane16")])
    out.append(TABLE(rows, [0.19, 0.13, 0.15, 0.12, 0.13, 0.15, 0.13], "Effect of each stage on the shapes where at least 4 of 5 runs agree within 2% for every version (adjacent-version ratios of medians).", font=7.2))
    out.append(P(
        "The ladder is not monotone. Widening the loads and staging activations in shared memory (v2) made the kernel <i>slower</i> than v1 on the large "
        "layers, and removing the predicted bank conflict with padding (v3) recovered it. The algebraic rewrite (v4) helped on the Llama K&nbsp;=&nbsp;4096 shape "
        "and hurt on the Qwen shapes. Three explanations for the v4 slowdown were tested and rejected "
        "(streaming loads on short rows, a per-block prologue, late issue of scale loads). Two later changes removed it: dropping shared-memory staging altogether "
        "(v5: activations read through the read-only cache, each weight chunk decoded once into registers) and giving every lane the same number of chunks on 48-chunk rows "
        "(v6). That is consistent with the two remaining candidate explanations (the staging phase and the lane imbalance), but no profiler counter has confirmed which one is responsible. "
        "Loading several chunks per lane before consuming them never helped once lanes were balanced."))
    out += H(3, "Small batches")
    hdr = ["B"] + [data.batch_rows(i)[1].replace("_", " ") for i in range(3)] + ["byte-ratio ceiling (first shape)"]
    br = [data.batch_rows(i)[0] for i in range(3)]
    rows = [hdr]
    for j in range(1, len(br[0])):
        rows.append([br[0][j][0], br[0][j][3], br[1][j][3], br[2][j][3], br[0][j][4]])
    out.append(TABLE(rows, [0.1, 0.22, 0.22, 0.22, 0.24], "Speedup of the custom kernel over fp16 cuBLAS (same B) for B activation vectors; median of 3 invocations, all results matched the CPU reference.", font=8,
                     align=["RIGHT"] * 5))
    out.append(P(
        "The kernel keeps a clear advantage through B of about 4 to 6, is roughly at break-even around B of 7 to 8, and is slower than fp16 cuBLAS beyond that. "
        "Per weight it costs about three instructions per batch element on CUDA cores, so it leaves the memory-bound regime early; a tensor-core kernel is needed for larger batches. "
        "The result is therefore a batch-1 and very-small-batch result (for example speculative decoding), not a general replacement."))

    # ------------------------------------------------------------------ 9.2 library
    out += H(2, "Comparison with vLLM&rsquo;s kernels, end-to-end decode and server capacity")
    out += H(3, "Kernel time in situ")
    lib = data.library_rows()
    out.append(TABLE(lib, [0.09, 0.12, 0.14, 0.14, 0.13, 0.12, 0.13, 0.13], "Custom kernel versus the vLLM kernels, in situ (CUDA-graph decode, kernel time in microseconds, median of 3 fresh processes; ratio below 1 means the custom kernel is faster).", font=7.4,
                     align=["LEFT", "LEFT"] + ["RIGHT"] * 6))
    out.append(TABLE(data.insitu_mode_rows(), [0.26, 0.11, 0.12, 0.12, 0.13, 0.13, 0.13], "The same kernels under both engine modes (in-situ median of 3 fresh processes, microseconds).", font=7.6,
                     align=["LEFT", "LEFT"] + ["RIGHT"] * 5))
    sp_ = data.insitu_spread()
    slow = data.insitu_slow()
    rm = [float(r[6].rstrip("x")) for r in lib[1:]]
    ra = [float(r[7].rstrip("x")) for r in lib[1:]]
    cal = F["cal"]
    out.append(P(
        f"In CUDA-graph decode, the custom kernel takes between {min(rm):.2f} and {max(rm):.2f} of Marlin&rsquo;s time on the four layer shapes (a ratio above 1 means slower) and between "
        f"{min(ra):.2f} and {max(ra):.2f} of the classic AWQ kernel&rsquo;s. Marlin is at the bandwidth limit on the large layers at batch 1. <b>An earlier version of this report claimed the custom "
        f"kernel beat Marlin on all four shapes; that claim was wrong.</b> It rested on a Marlin timing about twice as slow as the values above, which could not be reproduced: end-to-end throughput "
        f"disagreed with it, and re-profiling Marlin in the same session under both engine modes gave the current values. The cause of the earlier number was not established, and the file was retired. One plausible cause: {slow[0]} of the {slow[1]} profiling processes run for this report had every kernel, "
        f"FlashAttention included, running 2 to 5 times slower than in their sibling processes (a GPU-wide slowdown, for example throttling), and a single-process measurement taken during such an episode "
        f"would look exactly like the retracted one; this could not be verified for the old run. Each in-situ time is therefore the median of three fresh processes, and even among the remaining processes the largest "
        f"process-to-process spread was {sp_[0]:.0f}% ({sp_[1].replace('awq_inferno', 'custom').replace('awq_marlin', 'Marlin').replace('awq', 'classic AWQ')}, {sp_[2]})."))
    out += H(3, "End-to-end decode throughput")
    out.append(TABLE(data.e2e_rows(), [0.34, 0.165, 0.165, 0.165, 0.165], "Decode throughput in output tokens per second (mean of three process repetitions with 95% interval; parentheses: paired ratio to vLLM&rsquo;s default Marlin path).", font=7.6,
                     align=["LEFT"] + ["RIGHT"] * 4))
    out += _fig(RES / "e2e_kernel" / "e2e_light.png", "End-to-end decode throughput in vLLM 0.11 with CUDA graphs.", 0.95)
    e = F["e2e"]
    r1 = F["e2e_ratio_b8"]["1"]
    r4 = F["e2e_ratio_b8"]["4"]
    lc = F["e2e_layer_check"]
    ag = F["e2e_agreement"]
    all_same = all(v["identical_total"] for v in ag.values())
    agtxt = "; ".join(f"{k_.replace('inferno_', 'custom ')}: {v['identical_prefix']} of {v['tokens']} tokens identical" for k_, v in ag.items())
    out.append(P(
        f"With one sequence the custom kernel decodes at {e['inferno_b8']['1']:.0f} tokens/s against {e['awq_marlin']['1']:.0f} for Marlin (paired ratio {r1['mean']:.2f}, range {r1['min']:.2f} to {r1['max']:.2f}), "
        f"{e['awq_classic']['1']:.0f} for the classic AWQ kernel and {e['fp16']['1']:.0f} for fp16. That is parity with Marlin, an improvement over the classic kernel, and about "
        f"{e['inferno_b8']['1'] / e['fp16']['1']:.1f}&times; fp16. With four concurrent sequences the paired ratio to Marlin is {r4['mean']:.2f}: Marlin&rsquo;s tensor-core kernel is faster once several sequences decode together. "
        f"A hybrid that uses the custom kernel only for steps of at most two tokens follows the same pattern. All {lc['n_layers_checked']} patched layers of the loaded checkpoint matched vLLM&rsquo;s own "
        f"dequantize-and-matmul to a maximum relative error of {lc['max_rel_err']:.1e}. Greedy text agreement on the first prompt: {agtxt}. "
        + ("Every configuration, stock or custom, produced the identical 256-token continuation. " if all_same else
           "Different kernels accumulate in different orders, so a token stream can diverge once two near-equal logits swap; agreement over a prefix is the expected outcome, not a defect. ")
        + "This is one prompt, so it is a sanity check on the integration and not an accuracy evaluation."))
    out += H(3, "Server-level capacity")
    out.append(P(
        f"Saturated with the same open-loop request process, a vLLM server using the Marlin path sustained {cal['steady_tok']:.0f} output tokens/s against {cal['awq_tok']:.0f} for the classic AWQ "
        f"path, a ratio of {F['marlin_ratio']:.2f}&times;. A batch-1 kernel is a latency-sensitive, low-concurrency result and says little about saturated serving."))

    out += _layer_matrix_section()

    # ------------------------------------------------------------------ 9.3 campaign
    out += H(2, "Campaign: adaptive versus fixed routing")
    out.append(P(
        "The three tables below give, for each load pattern, the four policies over eight repetitions. Latencies are end-to-end in seconds; throughput is output tokens "
        "per second over the busy span; goodput is requests per second meeting both SLO bounds."))
    tabs = data.campaign_tables()
    for pat, cap in (("steady", "Steady load."), ("bursty", "Bursty load."), ("mixed_length", "Mixed-length load.")):
        out.append(TABLE(tabs[pat], [0.13, 0.04, 0.115, 0.115, 0.115, 0.115, 0.135, 0.115, 0.1],
                         f"{cap} Mean &plusmn; 95% CI over 8 paired repetitions.", font=6.8, align=["LEFT"] + ["RIGHT"] * 8))
    rows, h1 = data.h1_rows()
    rows[0] = ["load pattern", "best fixed", "n", "adaptive p99 (s)", "best-fixed p99 (s)", "change", "Holm p", "d<sub>z</sub>", "thr. ratio"]
    out.append(TABLE(rows, [0.14, 0.13, 0.04, 0.13, 0.13, 0.09, 0.09, 0.07, 0.1], "Pre-specified test H<sub>1</sub>: adaptive versus the best fixed configuration (paired, Holm-corrected).", font=7.4,
                     align=["LEFT", "LEFT"] + ["RIGHT"] * 7))
    out += _fig(ROOT / "docs/results/campaign/figures/p99_by_pattern_light.png", "End-to-end p99 latency by policy and load pattern (mean and 95% CI).", 0.92)
    out.append(P(
        "In every pattern all eight pairs point the same way (the smallest Wilcoxon <i>p</i> attainable at <i>n</i>&nbsp;=&nbsp;8, 0.0078, "
        "which Holm-adjusts to 0.023 across three patterns). The adaptive policy is significantly <b>worse</b> than the best fixed configuration on p99 in all three "
        "patterns, and its throughput is between 94% and 96% of the best fixed policy&rsquo;s, so it also fails the non-inferiority guard. "
        "<b>H<sub>1</sub> is rejected.</b>"))
    out += H(3, "Headroom analysis")
    hr = data.headroom()
    rows = [["load pattern", "fixed-fast p99 (s)", "fixed-bulk p99 (s)", "better fixed tier", "gain of per-pattern choice over " + hr["single_fixed_tier"], "adaptive vs better fixed"]]
    for r in hr["rows"]:
        rows.append([r["pattern"].replace("_", " "), f"{r['p99_fast']:.2f}", f"{r['p99_bulk']:.2f}", r["oracle_tier"], f"{r['oracle_gain_vs_single_pct']:.1f}%", f"{r['adaptive_vs_oracle_pct']:+.0f}%"])
    out.append(TABLE(rows, [0.16, 0.15, 0.15, 0.16, 0.22, 0.16], "Post hoc headroom check on the campaign data: how much could any policy that chooses the better fixed tier per pattern gain?", font=7.6,
                     align=["LEFT", "RIGHT", "RIGHT", "LEFT", "RIGHT", "RIGHT"]))
    out.append(P(
        f"The better of the two fixed tiers differs by pattern, but the tier that is best overall ({hr['single_fixed_tier']}) is within {hr['max_headroom_pct']:.1f}% of the per-pattern best on every pattern. "
        f"On this shared-GPU pool a pattern-aware switch between the tiers could therefore have gained at most {hr['max_headroom_pct']:.1f}% of p99; the test of H<sub>1</sub> "
        f"was close to unwinnable independent of the co-residency penalty of the policy actually tested. This analysis was done after seeing the data and is explanatory."))
    out += H(3, "Energy and throughput")
    df = data.campaign_df()
    rows = [["load pattern", "adaptive tokens/J", "best-fixed tokens/J", "change", "adaptive throughput", "best-fixed throughput", "ratio"]]
    for r in h1:
        pat, bf = r["pattern"], r["best_fixed"]
        a, b = df[(df.pattern == pat) & (df.config == "adaptive")], df[(df.pattern == pat) & (df.config == bf)]
        rows.append([pat.replace("_", " "), f"{a.tokens_per_joule.mean():.1f}", f"{b.tokens_per_joule.mean():.1f}",
                     f"{100 * (a.tokens_per_joule.mean() / b.tokens_per_joule.mean() - 1):+.1f}%",
                     f"{a.throughput_tok_per_s.mean():.0f}", f"{b.throughput_tok_per_s.mean():.0f}", f"{a.throughput_tok_per_s.mean() / b.throughput_tok_per_s.mean():.2f}"])
    out.append(TABLE(rows, [0.15, 0.15, 0.16, 0.1, 0.16, 0.18, 0.1], "Energy efficiency and throughput, adaptive versus the best fixed configuration for p99 (means over 8 repetitions).", font=7.8,
                     align=["LEFT"] + ["RIGHT"] * 6))
    share = df[df.config == "adaptive"].groupby("pattern")[["share_tier0", "share_tier1"]].mean()
    out.append(P(
        "Mean fraction of requests the adaptive controller sent to tier 0 / tier 1: " +
        "; ".join(f"{p.replace('_', ' ')} {100 * r.share_tier0:.0f}% / {100 * r.share_tier1:.0f}%" for p, r in share.iterrows()) +
        ". Its behaviour is close to a load balancer once the controller has escalated."))
    out += _fig(ROOT / "docs/results/campaign/figures/pareto_bursty_light.png", "Latency-throughput trade-off per policy under bursty load (each point is one repetition).", 0.8)

    # ------------------------------------------------------------------ 9.4 contention
    out += H(2, "Mechanism study: two active servers on one GPU")
    crow, comp = data.contention_rows()
    out.append(TABLE(crow, [0.12, 0.16, 0.06, 0.24, 0.22, 0.2], "Contention experiment cells (5 paired repetitions).", font=7.8, align=["LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    out.append(TABLE(comp, [0.13, 0.3, 0.2, 0.15, 0.22], "Paired ratios (first over second condition).", font=7.8, align=["LEFT", "LEFT", "LEFT", "RIGHT", "RIGHT"]))
    out += _fig(RES / "contention" / "contention_light.png", "Contention experiment summary.", 0.95)
    a, i = F["cont_active"], F["cont_idle"]
    out.append(P(
        f"With a second replica loaded but idle, p99 differs from the single-replica reference by only {100 * (i[0] - 1):+.1f}% and {100 * (i[1] - 1):+.1f}% "
        f"at the two loads. With the second replica <i>active</i> at the same total offered load, p99 rises by a factor of {a[0]:.2f} and {a[1]:.2f}. The "
        f"cost comes from activity, not from memory being split. With five pairs the Wilcoxon test cannot go below 0.0625, so the strength of this "
        f"conclusion rests on the effect size and consistency across pairs, not on a <i>p</i>-value."))

    # ------------------------------------------------------------------ 9.5 mpi
    out += H(2, "MPI routing")
    pp, ds = F["mpi_pp"], F["mpi_disp"]
    rows = [["message size", "round trip p50 (&micro;s)", "round trip p99 (&micro;s)", "one-way estimate (&micro;s)"]]
    for n in sorted(pp):
        r = pp[n]
        lab = f"{n} B" if n < 1024 else f"{n // 1024} KB"
        rows.append([lab, f"{r['rtt_us_p50']:.1f}", f"{r['rtt_us_p99']:.1f}", f"{r['one_way_us_est']:.1f}"])
    out.append(TABLE(rows, [0.22, 0.26, 0.26, 0.26], "Ping-pong latency between two ranks on one host.", font=8.2, align=["LEFT", "RIGHT", "RIGHT", "RIGHT"]))
    rows = [["workers", "jobs / s", "round trip p50 (&micro;s)", "round trip p99 (&micro;s)", "note"]]
    for w in sorted(ds):
        r = ds[w]
        rows.append([str(w), f"{r['jobs_per_s']:,.0f}", f"{r['rtt_us_p50']:.0f}", f"{r['rtt_us_p99']:.0f}", "oversubscribed" if w >= F["cores"] - 1 else ""])
    out.append(TABLE(rows, [0.12, 0.2, 0.24, 0.24, 0.2], f"Dispatch-loop throughput through rank 0 (payload 512 B, window 64; {F['cores']} logical cores).", font=8.2, align=["RIGHT", "RIGHT", "RIGHT", "RIGHT", "LEFT"]))
    out.append(P(
        "One-way latency is a few microseconds for messages up to 4&nbsp;KB, about 13&nbsp;&micro;s at 32&nbsp;KB and a few hundred microseconds at 256&nbsp;KB. "
        "The jump is steeper than bandwidth alone would explain, consistent with a switch of message protocol; the threshold was not determined. "
        "Dispatch throughput falls as workers are added, and the largest worker count, which is oversubscribed, is the slowest."))

    out += H(2, "Scaling and MPI versus HTTP with real replicas")
    rows = [["router", "replicas", "n", "throughput (tok/s)", "p99 (s)", "efficiency vs 1 replica"]]
    for (rt, rep), c in sorted(F["scal_cells"].items()):
        rows.append([rt.upper(), str(rep), str(c["n"]), data.ci_str(c["throughput"], c["throughput_ci"], 0), data.ci_str(c["p99"], c["p99_ci"], 1), f"{c['efficiency_vs_1x']:.2f}"])
    out.append(TABLE(rows, [0.12, 0.12, 0.07, 0.27, 0.2, 0.22], "Scaling under saturating load with all replicas on one GPU (paired repetitions).", font=8.2, align=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    rows = [["replicas", "MPI vs HTTP throughput", "paired t-test p", "Wilcoxon p", "pairs"]]
    for r in F["scal_mv"]:
        rows.append([str(r["replicas"]), f"{r['pct_diff_mpi_vs_http']:+.2f}%", f"{r['p_ttest']:.3f}", f"{r['p_wilcoxon']:.3f}", str(r["n_pairs"])])
    out.append(TABLE(rows, [0.15, 0.3, 0.2, 0.2, 0.15], "MPI router compared with the HTTP router (both tests must agree for significance).", font=8.2, align=["RIGHT"] * 5))
    ov = F["scal_over"]
    out.append(P(
        f"Adding a second replica on the same GPU <i>reduces</i> total throughput (efficiency "
        f"{F['scal_cells'][('http', 2)]['efficiency_vs_1x']:.2f} of linear scaling). MPI and HTTP routing differ by less than 1% in throughput; "
        f"with four pairs the Wilcoxon test cannot go below 0.125, and where the <i>t</i>-test alone is borderline the two tests disagree, so no difference "
        f"is claimed. Inside the real router, the measured dispatch hop has a median of "
        f"{ov[0]['mpi_dispatch_us_p50'] / 1000:.1f}&nbsp;ms, about a thousand times the microbenchmark: the worker&rsquo;s single asyncio loop is also parsing "
        f"streaming responses for many requests, so hop time includes queueing behind that work. Even so it is small next to multi-second request latencies at this load."))

    # ------------------------------------------------------------------ 9.7 sweep
    out += H(2, "Quantization and batch-cap sweep")
    out.append(TABLE(data.sweep_rows(), [0.12, 0.07, 0.22, 0.12, 0.12, 0.15, 0.2], "Sweep at one steady load (3 repetitions per cell; unfinished requests 60 s after the last arrival counted as failures).", font=7.8,
                     align=["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    out += _fig(FIGS / "sweep_steady_light.png", "Sweep results: latency, throughput and energy efficiency by configuration.", 0.95)
    out.append(P(
        f"Cells with batch cap 1 could not sustain the load, and some completed none of their requests inside the window, so their latency entries are "
        f"not comparable. Among cells that finish, int4 lowers median latency and raises tokens per joule relative to fp16 at the same offered load; the "
        f"exact figures are in the table. Restarting a server to change configuration took a median of {F['cold']:.0f}&nbsp;s of cold start, several times "
        f"the 10&nbsp;s burst duration used in the bursty pattern. Cold starts above 100&nbsp;s, which occurred on the first load of the fp16 and int8 weights, are excluded from that median."))
    return out


# ============================================================================================ chapter 10-12
def chapter_discussion():
    F = data.facts()
    LF = layer_facts()
    k = F["kern"]
    out = _Out(H(1, "Discussion"))
    out += H(2, "Answers to the research questions")
    out.append(TABLE([
        ["Question", "Answer supported by the data"],
        ["RQ1 Kernel vs roofline and vLLM", f"On the best bandwidth-bound layer the kernel reached {k['pct']:.0f}% of theoretical DRAM bandwidth ({100 * k['gbps'] / F['stream_gbps']:.0f}% of the measured streaming-read bandwidth) and {k['speedup']:.2f}&times; cuBLAS. Inside vLLM with CUDA graphs it decodes at parity with the default Marlin path at one sequence ({F['e2e_ratio_b8']['1']['mean']:.2f}&times;), ahead of the classic AWQ kernel, and slower than Marlin with four sequences ({F['e2e_ratio_b8']['4']['mean']:.2f}&times;). It does not beat Marlin; an earlier claim that it did was a measurement error. Above a batch of about 8 a CUDA-core kernel loses to cuBLAS." + (f" Across every Qwen layer at batch 1 to 64 it wins only {'/'.join(LF['wins'])} at batch 1 ({LF['wins_share']:.0f}% of decode GEMM time) and is up to {LF['max_ratio']:.1f}&times; slower than Marlin, so Marlin stays the default." if LF else "")],
        ["RQ2 Adaptive vs fixed", "No. Adaptive routing was significantly worse on p99 in all three patterns, with lower throughput and lower tokens per joule. H<sub>1</sub> is rejected."],
        ["RQ3 Mechanism", f"Two active replicas time-slicing one GPU roughly double p99 ({F['cont_active'][0]:.2f}&times; and {F['cont_active'][1]:.2f}&times;) at fixed total load; an idle second replica costs about nothing."],
        ["RQ4 MPI", "Point-to-point latency is a few microseconds; end-to-end, MPI and HTTP routing are indistinguishable in this setup, and dispatch adds about two milliseconds inside the real router. Scaling on one GPU is negative because of co-residency, not because of MPI."],
        ["RQ5 Quantization, batch cap, reconfiguration", f"int4 improves latency and energy per token at fixed load; small batch caps cannot sustain the load; reconfiguration costs about {F['cold']:.0f}&nbsp;s."],
    ], [0.27, 0.73], "Research questions and answers.", font=8))
    out += H(2, "Status of the four hypotheses")
    out.append(TABLE([
        ["ID", "Outcome", "Evidence and qualifications"],
        ["H1", "Rejected", "Significantly worse p99 in all patterns; non-inferiority guard also failed. Applies to single-GPU replicas."],
        ["H2", "Partly supported", "Scaling on the shared GPU is sub-linear (efficiency below 1); MPI overhead is measured (microseconds isolated, about 2 ms in situ). No replica count at which MPI overhead erodes gains was reached, because two replicas were the maximum on one GPU."],
        ["H3", "Refuted as stated; explanation unresolved", "The kernel reaches the memory limit on large layers rather than a fraction of the library path, and is at parity with Marlin end to end at batch 1 (not a win). The layer matrix confirms it wins only small layers at batch 1. The part of H3 that requires profiler counters to explain differences was not completed."],
        ["H4", "Not supported", "The adaptive policy lowered tokens per joule by a few percent. The only efficiency gain measured came from quantization, which is a model change."],
    ], [0.07, 0.24, 0.69], "Hypothesis outcomes.", font=8))
    out += H(2, "Why the adaptive policy lost")
    out.append(NUM([
        "<b>The routing premise assumed independent capacity.</b> Sending some requests to the second replica only helps if the second replica adds capacity. On one "
        "GPU it does not: the device is shared, and total useful capacity does not grow.",
        "<b>Active co-residency is expensive.</b> The contention experiment shows that an active second server roughly doubles p99 at fixed load, while an idle "
        "one is free. The best fixed policies avoid this by using only one server.",
        "<b>The adaptive policy activates the second server.</b> After escalation it spreads requests across both replicas, so it inherits the penalty, and it tracks the "
        "static load balancer closely.",
        "<b>Smaller per-server batches</b> are a likely additional contributor: splitting the stream reduces what each server can batch. This was not separated from time-slicing in the experiment.",
    ]))
    out.append(P(
        f"This is consistent with the premise of disaggregation systems that put specialized replicas on separate devices {cite('distserve', 'splitwise')}. "
        f"The result does not say that adaptive routing is a bad idea; it says the idea was tested here in the one configuration where its precondition "
        f"(independent capacity per replica) fails. The prediction that follows, and which this study did not test, is that with one replica per GPU the penalty "
        f"disappears and the comparison becomes open."))
    out += H(2, "Why the kernel result should not be over-read")
    out.append(BUL([
        "The kernel result is <b>batch 1 and very small batches</b>, memory-bound, on one GPU. Above a batch of about 8 it loses to fp16 cuBLAS, and end to end Marlin is clearly faster from four sequences.",
        "<b>Parity with Marlin is not a win.</b> Marlin is at the bandwidth limit on the large layers, so there is no headroom to beat it at batch 1 on this GPU; the honest claim is that a from-scratch kernel reached the same place.",
        "Reaching the byte-ratio ceiling on one shape shows the kernel is near the limit for that shape, not that it is state-of-the-art for all shapes. The small layers sit far below the ceiling, consistent with launch and clock jitter dominating 12 to 15 microsecond kernels.",
        "The layer matrix makes this concrete: the only clear wins are the two small layers at batch 1, which carry about one seventh of the quantized-GEMM time.",
        "End-to-end decode also spends time outside the quantized layers (the fp16 output projection over the vocabulary, attention, normalization, sampling), so a faster int4 kernel moves tokens per second less than it moves kernel time.",
    ]))
    out += H(2, "Methodological observations")
    out.append(BUL([
        "Several defects in the audit trail (Section 7.10) were found only because an instrument was deliberately checked. The correctness gate, the drain-aware throughput, and calibration each changed reported results.",
        "The power argument mattered in practice: with fewer than seven pairs the pre-specified test could not have confirmed anything, and with seven only with no margin. The contention experiment, at five pairs, is therefore reported through effect sizes rather than <i>p</i>-values.",
        "Deriving loads from capacity prevented a class of results in which every policy looked bad because the offered load exceeded what the GPU could serve.",
        "Two of the errors overturned earlier claims: a benchmark that let the L2 serve part of the traffic (bandwidth figures were too high by about 6%), and a baseline timing that made the custom kernel look faster than Marlin. Both were caught by cross-checks (a reading above 100% of peak, and end-to-end throughput disagreeing with kernel times), not by review. A kernel comparison is only credible when it is closed at a second level of measurement.",
    ]))
    return out


def chapter_limits():
    out = _Out(H(1, "Limitations and future work"))
    out += H(2, "Limitations")
    out.append(BUL([
        "<b>One GPU, one model family, one engine.</b> Every conclusion is conditional on an RTX 4060 Laptop GPU, Qwen2.5-1.5B and vLLM 0.11.",
        "<b>Replicas share a GPU.</b> This is the dominant limitation for the routing result and the reason it should not be generalized to multi-GPU deployments.",
        "<b>No profiler counters.</b> The bank-conflict prediction and the causes of the v4 slowdown and of the v5/v6 gains are hypotheses (the counter permission was re-checked during the work and is still blocked).",
        "<b>The end-to-end comparison is one model on one GPU</b> with three process repetitions per configuration, so intervals are wide and the parity with Marlin is a statement about a few percent, not about small differences.",
        "<b>The Marlin correctness fault is observed on one machine.</b> It is reproducible there at the kernel level, but it has not been checked on the real checkpoint end to end or on a second GPU, so it is not attributed to vLLM.",
        "<b>No cloud run.</b> Infrastructure code with a cost cap is provided but was not applied; no multi-node or multi-GPU result is reported.",
        "<b>Synthetic workloads and quality not evaluated.</b> No accuracy or perplexity comparison across quantization levels was made.",
        "<b>Modest repetition counts.</b> Eight repetitions for the campaign, five for the mechanism study, four for scaling, three for the sweep; small effects are out of reach.",
        f"<b>Bibliography.</b> {len(VERIFIED)} references were verified against publisher or proceedings pages during preparation (marked &dagger;); the rest are cited from knowledge and should be confirmed before formal submission.",
        "<b>The dashboard</b> was checked with rendered screenshots rather than interactive sessions across browsers.",
    ]))
    out += H(2, "Future work, in order of expected value")
    out.append(NUM([
        "<b>Enable GPU performance counters and run Nsight Compute</b> on the kernel ladder (one-time driver setting, then <font name='Mono'>make kernel-profile</font>), to confirm or refute the bank-conflict and occupancy hypotheses and to explain the v4 slowdown.",
        "<b>Reproduce the Marlin fault elsewhere</b> (<font name='Mono'>make quick-bench</font> on another GPU, then the real-checkpoint probe). If it reproduces, report it upstream with the minimal script; if not, record it as machine-specific.",
        "<b>Run the cloud phase</b> with one replica per GPU. This is the experiment that tests the prediction in Chapter 10: with independent capacity, does load- and length-aware routing beat the best fixed configuration?",
        "<b>Try MPS</b> for concurrent kernel execution across the two replicas on one GPU, to see whether the co-residency penalty is reducible without extra hardware.",
        "<b>A tensor-core variant for batches of 8 to 64</b>, the range where the CUDA-core kernel loses to cuBLAS and to Marlin, so that one code path covers decode at every concurrency.",
        "<b>Extend the vLLM integration and upstream it as a discussion</b>: the GPTQ path, the quantized output projection (which needs an accuracy evaluation), and a benchmark issue that presents the measured parity and its limits.",
        "<b>Real traces and quality.</b> Replace synthetic lengths with a published conversation trace and add a task-accuracy check for the quantized checkpoints.",
        "<b>Additional adaptive mechanisms</b> such as migration of running requests, and predictive rather than reactive control.",
    ]))
    return out


def chapter_conclusion():
    F = data.facts()
    k = F["kern"]
    h1 = F["h1"]
    lo, hi = min(r["pct_change"] for r in h1), max(r["pct_change"] for r in h1)
    out = _Out(H(1, "Conclusion"))
    LF = layer_facts()
    e = F["e2e"]
    r1, r4 = F["e2e_ratio_b8"]["1"]["mean"], F["e2e_ratio_b8"]["4"]["mean"]
    out.append(P(
        f"This study built and measured an LLM-serving testbed spanning a CUDA kernel, its integration into vLLM, an MPI router and an adaptive routing controller on a single 8&nbsp;GB GPU, "
        f"with the protocol and statistics fixed in advance. " + ("Five" if LF else "Four") + f" findings stand. First, a hand-written fused int4 GEMV can reach the memory roofline "
        f"({k['pct']:.0f}% of theoretical bandwidth, {k['speedup']:.2f}&times; cuBLAS, against a {k['ceil']:.2f}&times; byte-ratio ceiling) on a large layer. Second, inside vLLM it is at parity "
        f"with the default Marlin path at one sequence ({e['inferno_b8']['1']:.0f} against {e['awq_marlin']['1']:.0f} tokens/s, ratio {r1:.2f}), ahead of the classic AWQ kernel and of fp16, and behind Marlin "
        f"with four concurrent sequences ({r4:.2f}); it does not beat the library. Third, the hypothesis that adaptive routing over pre-loaded replicas would beat the best fixed configuration was "
        f"<b>rejected</b>: p99 was {lo:.0f}% to {hi:.0f}% worse, a controlled experiment attributes this to two active servers contending for one GPU, and the headroom for any pattern-aware choice was at most "
        f"{F['headroom']['max_headroom_pct']:.1f}% anyway. Fourth, MPI routing is cheap in isolation and indistinguishable from HTTP routing end to end in this setup." + (f" Fifth, across every decode layer at batch 1 to 64 the kernel wins only layers worth {LF['wins_share']:.0f}% of decode GEMM time, so by the pre-stated rule Marlin stays the default; the same benchmark&rsquo;s reference check exposed batch-dependent wrong results from Marlin on this machine, a finding that awaits reproduction elsewhere." if LF else "")))
    out.append(P(
        "The main contribution is less any one number than the demonstration that each claim was tested with an instrument that had itself been checked, "
        "including two checks that overturned the study&rsquo;s own earlier claims, and that the study reports what it did not achieve: no profiler counters, no cloud run, no win "
        "over the production kernel, no independent reproduction yet, and a routing result that applies only to replicas sharing a device."))
    return out
