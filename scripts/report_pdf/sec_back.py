"""Appendices and references."""
from __future__ import annotations

from reportlab.platypus import Spacer

from . import data
from .refs import VERIFIED, reference_flowables
from .sec_lit import _Out
from .style import CODE, P, TABLE, UH


def appendices():
    ROOT = data.ROOT
    out = _Out(UH("Appendix A. Reproducing the study"))
    out.append(P(
        "The commands below are the Makefile targets used for the reported runs, run from the repository root inside WSL2 (Ubuntu) with the "
        "environment created by the setup scripts. Long targets run for tens of minutes to hours; the campaign alone is 96 runs of 75&nbsp;s plus server start-up."))
    out.append(CODE(
        "make setup           # date-pinned Python environment (uv), vLLM, PyTorch\n"
        "make toolchain       # CUDA 12.4 toolchain, gcc 12, Open MPI (micromamba)\n"
        "make test            # unit tests\n"
        "bash scripts/repeat_kernel_bench.sh 5   # ladder benchmark, five invocations, CPU-reference gate\n"
        "bash scripts/rerun_kernel_all.sh    # library + 5 ladder runs + 3 batch runs\n"
        "bash scripts/flush_sweep.sh         # rotation-size check\n"
        "bash scripts/insitu_graphs.sh       # in-situ kernel timing, both engine modes\n"
        "bash scripts/run_e2e_matrix.sh 3    # end-to-end decode matrix (vLLM integration)\n"
        "python scripts/headroom_analysis.py # post hoc routing headroom\n"
        "make mpi-bench       # MPI ping-pong and dispatch microbenchmarks\n"
        "make calibrate       # measure capacity, derive load rates\n"
        "make sweep           # quantization x batch-cap sweep\n"
        "make scaling         # MPI vs HTTP, 1 and 2 replicas\n"
        "make campaign        # 4 policies x 3 patterns x 8 repetitions\n"
        "make contention      # mechanism study\n"
        "make report          # regenerate tables, figures, verdicts\n"
        "make site            # regenerate the showcase dashboard\n"
        "make kernel-profile  # Nsight Compute (requires GPU counters enabled)"))
    out.append(P(
        "The environment is date-pinned, so that a re-run resolves the same package versions. Results depend on the GPU model, driver, and thermal state; "
        "numbers from another machine should be compared by shape (ratios, ceiling fractions) rather than by absolute value."))

    out += UH("Appendix B. Configuration")
    pool = (ROOT / "configs" / "pool_config.yaml").read_text(encoding="utf8")
    out.append(P("<b>Replica pool and controller thresholds</b> (<font name='Mono'>configs/pool_config.yaml</font>):"))
    pool = "\n".join(ln.split("#")[0].rstrip() for ln in pool.splitlines() if ln.split("#")[0].strip())
    out.append(CODE(pool))
    out.append(P("<b>Workload definition</b> and derived rates for the campaign are stored with the results as "
                 "<font name='Mono'>docs/results/campaign/config_snapshot.json</font>; the calibrated rates are tabulated in "
                 "Section 7.2."))
    hw = data.hardware()
    gpu = (hw.get("gpus") or [{}])[0]
    pk = hw.get("packages", {})
    rows = [["item", "value"], ["GPU", str(gpu.get("name", ""))], ["driver", str(gpu.get("driver_version", ""))]]
    for kx in ("vllm", "torch", "numpy", "scipy", "mpi4py", "httpx"):
        if kx in pk:
            rows.append([kx, str(pk[kx])])
    out.append(TABLE(rows, [0.3, 0.7], "Hardware and package snapshot recorded with the campaign (<font name='Mono'>hardware.json</font>).", font=8.2))

    out += UH("Appendix C. Glossary")
    G = [
        ("AWQ", "Activation-aware Weight Quantization; a 4-bit weight-only quantization method."),
        ("Batch cap", "Maximum number of sequences a server decodes together (<font name='Mono'>max_num_seqs</font>)."),
        ("Byte-ratio ceiling", "Ratio of bytes moved by the fp16 path to those moved by the int4 path for one matrix-vector product; the upper bound on speedup for a bandwidth-bound kernel."),
        ("Chunked prefill", "Splitting a long prompt into pieces processed over several scheduler iterations."),
        ("Cohen&rsquo;s d<sub>z</sub>", "Paired effect size: mean of the paired differences divided by their standard deviation."),
        ("Decode", "The token-by-token generation phase; memory-bound at small batch."),
        ("Goodput", "Rate of requests that meet the latency service-level objective."),
        ("Holm correction", "Step-down adjustment of <i>p</i>-values controlling the family-wise error rate."),
        ("KV cache", "Stored attention keys and values of previous tokens."),
        ("Open loop", "Load generation in which arrivals do not depend on completions."),
        ("Prefill", "The prompt-processing phase; compute-bound for long prompts."),
        ("Roofline", "Performance model bounding attainable throughput by compute and memory bandwidth roofs."),
        ("SLO", "Service-level objective; here TTFT &le; 0.5 s and TPOT &le; 0.05 s."),
        ("TPOT", "Time per output token after the first."),
        ("TTFT", "Time to first token, from scheduled arrival."),
        ("Tokens per joule", "Output tokens divided by GPU energy over the busy span."),
    ]
    out.append(TABLE([["term", "meaning"], *[[a, b] for a, b in G]], [0.22, 0.78], None, font=8.2))

    out += UH("Appendix D. Provenance of results")
    out.append(P(
        "Every table in Chapters 6 to 9 that contains a measured value is generated by the modules in "
        "<font name='Mono'>scripts/report_pdf</font> from the following files, all under <font name='Mono'>docs/results</font>:"))
    out.append(TABLE([
        ["Result", "Source file"],
        ["Kernel times, correctness, spread", "kernel_bench.json (median of 5 invocations)"],
        ["In-situ kernel times (both engine modes)", "insitu/awq_marlin_{graphs,eager}.json, insitu/awq_{graphs,eager}.json, insitu/awq_inferno_{graphs,eager}.json"],
        ["Small-batch kernel sweep", "kernel_batch_runs/run_*.json"],
        ["Rotation-size check", "flush_sweep/*.json"],
        ["End-to-end decode in vLLM", "e2e_kernel/*_rep*.json, e2e_kernel/summary.json"],
        ["Headroom analysis", "campaign/headroom.json"],
        ["Capacity and derived rates", "calibration.json, calibration_awq_kernel.json"],
        ["Campaign", "campaign/summary.csv, campaign/summary.json, campaign/config_snapshot.json"],
        ["Contention", "contention/contention_summary.json"],
        ["MPI microbenchmarks", "mpi_microbench.json"],
        ["Scaling and router comparison", "scaling/scaling_summary.json"],
        ["Sweep", "sweep/summary.csv"],
        ["Hardware and packages", "campaign/hardware.json"],
    ], [0.36, 0.64], "Result files behind the tables.", font=8.2))
    return out


def references():
    out = _Out(UH("References"))
    out.append(P(
        f"References are numbered in order of first citation. Entries marked &dagger; ({len(VERIFIED)} of them) were checked against the publisher, "
        f"proceedings or arXiv page while preparing this report. The remaining entries are cited from the author&rsquo;s knowledge; venue, year and page "
        f"details should be confirmed before formal submission. Software documentation is cited by name and version."))
    out.append(Spacer(1, 4))
    out += reference_flowables()
    return out
