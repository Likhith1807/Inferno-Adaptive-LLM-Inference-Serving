# Roadmap (phases, no timeline)

Status legend: **built** = code written and unit-tested; **run** = executed on real hardware with results
committed. Nothing below is marked "run" unless a results directory backs it.

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Reproducible environment: uv, date-pinned CUDA 12.x stack, no-root toolchain, Docker, CI, `hardware.json`, doctor script | **run**: environment verified (`scripts/check_env.py`), smoke test passed on the RTX 4060 |
| 1 | Harness: vLLM server manager, open-loop streaming client, TTFT/TPOT/p99/goodput, GPU power, paired seeded workloads, capacity calibration | **run** |
| 2 | Batching x quantization sweep with CIs | **run**: `docs/results/sweep` (subset grid, steady load) |
| 3 | Fused int4 GEMV kernel ladder v0-v4, cuBLAS/unfused baselines, CPU-reference correctness, in-situ vLLM kernel comparison | **run**: `docs/KERNEL_REPORT.md`. Nsight counters pending (needs GPU counter access, `docs/PROFILING.md`) |
| 4 | MPI dispatcher/worker, HTTP baseline, microbenchmarks, scaling experiment | **run**: microbench + real-replica scaling (`docs/results/scaling`) |
| 5 | Adaptive controller over heterogeneous pre-loaded replicas | **run**: H1 rejected; mechanism isolated by `run_contention.py` |
| 6 | Cloud: Terraform + orchestration with self-terminate cost cap, 7B configs | **written, not applied** (needs your AWS account) |
| 7 | Full campaign, paired statistics, generated report | **run**: 96 runs, 8 paired repetitions per cell (`docs/results/campaign`) |
| 8 | Packaging: generated README blocks, findings, replay dashboard, posting draft | **built**: `docs/FINDINGS.md`, `docs/POST_DRAFT.md`, `docs/demo/index.html` |

## Milestones after the first release (2026-11)

| # | Milestone | Status |
|---|---|---|
| M1 | Nsight Compute: why v4 helped Llama mlp_up but slowed Qwen gate_up | **blocked**: GPU counters (ERR_NVGPUCTRPERM) and Windows commit limit; `scripts/windows_dev_setup.ps1` fixes both (admin + reboot) |
| M2 | Every Qwen layer x batch 1-64 vs Marlin / AWQ, with conversion costs | **run**: `docs/results/layer_matrix/report.md`. Custom wins only qkv/o_proj at batch 1 (15% of decode GEMM time) |
| -- | Decision gate after M2 | **taken**: Marlin matches the kernel where it matters, so M3/M4 (kernel in the serving path, paired end-to-end test) are not pursued; the opt-in switch `vllm_int4.install()` remains as a research tool |
| -- | Marlin correctness probe (found by M2's reference check) | **kernel-level run** (`docs/results/marlin_fault/report.md`); real-checkpoint check and a second machine pending |
| M5 | One-command figure + short GPU benchmark + outside reproduction | **built**: `make figure`, `make quick-bench`, `docs/REPRODUCE.md`; independent reproduction pending |

## What remains (needs the owner)

1. Enable GPU performance counters (NVIDIA Control Panel) and run `make kernel-profile`; then resolve the open v4 question in `docs/KERNEL_REPORT.md`.
2. Run the cloud phase on a multi-GPU instance (`scripts/launch_cloud.sh`): with one replica per GPU the co-residency penalty disappears, so H1 becomes testable in its intended setting.
3. Publish: push the repo, post using `docs/POST_DRAFT.md`, host `docs/demo/`, and pursue one upstream contribution (`docs/UPSTREAM.md`).

## Non-negotiables

1. Every configuration is repeated; single runs are never reported.
2. The kernel result is reported honestly even if it loses to the library path.
3. The MPI overhead crossover is measured, not assumed.
4. Where the adaptive policy does not help, the report says so.
5. No number enters the README, the report, or a post unless a script produced it from raw data.
