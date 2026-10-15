# Profiling the kernels with Nsight Compute (WSL2)

The benchmark reports *what* is fast; Nsight Compute reports *why*. On Windows/WSL2 the GPU performance counters
are locked by default, and `ncu` fails with:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

## One-time fix (Windows, needs admin)

1. Right-click the desktop, open **NVIDIA Control Panel**.
2. Menu **Desktop** then tick **Enable Developer Settings**.
3. In the left tree open **Developer**, then **Manage GPU Performance Counters**.
4. Select **Allow access to the GPU performance counters to all users**, then **Apply**.
5. In PowerShell run `wsl --shutdown`, then reopen your WSL shell.

(This is a Windows driver setting. Only enable it on a machine you control.)

## Run

```bash
source scripts/toolchain_env.sh
make -C src/inferno/kernels bench_int4_gemv
make kernel-profile          # full-set .ncu-rep per kernel + summary table at results/ncu/summary.md
```

Open the `.ncu-rep` files in the Nsight Compute GUI (on Windows, install Nsight Compute and open the file from
`\\wsl$\Ubuntu\...`) for the full source-level view. The summary table gives, per kernel and shape:
achieved DRAM throughput, shared-memory bank-conflict ratio, achieved occupancy, executed instructions, registers.

## What the counters should settle

| question | counter |
|---|---|
| Does v2 really have the predicted 16-way bank conflict (half-precision activations, lane stride 16 words), and does v3 remove it? | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` / wavefronts |
| Is v4 on K=1536 and K=8960 occupancy-limited (float smem footprint)? | `sm__warps_active`, `launch__occupancy_limit_shared_mem` |
| Is v4 losing to lane imbalance on non-multiple row lengths? | warp stall reasons, `smsp__thread_inst_executed_per_inst_executed` |
| Is any kernel actually DRAM-bound? | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |

Record the counter values in `docs/KERNEL_REPORT.md` (regenerate with `python scripts/build_kernel_report.py`
after adding the summary) and revise the "unresolved" section with what they show.
