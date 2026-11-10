# Marlin int4 GEMM: batch-size-dependent wrong results (generated)

GPU NVIDIA GeForce RTX 4060 Laptop GPU (24 SMs), vLLM 0.11.0 `gptq_marlin_gemm` with AWQ uint4 weights (group 128), called exactly as `apply_awq_marlin_linear` does. Each row: 40 calls with identical inputs, compared with `awq_dequantize` + fp32 matmul; a call is bad if its relative max error exceeds 1e-2. Variants: `plain` back-to-back with vLLM's shared workspace; `dirty` NaN-filled allocations between calls; `fresh_ws` a newly zeroed workspace per call; `sync` a device synchronize between calls.

**Status: observed on one machine (WSL2, laptop GPU); not yet reproduced on other hardware.** Not reported upstream until reproduced.

## Bad calls out of 40 (worst relative error)

### down (N=1536, K=8960)

| B | plain | dirty | fresh_ws | sync |
|---|---|---|---|---|
| 1 | 0/40 (3.1e-04) | 0/40 (3.1e-04) | 0/40 (3.1e-04) | 0/40 (3.1e-04) |
| 8 | 0/40 (2.5e-04) | 0/40 (2.5e-04) | 0/40 (2.5e-04) | 0/40 (2.5e-04) |
| 16 | 0/40 (3.2e-04) | 0/40 (3.2e-04) | 0/40 (3.2e-04) | 0/40 (3.2e-04) |
| 32 | 19/40 (1.1e-01) | 40/40 (1.4e-01) | 40/40 (1.4e-01) | 31/40 (1.3e-01) |
| 48 | 0/40 (2.3e-04) | 0/40 (2.3e-04) | 0/40 (2.3e-04) | 0/40 (2.3e-04) |
| 64 | 6/40 (1.2e-01) | 13/40 (1.2e-01) | 40/40 (1.6e-01) | 27/40 (1.4e-01) |
| 128 | 12/40 (1.0e-01) | 29/40 (1.6e-01) | 40/40 (1.5e-01) | 38/40 (1.5e-01) |

### gate_up (N=17920, K=1536)

| B | plain | dirty | fresh_ws | sync |
|---|---|---|---|---|
| 1 | 0/40 (2.7e-04) | 0/40 (2.7e-04) | 0/40 (2.7e-04) | 0/40 (2.7e-04) |
| 8 | 0/40 (2.7e-04) | 0/40 (2.7e-04) | 0/40 (2.7e-04) | 0/40 (2.7e-04) |
| 16 | 0/40 (2.5e-04) | 0/40 (2.5e-04) | 0/40 (2.5e-04) | 0/40 (2.5e-04) |
| 32 | 27/40 (2.0e-01) | 32/40 (2.3e-01) | 34/40 (2.3e-01) | 11/40 (1.8e-01) |
| 48 | 0/40 (3.9e-04) | 0/40 (3.9e-04) | 0/40 (3.9e-04) | 0/40 (3.9e-04) |
| 64 | 33/40 (2.8e-01) | 35/40 (2.8e-01) | 10/40 (2.4e-01) | 30/40 (2.4e-01) |
| 128 | 14/40 (1.7e-01) | 25/40 (2.3e-01) | 28/40 (2.0e-01) | 16/40 (1.6e-01) |

### qkv (N=2048, K=1536)

| B | plain | dirty | fresh_ws | sync |
|---|---|---|---|---|
| 1 | 0/40 (3.0e-04) | 0/40 (3.0e-04) | 0/40 (3.0e-04) | 0/40 (3.0e-04) |
| 8 | 0/40 (2.8e-04) | 0/40 (2.8e-04) | 0/40 (2.8e-04) | 0/40 (2.8e-04) |
| 16 | 0/40 (2.6e-04) | 0/40 (2.6e-04) | 0/40 (2.6e-04) | 0/40 (2.6e-04) |
| 32 | 21/40 (1.9e-01) | 0/40 (2.7e-04) | 17/40 (1.8e-01) | 31/40 (2.0e-01) |
| 48 | 0/40 (2.5e-04) | 0/40 (2.5e-04) | 0/40 (2.5e-04) | 0/40 (2.5e-04) |
| 64 | 12/40 (1.9e-01) | 3/40 (1.7e-01) | 20/40 (1.8e-01) | 21/40 (2.0e-01) |
| 128 | 20/40 (2.4e-01) | 31/40 (2.4e-01) | 16/40 (1.7e-01) | 25/40 (1.9e-01) |

Batch sizes with any bad call: [32, 64, 128]. Always correct: [1, 8, 16, 48] (worst error 3.9e-04). Worst faulty error 0.28. The custom kernel, classic AWQ and fp16 are deterministic and correct on the same inputs (`docs/results/layer_matrix/report.md`). A fresh zeroed workspace does not remove the fault, so stale lock state in the shared workspace is not the explanation.

## Real checkpoint

Not run yet (`scripts/probe_marlin_e2e.sh`).

## Reproduce

```bash
bash scripts/probe_marlin_flaky.sh > results/marlin_fault/kernel_probe.txt
bash scripts/probe_marlin_e2e.sh
python scripts/summarize_marlin_fault.py
```
