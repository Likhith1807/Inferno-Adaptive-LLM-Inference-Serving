# Qwen2.5-1.5B int4 layers: which implementation wins? (generated)

GPU NVIDIA GeForce RTX 4060 Laptop GPU, vLLM 0.11.0, torch 2.8.0+cu128, group 128. CUDA graph of one call per weight copy, rotated over >= rotate_mb; us per call; rotation >= 768 MB, 7 trials x 20 graph replays per cell. Times are median microseconds per call. A win requires non-overlapping trial ranges, otherwise `tie`. `n/a`: the custom kernel is not instantiated for that batch (the deployed path falls back).

## Winner per cell (int4 implementations only)

| shape (N x K) | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|---|
| qkv (2048x1536) | **custom** 1.08x | tie (marlin/custom) | tie (marlin/awq) | tie (marlin/awq) | **marlin** 1.45x | **marlin** 1.45x | **marlin** 1.35x |
| o_proj (1536x1536) | **custom** 1.14x | tie (custom/marlin) | tie (marlin/awq) | **marlin** 1.38x | **marlin** 1.34x | tie (marlin/awq) | tie (marlin/awq) |
| gate_up (17920x1536) | tie (awq/marlin) | tie (marlin/awq) | **marlin** 1.06x | **marlin** 1.12x | **marlin** 1.14x | **marlin** 1.61x | **marlin** 1.65x |
| down (1536x8960) | tie (awq/marlin) | tie (awq/marlin) | tie (awq/marlin) | tie (awq/marlin) | tie (marlin/awq) | tie (marlin/awq) | **marlin** 1.21x |

Ratio = runner-up time / winner time.

## Median us per call

| shape | impl | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|---|---|
| qkv | custom | 12.7 | 14.2 | 25.5 | 40.7 | 79.2 | n/a | n/a |
| qkv | marlin | 13.8 | 13.8 | 13.7 | 13.9 | 14.7 | 22.7 | 38.7 |
| qkv | awq | 15.3 | 15.4 | 15.5 | 17.1 | 21.3 | 32.8 | 52.0 |
| qkv | fp16 | 41.1 | 42.0 | 43.2 | 41.1 | 43.1 | 43.4 | 41.3 |
| o_proj | custom | 9.8 | 10.4 | 20.8 | 34.2 | 65.8 | n/a | n/a |
| o_proj | marlin | 11.2 | 11.7 | 12.2 | 11.1 | 13.2 | 21.6 | 36.4 |
| o_proj | awq | 13.2 | 14.8 | 13.2 | 15.4 | 17.7 | 24.8 | 40.9 |
| o_proj | fp16 | 25.0 | 25.5 | 25.0 | 25.9 | 24.5 | 30.1 | 33.5 |
| gate_up | custom | 97.9 | 102.4 | 196.0 | 390.3 | 735.0 | n/a | n/a |
| gate_up | marlin | 96.7 | 97.2 | 97.2 | 97.1 | 99.1 | 113.1 | 216.6 |
| gate_up | awq | 92.8 | 101.8 | 103.0 | 108.9 | 113.3 | 181.9 | 358.4 |
| gate_up | fp16 | 355.9 | 368.9 | 369.3 | 369.7 | 372.4 | 372.5 | 361.7 |
| down | custom | 50.2 | 52.6 | 113.3 | 185.6 | 357.6 | n/a | n/a |
| down | marlin | 46.5 | 50.4 | 50.4 | 49.0 | 49.9 | 61.8 | 114.9 |
| down | awq | 46.1 | 49.0 | 48.5 | 43.9 | 49.9 | 68.7 | 138.8 |
| down | fp16 | 181.5 | 177.9 | 159.8 | 175.0 | 180.7 | 182.9 | 183.4 |

## Custom kernel vs Marlin (vLLM's default int4 path), time ratio custom/marlin

| shape | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|---|
| qkv | 0.92 | 1.03 | 1.85 | 2.93 | 5.39 | n/a | n/a |
| o_proj | 0.88 | 0.89 | 1.71 | 3.08 | 5.00 | n/a | n/a |
| gate_up | 1.01 | 1.05 | 2.02 | 4.02 | 7.42 | n/a | n/a |
| down | 1.08 | 1.04 | 2.25 | 3.79 | 7.17 | n/a | n/a |

< 1 means the custom kernel is faster.

## Share of quantized-GEMM time per decode step (using Marlin times)

| batch | qkv | o_proj | gate_up | down |
|---|---|---|---|---|
| 1 | 8% | 7% | 58% | 28% |
| 2 | 8% | 7% | 56% | 29% |
| 4 | 8% | 7% | 56% | 29% |
| 8 | 8% | 6% | 57% | 29% |
| 16 | 8% | 7% | 56% | 28% |
| 32 | 10% | 10% | 52% | 28% |
| 64 | 10% | 9% | 53% | 28% |

## One-time layout conversion (ms, paid once at model load) and correctness

| shape | custom convert ms | marlin convert ms | awq convert ms | custom max rel err | marlin max rel err | awq max rel err |
|---|---|---|---|---|---|---|
| qkv | 2.07 | 2.63 | 0.02 | 4.4e-04 | 3.4e-04 | 4.3e-04 |
| o_proj | 1.08 | 1.66 | 0.02 | 3.3e-04 | 3.3e-04 | 3.8e-04 |
| gate_up | 10.04 | 4.26 | 0.02 | 4.2e-04 | 3.9e-04 | 4.4e-04 |
| down | 4.82 | 2.39 | 0.02 | 3.7e-04 | 1.2e-01 | 5.3e-04 |

`awq` needs no conversion (it runs on the checkpoint layout). Errors are relative max errors vs awq_dequantize + fp32 matmul.

## Verdict

The custom kernel clearly wins 2 of the 20 cells where it runs. Cell winners: custom: 2, marlin: 11, awq: 0, tie: 15.

