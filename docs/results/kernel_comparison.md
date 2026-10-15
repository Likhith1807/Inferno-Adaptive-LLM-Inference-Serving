Device: NVIDIA GeForce RTX 4060 Laptop GPU (theoretical DRAM peak 256 GB/s, measured streaming read 252 GB/s), 5 standalone benchmark run(s). In-situ times: torch profiler on vLLM decode (graphs mode, Qwen2.5-1.5B AWQ, 28 layers), median over decode steps, then median across 3 fresh processes.

| layer | shape | cuBLAS fp16 (standalone) us | custom standalone us | best variant | custom in situ us | vLLM Marlin in situ us | vLLM AWQ in situ us | custom / vLLM Marlin | custom / vLLM AWQ | custom standalone vs cuBLAS | custom %peak BW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qkv | 2048x1536 | 28.88 | 13.88 | v6_lane16 | 10.24 | 10.27 | 12.22 | 1.00x | 0.84x | 2.08x faster | 47.2% |
| o_proj | 1536x1536 | 24.08 | 11.16 | v6_lane16 | 8.03 | 8.35 | 11.36 | 0.96x | 0.71x | 2.16x faster | 44.1% |
| gate_up | 17920x1536 | 221.70 | 61.46 | v6_lane16 | 69.86 | 60.09 | 72.26 | 1.16x | 0.97x | 3.61x faster | 93.2% |
| down | 1536x8960 | 113.40 | 33.03 | v6_lane16 | 37.06 | 31.77 | 41.92 | 1.17x | 0.88x | 3.43x faster | 86.7% |

Custom kernel (in situ) versus vLLM Marlin: faster on 1, on par on 1, slower on 2 of 4 layer shapes (on par = within 2%).
Custom kernel (in situ) versus vLLM AWQ: faster on 4, on par on 0, slower on 0 of 4 layer shapes (on par = within 2%).
