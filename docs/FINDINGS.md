# Findings

All measured on **NVIDIA GeForce RTX 4060 Laptop GPU** with vLLM 0.11 and Qwen2.5-1.5B-Instruct unless stated. Each number is read from a data file under `docs/results/` by `scripts/build_findings.py`; methods and caveats are in `docs/methodology.md`, `docs/KERNEL_REPORT.md` and the per-experiment reports.

## 1. A fused int4 GEMV kernel reaches 87-95% of DRAM bandwidth on large layers (parity with vLLM's Marlin end to end at batch 1, not a win)

- Best case **llama7B_mlp_up**: 95% of peak DRAM bandwidth, **3.67x faster than fp16 cuBLAS**, against a byte-ratio ceiling of 3.76x (int4 also reads fp16 scales and zero points, so it moves 0.53 bytes per weight, not 0.5). Median of 5 benchmark invocations; every kernel matched a CPU reference.
- Large layers: qwen1.5B_gate_up 3.61x (93% of peak); qwen1.5B_down 3.43x (87% of peak); llama7B_attn_proj 3.51x (90% of peak); llama7B_mlp_up 3.67x (95% of peak).
- Small layers (about 1-2 MB of weights) are dominated by launch and clock jitter (only 2-4 of 5 runs agree within 2%), so no ranking is claimed there.
- Measured streaming-read bandwidth is 252 GB/s (98% of the 256 GB/s computed from the memory clock); the best kernel reaches 97% of that achievable figure. The benchmark rotates 768 MB of weights: an earlier 192 MB rotation let a non-LRU L2 serve about 6% of the traffic and inflated every '% of peak' figure (`docs/KERNEL_REPORT.md`).
- The optimization ladder is not monotone: wider loads with shared-memory staging made the kernel *slower* until the predicted 16-way bank conflict (from address arithmetic; counters pending) was padded away; the algebraic rewrite (v4) helped on K=4096 but hurt on K=1536/8960; dropping shared-memory staging (v5) and balancing lanes for 48-chunk rows (v6) fixed that. Causes are hypotheses until Nsight counters can be read.
- Small batches (`qwen1.5B_gate_up`, speedup over fp16 cuBLAS with the same B): B=1 3.60x, B=2 3.37x, B=4 1.80x, B=8 1.22x. The CUDA-core kernel loses to tensor-core cuBLAS above B of about 8; it is a batch-1 / speculative-decoding-sized result.
### Against vLLM's own kernels (decode, in situ, CUDA graphs)

Device: NVIDIA GeForce RTX 4060 Laptop GPU (theoretical DRAM peak 256 GB/s, measured streaming read 252 GB/s), 5 standalone benchmark run(s). In-situ times: torch profiler on vLLM decode (graphs mode, Qwen2.5-1.5B AWQ, 28 layers), median over decode steps, then median across 3 fresh processes.

| layer | shape | cuBLAS fp16 (standalone) us | custom standalone us | best variant | custom in situ us | vLLM Marlin in situ us | vLLM AWQ in situ us | custom / vLLM Marlin | custom / vLLM AWQ | custom standalone vs cuBLAS | custom %peak BW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qkv | 2048x1536 | 28.88 | 13.88 | v6_lane16 | 10.24 | 10.27 | 12.22 | 1.00x | 0.84x | 2.08x faster | 47.2% |
| o_proj | 1536x1536 | 24.08 | 11.16 | v6_lane16 | 8.03 | 8.35 | 11.36 | 0.96x | 0.71x | 2.16x faster | 44.1% |
| gate_up | 17920x1536 | 221.70 | 61.46 | v6_lane16 | 69.86 | 60.09 | 72.26 | 1.16x | 0.97x | 3.61x faster | 93.2% |
| down | 1536x8960 | 113.40 | 33.03 | v6_lane16 | 37.06 | 31.77 | 41.92 | 1.17x | 0.88x | 3.43x faster | 86.7% |

Custom kernel (in situ) versus vLLM Marlin: faster on 1, on par on 1, slower on 2 of 4 layer shapes (on par = within 2%).
Custom kernel (in situ) versus vLLM AWQ: faster on 4, on par on 0, slower on 0 of 4 layer shapes (on par = within 2%).


**vLLM's Marlin kernel is at the bandwidth limit on the large layers at batch 1; the custom kernel does not beat it.** (An earlier version of this project reported the opposite from a Marlin timing that turned out to be about 2x too slow and irreproducible; it was found by cross-checking against end-to-end throughput, re-measured in both engine modes and retired. See `docs/KERNEL_REPORT.md`.)

### End to end in vLLM (decode throughput, tokens/s, CUDA graphs, mean of process repetitions)

| configuration | 1 seq | 2 seq | 4 seq | 8 seq |
|---|---|---|---|---|
| fp16 (no quantization) | 67 &plusmn; 0 (0.47x) | 132 &plusmn; 2 (0.47x) | 263 &plusmn; 3 (0.48x) | 515 &plusmn; 11 (0.49x) |
| AWQ-int4, vLLM classic kernel | 136 &plusmn; 4 (0.95x) | 264 &plusmn; 8 (0.94x) | 516 &plusmn; 8 (0.94x) | 968 &plusmn; 36 (0.91x) |
| AWQ-int4, vLLM Marlin (default) | 143 &plusmn; 2 | 280 &plusmn; 5 | 547 &plusmn; 7 | 1060 &plusmn; 6 |
| AWQ-int4, custom kernel (<= 2 tokens) | 143 &plusmn; 2 (1.00x) | 268 &plusmn; 11 (0.96x) | 512 &plusmn; 18 (0.94x) | 954 &plusmn; 68 (0.90x) |
| AWQ-int4, custom kernel (<= 8 tokens) | 144 &plusmn; 2 (1.01x) | 272 &plusmn; 7 (0.97x) | 370 &plusmn; 2 (0.68x) | 472 &plusmn; 6 (0.44x) |

Output tokens/s (mean of 3 process repetitions, +/- 95% CI); parentheses: paired ratio to AWQ-int4, vLLM Marlin (default).


At one sequence the custom kernel matches vLLM's default Marlin path and beats the classic AWQ kernel and fp16; with 4-8 concurrent sequences Marlin's tensor-core kernel is clearly faster. The custom kernel was verified on the real checkpoint (all patched layers within fp16 rounding of vLLM's own dequantize-and-matmul).

At server level with 64 concurrent sequences, vLLM's Marlin path sustains **2.09x** the throughput of the classic AWQ path (2828 vs 1354 tok/s): a batch-1 kernel is a low-concurrency, latency-sensitive result.

## 2. The adaptive router did not beat the best fixed configuration (H1 rejected)

Pre-registered comparison, 8 paired repetitions per load pattern, best fixed baseline chosen per pattern, Holm-corrected:

| load pattern | best fixed | adaptive p99 vs best fixed | Holm p | Cohen's dz | throughput ratio |
|---|---|---|---|---|---|
| bursty | fixed-bulk | +112% | 0.023 | 8.2 | 0.94x |
| mixed_length | fixed-fast | +204% | 0.023 | 12.0 | 0.96x |
| steady | fixed-bulk | +195% | 0.023 | 10.0 | 0.94x |

Adaptive was significantly *worse* on every pattern (all 8 pairs in the same direction, Holm p is the smallest attainable at n=8), with lower throughput and lower tokens/joule too. It tracked the static load balancer almost exactly.

Headroom check (post hoc, existing data): the two fixed tiers differ by at most 3.3% in p99 once the better one is chosen per pattern (fixed-bulk is the best single tier by worst-case regret), so even a perfect pattern-aware switch between them could not have won by more than that on this pool. The H1 test was close to unwinnable here, independent of the co-residency penalty of the policy actually tested.

### Why: two active replicas on one GPU contend

A controlled experiment holds total load fixed (30% and 60% of measured capacity) and varies only how the GPU is shared:

- Halving the memory budget with the second replica idle changes p99 by +0.8%, +0.9%: **no cost**.
- Making the second replica *active* multiplies p99 by 2.06x, 2.08x, with about 5% lower throughput and 8-9% lower tokens/joule: two processes time-slice one GPU and run smaller batches each.
- Design implication: replica-level adaptivity needs replicas on separate devices. The cloud configs (`configs/pool_config_cloud.yaml`, `infra/aws/`) put one replica per GPU; they have not been run yet.

## 3. MPI routing: microseconds per hop, unmeasurable end to end

- Ping-pong: one-way latency **1.9 us** for 64 B, 12.7 us for 32 KB, and 377 us for 256 KB (a protocol cliff between 32 KB and 256 KB).
- Dispatch loop: 186k jobs/s with 1 worker, 123k with 8, 50k with 15 (on 16 logical cores, so the 15-worker point is oversubscribed and shows p99 spikes).
- Real replicas (saturating load, MPI vs HTTP router, paired by repetition): 1 replica(s): MPI +0.58% throughput vs HTTP (p_ttest 0.05, p_wilcoxon 0.12); 2 replica(s): MPI +0.01% throughput vs HTTP (p_ttest 0.95, p_wilcoxon 1.00). No end-to-end difference is detectable.
- Two replicas on one GPU reach only **0.47** of ideal linear scaling (2106 to 1980 tok/s): throughput falls, the same co-residency penalty as above.
- Inside the real router the measured MPI dispatch hop is about 2.0 ms (p50) rather than microseconds, because the worker's single asyncio loop also parses hundreds of streaming responses; still under 0.1% of a request's latency.

## 4. Quantization and batch cap (Phase 2 characterization)

Steady load at 0.6x the measured capacity of the best configuration; 3 repetitions per cell; unfinished requests 60 s after the last arrival are counted as failures (see `docs/results/sweep/report.md`).

- At the same offered load, moving fp16 to int4 (batch cap 64) cuts median latency from 5.5 s to **2.6 s (2.1x)** and raises tokens per joule from 17.9 to **19.9 (+12%)**; int8 sits between (3.6 s, 18.8 tok/J). Completed throughput differs by only 7% because the load, not the hardware, limits it.
- A batch cap of 1 cannot sustain this load at any quantization level (fp16 and int8 complete 0% of requests in the window); caps of 32 and 64 are indistinguishable at this load because in-flight concurrency stays under 32.
- **Reconfiguration is expensive:** restarting the server to change batch size or quantization took a median of **48 s** (model already cached), several times a 10 s burst. This is why the controller routes across pre-loaded replicas instead of hot-swapping.

## 5. Where the custom kernel earns its place: almost nowhere (decision: keep vLLM's Marlin)

Every Qwen2.5-1.5B decode layer at batch 1-64, same AWQ weights for every implementation, weights streamed from DRAM, CUDA-graph timing (`docs/results/layer_matrix/report.md`, `scripts/run_layer_matrix.sh`).

- Custom/Marlin time at batch 1: qkv 0.92, o_proj 0.88, gate_up 1.01, down 1.08. The custom kernel clearly wins only on qkv, o_proj, which are 15% of quantized-GEMM time per decode step; the two large layers (gate_up 58%, down 28%) are ties.
- From batch 4 the custom kernel is 1.7-7.4x slower: its time grows roughly linearly with batch (CUDA-core arithmetic per weight scales with batch), while Marlin's tensor-core path stays flat to batch 16.
- One-time layout conversion is negligible for both (milliseconds per layer at load).
- **Decision (pre-stated rule):** the built-in path already matches the custom kernel on the layers that dominate decode, so the kernel is not carried into the serving path as a default; engineering effort goes to routing. The opt-in integration (`inferno.kernels.vllm_int4.install`) stays as a research tool.

## 6. vLLM's Marlin int4 GEMM returned wrong results at some batch sizes on this machine

The layer benchmark's correctness check flagged Marlin. In isolation (40 identical calls per cell, `docs/results/marlin_fault/report.md`), batches [32, 64, 128] give intermittent relative errors up to 0.28 on every Qwen shape tested, while batches [1, 8, 16, 48] are always correct (worst 4e-04). The custom kernel, classic AWQ and fp16 are correct on the same inputs; a fresh zeroed workspace does not remove it.
- **Status: observed on one machine (WSL2, laptop GPU); not yet reproduced on other hardware.** It may be specific to WSL2 or this GPU; it is not called a vLLM bug until it reproduces elsewhere (`scripts/probe_marlin_flaky.sh`).
- Real-checkpoint end-to-end check (`scripts/probe_marlin_e2e.sh`) not run yet.
- Marlin timings at those batch sizes in section 5 should be read with this caveat.

## Limitations

Single small GPU; replicas share it. Synthetic prompt text with server-verified token counts; checkpoint quality is not evaluated. n=4-8 repetitions per cell: enough for large effects, not for effects near 1%. See `docs/methodology.md`.
