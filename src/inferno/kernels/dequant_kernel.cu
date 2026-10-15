/*
 * Custom int4 -> fp16 dequantization kernel.
 *
 * This is the CO2 CUDA deliverable (Week 5). Naive-correct version is
 * below; the Week 5 task is to profile it against the library default
 * (bitsandbytes/AWQ's kernel) with Nsight Compute and then optimize:
 *   1. shared-memory tiling for the scale/zero-point lookup
 *   2. memory coalescing on the packed int4 reads
 *   3. vectorized fp16 writes (half2)
 *
 * IMPORTANT (see docs/methodology.md "honesty checkpoints"):
 * report the measured ratio against the library kernel honestly, even
 * if this loses. That comparison, explained, is worth more than a
 * false "beats the library" claim.
 */

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Naive baseline: one thread per output element, no shared memory, no
// vectorization. Correct, not fast — the optimization target for Week 5.
__global__ void dequantize_int4_to_fp16_naive(
    const uint8_t* __restrict__ packed_weights,  // 2 int4 values per byte
    const half* __restrict__ scales,             // per-group scale
    const half* __restrict__ zero_points,        // per-group zero point
    half* __restrict__ output,
    int num_elements,
    int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int byte_idx = idx / 2;
    bool high_nibble = (idx % 2) == 1;
    uint8_t packed = packed_weights[byte_idx];
    int q = high_nibble ? (packed >> 4) : (packed & 0x0F);

    int group_idx = idx / group_size;
    half scale = scales[group_idx];
    half zero = zero_points[group_idx];

    // dequantized = (q - zero) * scale
    output[idx] = __hmul(__hsub(__int2half_rn(q), zero), scale);
}

// TODO (Week 5): tiled/coalesced version.
// __global__ void dequantize_int4_to_fp16_tiled(...) { ... }

extern "C" void launch_dequantize_int4_to_fp16_naive(
    const uint8_t* packed_weights,
    const half* scales,
    const half* zero_points,
    half* output,
    int num_elements,
    int group_size,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    dequantize_int4_to_fp16_naive<<<blocks, threads, 0, stream>>>(
        packed_weights, scales, zero_points, output, num_elements, group_size
    );
}
