/*
 * Fused int4 dequantize + GEMV kernels (decode-phase matvec: y = W_q * x).
 *
 * Why this kernel: at batch size 1, LLM decode is memory-bandwidth bound. Every
 * weight is read once and used for one multiply-add, so runtime ~= bytes moved
 * / DRAM bandwidth. int4 weights move 4x fewer bytes than fp16, but only if the
 * dequantization is fused into the matvec (no fp16 weight matrix is ever
 * materialized in DRAM). The figure of merit is therefore *effective bandwidth*
 * (% of the GPU's peak), not FLOPs.
 *
 * Weight layout (AWQ/GPTQ-like, asymmetric, group-wise):
 *   W      : uint32 [N][K/8]   nibble j (bits 4j..4j+3) of word w holds k = 8*w + j
 *   scales : half   [N][K/G]
 *   zeros  : half   [N][K/G]   (float-valued zero point)
 *   x      : half   [K]
 *   y      : half   [N]
 *   y[n] = sum_k (q[n][k] - zeros[n][k/G]) * scales[n][k/G] * x[k]
 *
 * Optimization ladder (each stage is benchmarked and profiled separately):
 *   v0  naive:       one thread per output row. Adjacent threads read addresses
 *                    K/2 bytes apart, so loads are uncoalesced.
 *   v1  warp-row:    one warp per row; lanes read consecutive 32-bit words
 *                    (coalesced), shuffle reduction.
 *   v2  vec+smem:    128-bit loads (32 weights per load per lane), activation
 *                    vector staged once per block in shared memory. Each lane
 *                    reads 32 consecutive half activations (64 B), so lanes are
 *                    16 words apart and fall on only 2 of 32 banks: a predicted
 *                    16-way shared-memory bank conflict (by address arithmetic;
 *                    check with l1tex__data_bank_conflicts_pipe_lsu_mem_shared).
 *   v3  padded smem: v2 plus one pad word per 32 activations (stride 33/17),
 *                    which makes the lane access pattern conflict-free.
 *   v4  algebraic:   sum_k (q-z)*s*x = s * (sum q*x  -  z * sum x) over any span
 *                    that shares one (s, z). A lane's 32-weight chunk lies inside
 *                    one group, so each lane accumulates sum(q*x) and sum(x) over its
 *                    own chunk in the same pass (no per-block prologue), then applies
 *                    s and z once per chunk: 1 convert + 2 FADD/FMA per weight.
 *                    Plus streaming (evict-first) loads and 2 loads in flight.
 *
 *   v5  register reuse (int4_gemv_batch.cuh, B = 1): the batch kernel instantiated at one activation vector. The lane
 *                    decodes its 32 weights to registers once, activations are read through the read-only cache
 *                    (no shared-memory staging, no block prologue). Measured faster than v3/v4 on every large shape.
 *   v6  lane-balanced (int4_gemv_batch.cuh, gemv_int4_lane<1,16,1>): v5 with 16 lanes per row instead of 32.
 *                    K = 1536 has 48 chunks per row, so with 32 lanes half the lanes do two chunks and half do
 *                    one; with 16 every lane does exactly three. Best or tied on every shape.
 *
 * Constraints: K % 32 == 0, G % 32 == 0. Rows are 16-byte aligned when K % 32 == 0.
 */
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace inferno {

constexpr int kWarpSize = 32;
constexpr int kBlockThreads = 256;                      // 8 warps -> 8 rows per block
constexpr int kRowsPerBlock = kBlockThreads / kWarpSize;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------- v0: naive
__global__ void gemv_int4_v0_naive(const uint32_t* __restrict__ W, const half* __restrict__ S,
                                   const half* __restrict__ Z, const half* __restrict__ x,
                                   half* __restrict__ y, int N, int K, int G) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const uint32_t* row = W + (size_t)n * (K / 8);
    const half* s = S + (size_t)n * (K / G);
    const half* z = Z + (size_t)n * (K / G);
    float acc = 0.f;
    for (int w = 0; w < K / 8; ++w) {
        uint32_t packed = row[w];
        for (int j = 0; j < 8; ++j) {
            int k = 8 * w + j;
            float q = (float)((packed >> (4 * j)) & 0xF);
            acc += (q - __half2float(z[k / G])) * __half2float(s[k / G]) * __half2float(x[k]);
        }
    }
    y[n] = __float2half(acc);
}

// ---------------------------------------------------------------- v1: warp per row
__global__ void gemv_int4_v1_warp(const uint32_t* __restrict__ W, const half* __restrict__ S,
                                  const half* __restrict__ Z, const half* __restrict__ x,
                                  half* __restrict__ y, int N, int K, int G) {
    const int warp = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;
    const int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;
    const uint32_t* row = W + (size_t)n * (K / 8);
    const half* s = S + (size_t)n * (K / G);
    const half* z = Z + (size_t)n * (K / G);
    float acc = 0.f;
    for (int w = lane; w < K / 8; w += kWarpSize) {
        uint32_t packed = row[w];
        const int k0 = 8 * w, g = k0 / G;
        const float sc = __half2float(s[g]), zp = __half2float(z[g]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float q = (float)((packed >> (4 * j)) & 0xF);
            acc += (q - zp) * sc * __half2float(x[k0 + j]);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[n] = __float2half(acc);
}

// ---------------------------------------------------------------- v2/v3: 128-bit loads + smem x
// Dynamic smem: K halves (+ 2 pad halves per 32 when PAD). PAD=false is v2 (bank-conflicted), PAD=true is v3.
template <bool PAD>
__device__ __forceinline__ int smem_half_index(int k) { return PAD ? k + 2 * (k >> 5) : k; }

template <bool PAD>
__global__ void gemv_int4_vec(const uint32_t* __restrict__ W, const half* __restrict__ S,
                              const half* __restrict__ Z, const half* __restrict__ x,
                              half* __restrict__ y, int N, int K, int G) {
    extern __shared__ half xs_h[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs_h[smem_half_index<PAD>(i)] = x[i];
    __syncthreads();

    const int warp = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;
    const int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;
    const uint4* row = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 8));
    const half* s = S + (size_t)n * (K / G);
    const half* z = Z + (size_t)n * (K / G);
    float acc = 0.f;
    for (int i = lane; i < K / 32; i += kWarpSize) {
        const uint4 v = row[i];
        const int k0 = 32 * i, g = k0 / G;
        const int xb = smem_half_index<PAD>(k0);       // k0 is a multiple of 32, so +c stays inside the chunk
        const float sc = __half2float(s[g]), zp = __half2float(z[g]);
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float q = (float)((words[wi] >> (4 * j)) & 0xF);
                acc += (q - zp) * sc * __half2float(xs_h[xb + 8 * wi + j]);
            }
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[n] = __float2half(acc);
}

// ---------------------------------------------------------------- v4: algebraic + streaming loads
// Dynamic smem: (K + K/32) floats of x with one pad float per 32 (conflict-free). No prologue beyond staging x:
// an earlier variant precomputed per-chunk sums of x in shared memory; it was slower because with short rows
// (K = 1536: 1-2 loop iterations per lane) any per-block prologue phase dominates the block's lifetime.
__device__ __forceinline__ int smem_f_index(int k) { return k + (k >> 5); }
// Streaming (evict-first) loads; -DINFERNO_NO_STREAM_LOADS swaps in ordinary loads for A/B measurement.
#ifdef INFERNO_NO_STREAM_LOADS
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) { return __ldg(p); }
#else
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) { return __ldcs(p); }
#endif

// Returns (sum_j q_j * x_j, sum_j x_j) over one 32-weight chunk; xs points at the chunk start.
__device__ __forceinline__ float2 dot_and_sum32(const uint4& v, const float* __restrict__ xs) {
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
    float a0 = 0.f, a1 = 0.f, s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float x0 = xs[8 * wi + j], x1 = xs[8 * wi + j + 1];
            a0 = fmaf((float)((words[wi] >> (4 * j)) & 0xF), x0, a0);
            a1 = fmaf((float)((words[wi] >> (4 * (j + 1))) & 0xF), x1, a1);
            s0 += x0;
            s1 += x1;
        }
    }
    return make_float2(a0 + a1, s0 + s1);
}

__global__ void gemv_int4_v4_alg(const uint32_t* __restrict__ W, const half* __restrict__ S,
                                 const half* __restrict__ Z, const half* __restrict__ x,
                                 half* __restrict__ y, int N, int K, int G) {
    extern __shared__ float xs[];             // K + K/32 (padded)
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[smem_f_index(i)] = __half2float(x[i]);
    __syncthreads();

    const int warp = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;
    const int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;
    const uint4* row = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 8));
    const half* s = S + (size_t)n * (K / G);
    const half* z = Z + (size_t)n * (K / G);
    const int iters = K / 32;
    float acc = 0.f;

    int i = lane;
    // Two independent 128-bit loads in flight per lane to hide DRAM latency.
    for (; i + kWarpSize < iters; i += 2 * kWarpSize) {
        // Issue every global load (weights, scales, zeros) before any compute so their latencies overlap.
        const int g0 = (32 * i) / G, g1 = (32 * (i + kWarpSize)) / G;
        const uint4 v0 = ldg_stream(row + i);
        const uint4 v1 = ldg_stream(row + i + kWarpSize);
        const float s0 = __half2float(s[g0]), z0 = __half2float(z[g0]);
        const float s1 = __half2float(s[g1]), z1 = __half2float(z[g1]);
        const float2 d0 = dot_and_sum32(v0, xs + smem_f_index(32 * i));
        const float2 d1 = dot_and_sum32(v1, xs + smem_f_index(32 * (i + kWarpSize)));
        acc += s0 * (d0.x - z0 * d0.y);
        acc += s1 * (d1.x - z1 * d1.y);
    }
    for (; i < iters; i += kWarpSize) {
        const int g0 = (32 * i) / G;
        const uint4 v0 = ldg_stream(row + i);
        const float s0 = __half2float(s[g0]), z0 = __half2float(z[g0]);
        const float2 d0 = dot_and_sum32(v0, xs + smem_f_index(32 * i));
        acc += s0 * (d0.x - z0 * d0.y);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[n] = __float2half(acc);
}

// ---------------------------------------------------------------- host launchers
inline int grid_rows(int N) { return (N + kRowsPerBlock - 1) / kRowsPerBlock; }

inline void launch_v0(const uint32_t* W, const half* S, const half* Z, const half* x, half* y,
                      int N, int K, int G, cudaStream_t st = 0) {
    gemv_int4_v0_naive<<<(N + kBlockThreads - 1) / kBlockThreads, kBlockThreads, 0, st>>>(W, S, Z, x, y, N, K, G);
}
inline void launch_v1(const uint32_t* W, const half* S, const half* Z, const half* x, half* y,
                      int N, int K, int G, cudaStream_t st = 0) {
    gemv_int4_v1_warp<<<grid_rows(N), kBlockThreads, 0, st>>>(W, S, Z, x, y, N, K, G);
}
inline size_t v2_smem_bytes(int K, bool pad) { return (size_t)(K + (pad ? 2 * (K / 32) : 0)) * sizeof(half); }
inline void launch_v2(const uint32_t* W, const half* S, const half* Z, const half* x, half* y,
                      int N, int K, int G, cudaStream_t st = 0) {
    gemv_int4_vec<false><<<grid_rows(N), kBlockThreads, v2_smem_bytes(K, false), st>>>(W, S, Z, x, y, N, K, G);
}
inline void launch_v3(const uint32_t* W, const half* S, const half* Z, const half* x, half* y,
                      int N, int K, int G, cudaStream_t st = 0) {
    gemv_int4_vec<true><<<grid_rows(N), kBlockThreads, v2_smem_bytes(K, true), st>>>(W, S, Z, x, y, N, K, G);
}
inline size_t v4_smem_bytes(int K, int G) { (void)G; return (size_t)(K + K / 32) * sizeof(float); }
inline void launch_v4(const uint32_t* W, const half* S, const half* Z, const half* x, half* y,
                      int N, int K, int G, cudaStream_t st = 0) {
    const size_t smem = v4_smem_bytes(K, G);
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(gemv_int4_v4_alg, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemv_int4_v4_alg<<<grid_rows(N), kBlockThreads, smem, st>>>(W, S, Z, x, y, N, K, G);
}

}  // namespace inferno
