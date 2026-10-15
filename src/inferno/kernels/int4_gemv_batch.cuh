/*
 * Small-batch extension of the fused int4 kernel: Y[b][n] = sum_k W_q[n][k] * X[b][k], for B = 1..16 activation
 * vectors (speculative decoding, a few concurrent sequences, short prefill chunks).
 *
 * Idea: the weight chunk a lane loads (32 int4 weights, one 128-bit load) is dequantized into registers ONCE and
 * reused for all B activation vectors, so weight traffic stays constant while the arithmetic grows with B.
 * Same weight layout and the same algebra as v4:  sum_k (q-z)*s*x = s * (sum q*x - z * sum x)  per 32-chunk.
 *
 * This kernel runs on CUDA cores (FMA), not tensor cores. Per weight it costs about 3 instructions per batch
 * element, so it stops being memory-bound at small B: the benchmark (bench_int4_batch.cu) measures where it
 * stops paying off against fp16 cuBLAS. Activations are read through the read-only cache (they are tiny and shared
 * by every warp), so there is no shared-memory footprint and no limit on K.
 *
 * Layouts: X is half [B][K] row-major, Y is half [B][N] row-major.
 */
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "int4_gemv.cuh"

namespace inferno {

template <int B>
__global__ void gemv_int4_batch(const uint32_t* __restrict__ W, const half* __restrict__ S,
                                const half* __restrict__ Z, const half* __restrict__ X,
                                half* __restrict__ Y, int N, int K, int G) {
    const int warp = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;
    const int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;
    const uint4* row = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 8));
    const half* s = S + (size_t)n * (K / G);
    const half* z = Z + (size_t)n * (K / G);
    const int iters = K / 32;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;

    for (int i = lane; i < iters; i += kWarpSize) {
        const uint4 v = ldg_stream(row + i);
        const int g = (32 * i) / G;
        const float sc = __half2float(s[g]), zp = __half2float(z[g]);
        float q[32];
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int wi = 0; wi < 4; ++wi)
#pragma unroll
            for (int j = 0; j < 8; ++j) q[8 * wi + j] = (float)((words[wi] >> (4 * j)) & 0xF);
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const uint4* xp = reinterpret_cast<const uint4*>(X + (size_t)b * K + 32 * i);
            float dot = 0.f, sum = 0.f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 xv = __ldg(xp + c);
                const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const float2 f = __half22float2(*reinterpret_cast<const half2*>(&xw[h]));
                    dot = fmaf(q[8 * c + 2 * h], f.x, dot);
                    dot = fmaf(q[8 * c + 2 * h + 1], f.y, dot);
                    sum += f.x + f.y;
                }
            }
            acc[b] += sc * (dot - zp * sum);
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        const float r = warp_reduce_sum(acc[b]);
        if (lane == 0) Y[(size_t)b * N + n] = __float2half(r);
    }
}

// ---------------------------------------------------------------- lane-balanced variant
// L lanes cooperate on one row (L = 8, 16 or 32), so 32/L rows share a warp. K = 1536 has only 48 chunks of 32
// weights per row: with L = 32 half the lanes do two chunks and half do one; with L = 16 every lane does exactly
// three. U chunks are loaded before any is consumed (memory-level parallelism).
template <int B, int L, int U>
__global__ void gemv_int4_lane(const uint32_t* __restrict__ W, const half* __restrict__ S,
                               const half* __restrict__ Z, const half* __restrict__ X,
                               half* __restrict__ Y, int N, int K, int G) {
    constexpr int RPW = kWarpSize / L;
    const int warp = threadIdx.x / kWarpSize, lane = threadIdx.x % kWarpSize;
    const int sub = lane / L, l = lane % L;
    const int n = (blockIdx.x * (kBlockThreads / kWarpSize) + warp) * RPW + sub;
    const bool valid = n < N;
    const int nn = valid ? n : 0;
    const uint4* row = reinterpret_cast<const uint4*>(W + (size_t)nn * (K / 8));
    const half* s = S + (size_t)nn * (K / G);
    const half* z = Z + (size_t)nn * (K / G);
    const int iters = K / 32;
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;

    for (int i0 = l; i0 < iters; i0 += L * U) {
        uint4 v[U];
        float sc[U], zp[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = i0 + u * L;
            if (i < iters) {
                v[u] = ldg_stream(row + i);
                const int g = (32 * i) / G;
                sc[u] = __half2float(s[g]);
                zp[u] = __half2float(z[g]);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = i0 + u * L;
            if (i < iters) {
                float q[32];
                const uint32_t words[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
                for (int wi = 0; wi < 4; ++wi)
#pragma unroll
                    for (int j = 0; j < 8; ++j) q[8 * wi + j] = (float)((words[wi] >> (4 * j)) & 0xF);
#pragma unroll
                for (int b = 0; b < B; ++b) {
                    const uint4* xp = reinterpret_cast<const uint4*>(X + (size_t)b * K + 32 * i);
                    float dot = 0.f, sum = 0.f;
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint4 xv = __ldg(xp + c);
                        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                        for (int h = 0; h < 4; ++h) {
                            const float2 f = __half22float2(*reinterpret_cast<const half2*>(&xw[h]));
                            dot = fmaf(q[8 * c + 2 * h], f.x, dot);
                            dot = fmaf(q[8 * c + 2 * h + 1], f.y, dot);
                            sum += f.x + f.y;
                        }
                    }
                    acc[b] += sc[u] * (dot - zp[u] * sum);
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) {
        float r = acc[b];
#pragma unroll
        for (int off = L / 2; off > 0; off >>= 1) r += __shfl_xor_sync(0xffffffffu, r, off);
        if (l == 0 && valid) Y[(size_t)b * N + n] = __float2half(r);
    }
}

// B = 1 launcher for the experimental lane/unroll variants used by the benchmark. Returns false for unknown (L, U).
inline bool launch_lane1(int L, int U, const uint32_t* W, const half* S, const half* Z, const half* X, half* Y, int N,
                         int K, int G, cudaStream_t st = 0) {
    const int rows_per_block = (kBlockThreads / kWarpSize) * (kWarpSize / L);
    const int grid = (N + rows_per_block - 1) / rows_per_block;
#define INFERNO_LANE_CASE(LL, UU)                                                                            \
    if (L == LL && U == UU) {                                                                                \
        gemv_int4_lane<1, LL, UU><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G);                  \
        return true;                                                                                         \
    }
    INFERNO_LANE_CASE(32, 1) INFERNO_LANE_CASE(32, 2) INFERNO_LANE_CASE(32, 4)
    INFERNO_LANE_CASE(16, 1) INFERNO_LANE_CASE(16, 2) INFERNO_LANE_CASE(16, 3) INFERNO_LANE_CASE(16, 4)
    INFERNO_LANE_CASE(8, 1) INFERNO_LANE_CASE(8, 2) INFERNO_LANE_CASE(8, 4) INFERNO_LANE_CASE(8, 6)
#undef INFERNO_LANE_CASE
    return false;
}

// Returns false if B is not one of the instantiated batch sizes.
inline bool launch_batch(int B, const uint32_t* W, const half* S, const half* Z, const half* X, half* Y,
                         int N, int K, int G, cudaStream_t st = 0) {
    const int grid = grid_rows(N);
    switch (B) {
        case 1:  gemv_int4_batch<1><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 2:  gemv_int4_batch<2><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 3:  gemv_int4_batch<3><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 4:  gemv_int4_batch<4><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 5:  gemv_int4_batch<5><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 6:  gemv_int4_batch<6><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 7:  gemv_int4_batch<7><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 8:  gemv_int4_batch<8><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 12: gemv_int4_batch<12><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        case 16: gemv_int4_batch<16><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G); return true;
        default: return false;
    }
}

// Production launcher: 16 lanes per row (two rows per warp). Measured best or tied on every shape at B = 1
// (bench_int4_gemv --variants); the unroll factor U > 1 never helped, so it is fixed at 1.
inline bool launch_batch16(int B, const uint32_t* W, const half* S, const half* Z, const half* X, half* Y,
                           int N, int K, int G, cudaStream_t st = 0) {
    const int grid = (N + (kBlockThreads / kWarpSize) * 2 - 1) / ((kBlockThreads / kWarpSize) * 2);
#define INFERNO_B16(BB)                                                                       \
    case BB:                                                                                  \
        gemv_int4_lane<BB, 16, 1><<<grid, kBlockThreads, 0, st>>>(W, S, Z, X, Y, N, K, G);    \
        return true;
    switch (B) {
        INFERNO_B16(1) INFERNO_B16(2) INFERNO_B16(3) INFERNO_B16(4) INFERNO_B16(5) INFERNO_B16(6) INFERNO_B16(7)
        INFERNO_B16(8) INFERNO_B16(12) INFERNO_B16(16)
        default: return false;
    }
#undef INFERNO_B16
}

}  // namespace inferno
