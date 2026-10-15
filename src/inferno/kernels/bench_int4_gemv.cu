/*
 * Standalone benchmark + correctness harness for the fused int4 GEMV kernels.
 * No PyTorch dependency: builds with plain nvcc + cuBLAS.
 *
 * Baselines compared against the custom kernels:
 *   cublas_fp16      cuBLAS GEMM (n=1) on the fp16 weights. What you get without
 *                    quantization; reads 4x more bytes.
 *   unfused_naive    naive dequant kernel materializes an fp16 matrix, then
 *                    cuBLAS. What you get from quantized storage without fusion.
 *
 * Measurement discipline:
 *   * Weights are rotated through enough distinct copies (>= 24x L2, >= 768 MB)
 *     that every launch streams from DRAM. Without this, small layers fit in L2 and
 *     report fictitious bandwidth. 6x L2 (192 MB) was NOT enough: a non-LRU L2 keeps a
 *     fraction of a cyclic working set, which made a 24 MB kernel ~6% too fast
 *     (scripts/flush_sweep.sh: 93 us at 192 MB vs 98 us at 768 MB and 1.5 GB).
 *   * Each timing sample is a batch of back-to-back launches (amortizes launch
 *     overhead); we report median / p10 / p90 over many batches.
 *   * Correctness is checked against a double-precision CPU reference before any
 *     timing is trusted.
 *
 * Output: a human-readable table on stderr and a JSON document on stdout.
 */
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "int4_gemv.cuh"
#include "int4_gemv_batch.cuh"

// Defined in dequant_kernel.cu (naive elementwise dequantization).
extern "C" void launch_dequantize_int4_to_fp16_naive(const uint8_t*, const half*, const half*, half*,
                                                    int, int, cudaStream_t);

#define CUDA_CHECK(x)                                                                    \
    do {                                                                                 \
        cudaError_t e = (x);                                                             \
        if (e != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)
#define CUBLAS_CHECK(x)                                                                  \
    do {                                                                                 \
        cublasStatus_t s = (x);                                                          \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__);   \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

struct Shape { const char* name; int N, K; };

// Read-only streaming kernel: the achievable DRAM read bandwidth on this device, measured rather than derived
// from the memory clock. It is the reference for "% of achievable" next to "% of theoretical".
__global__ void stream_read_kernel(const uint4* __restrict__ p, size_t n, uint32_t* out) {
    uint32_t a = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x9E3779B9u) out[0] = a;   // keeps the loads live
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double urand() {  // xorshift64*, deterministic across runs
    g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
    return (double)((g_rng * 2685821657736338717ull) >> 11) / 9007199254740992.0;
}

struct Result {
    std::string kernel;
    double us_median, us_p10, us_p90, gbps, pct_peak, max_abs_err, max_rel_err;
    bool correct;
};

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(size_t)std::min<double>(v.size() - 1, std::floor(p * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
    int G = 128, iters = 50, reps = 21, only_shape = -1, flush_mb = 0;
    bool variants = false;  // flush_mb = 0: default rotation size
    std::string only_kernel;   // substring filter; used for per-kernel profiling
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--group") && i + 1 < argc) G = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc) only_shape = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only_kernel = argv[++i];
        else if (!strcmp(argv[i], "--flush-mb") && i + 1 < argc) flush_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--variants")) variants = true;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_clock_khz = 0, bus_bits = 0, l2_bytes = 0, sm_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0);
    const double peak_gbps = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR: 2 transfers/clock
    const size_t flush_bytes = flush_mb > 0 ? ((size_t)flush_mb << 20) : std::max<size_t>(768ull << 20, (size_t)l2_bytes * 24);
    fprintf(stderr, "GPU: %s | SMs %d | L2 %.0f MB | theoretical DRAM BW %.1f GB/s | group %d\n",
            prop.name, prop.multiProcessorCount, l2_bytes / 1048576.0, peak_gbps, G);

    // Achievable read bandwidth: stream a 512 MB buffer (far beyond L2), median of 11 launches.
    double stream_gbps = 0;
    {
        const size_t bytes = 512ull << 20, n = bytes / sizeof(uint4);
        uint4* buf;
        uint32_t* sink;
        CUDA_CHECK(cudaMalloc(&buf, bytes));
        CUDA_CHECK(cudaMalloc(&sink, 4));
        CUDA_CHECK(cudaMemset(buf, 1, bytes));
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        const int blocks = prop.multiProcessorCount * 8;
        stream_read_kernel<<<blocks, 256>>>(buf, n, sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> t;
        for (int r = 0; r < 11; ++r) {
            cudaEventRecord(a);
            stream_read_kernel<<<blocks, 256>>>(buf, n, sink);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            t.push_back(bytes / (ms * 1e-3) / 1e9);
        }
        std::sort(t.begin(), t.end());
        stream_gbps = t[t.size() / 2];
        cudaFree(buf);
        cudaFree(sink);
        fprintf(stderr, "achievable streaming-read bandwidth (512 MB): %.1f GB/s = %.1f%% of theoretical\n", stream_gbps,
                100.0 * stream_gbps / peak_gbps);
    }

    // Qwen2.5-1.5B decoder projections as vLLM runs them (fused qkv and gate_up), then two Llama-7B shapes
    // for scale. Order matters: scripts/compare_kernels.py and profile_kernels.sh index into it.
    const Shape shapes[] = {
        {"qwen1.5B_qkv       (N=2048,K=1536)", 2048, 1536},
        {"qwen1.5B_o_proj    (N=1536,K=1536)", 1536, 1536},
        {"qwen1.5B_gate_up   (N=17920,K=1536)", 17920, 1536},
        {"qwen1.5B_down      (N=1536,K=8960)", 1536, 8960},
        {"llama7B_attn_proj  (N=4096,K=4096)", 4096, 4096},
        {"llama7B_mlp_up     (N=11008,K=4096)", 11008, 4096},
    };

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    printf("{\n  \"device\": \"%s\", \"peak_dram_gbps\": %.1f, \"stream_read_gbps\": %.1f, \"l2_mb\": %.1f, \"group_size\": %d,\n  \"shapes\": [\n",
           prop.name, peak_gbps, stream_gbps, l2_bytes / 1048576.0, G);
    bool first_shape = true;

    int shape_idx = -1;
    for (const Shape& sh : shapes) {
        ++shape_idx;
        if (only_shape >= 0 && shape_idx != only_shape) continue;
        const int N = sh.N, K = sh.K;
        if (K % 32 != 0 || K % G != 0) { fprintf(stderr, "skip %s (K not multiple of 32/G)\n", sh.name); continue; }
        const size_t n_words = (size_t)N * (K / 8), n_groups = (size_t)N * (K / G), n_w = (size_t)N * K;

        // ---- host data
        std::vector<uint32_t> h_W(n_words);
        std::vector<half> h_S(n_groups), h_Z(n_groups), h_x(K);
        std::vector<int> q(n_w);
        for (size_t w = 0; w < n_words; ++w) {
            uint32_t word = 0;
            for (int j = 0; j < 8; ++j) {
                uint32_t v = (uint32_t)(urand() * 16.0) & 0xF;
                q[w * 8 + j] = (int)v;
                word |= v << (4 * j);
            }
            h_W[w] = word;
        }
        for (size_t g = 0; g < n_groups; ++g) {
            h_S[g] = __float2half((float)(0.004 + 0.012 * urand()));
            h_Z[g] = __float2half((float)(std::floor(6.0 + 4.0 * urand())));
        }
        for (int k = 0; k < K; ++k) h_x[k] = __float2half((float)(urand() * 2.0 - 1.0));

        // ---- CPU double-precision reference (using the fp16-rounded scales/zeros/x the GPU sees)
        std::vector<double> ref(N);
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k) {
                size_t g = (size_t)n * (K / G) + k / G;
                acc += ((double)q[(size_t)n * K + k] - (double)__half2float(h_Z[g])) *
                       (double)__half2float(h_S[g]) * (double)__half2float(h_x[k]);
            }
            ref[n] = acc;
        }

        // ---- device buffers: rotating copies so each launch streams from DRAM
        const size_t bytes_int4 = n_words * 4 + n_groups * 2 * 2;
        const size_t bytes_fp16 = n_w * 2;
        const int copies_q = (int)std::max<size_t>(2, (flush_bytes + bytes_int4 - 1) / bytes_int4);
        const int copies_f = (int)std::max<size_t>(2, (flush_bytes + bytes_fp16 - 1) / bytes_fp16);
        std::vector<uint32_t*> d_W(copies_q);
        std::vector<half*> d_S(copies_q), d_Z(copies_q), d_F(copies_f), d_Wd(1);
        half *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), K * sizeof(half), cudaMemcpyHostToDevice));
        for (int c = 0; c < copies_q; ++c) {
            CUDA_CHECK(cudaMalloc(&d_W[c], n_words * 4));
            CUDA_CHECK(cudaMalloc(&d_S[c], n_groups * 2));
            CUDA_CHECK(cudaMalloc(&d_Z[c], n_groups * 2));
            CUDA_CHECK(cudaMemcpy(d_W[c], h_W.data(), n_words * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_S[c], h_S.data(), n_groups * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Z[c], h_Z.data(), n_groups * 2, cudaMemcpyHostToDevice));
        }
        // fp16 weights = exact dequantized values (for the cuBLAS baselines), built on the host.
        std::vector<half> h_F(n_w);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) {
                size_t g = (size_t)n * (K / G) + k / G;
                h_F[(size_t)n * K + k] = __float2half((float)(q[(size_t)n * K + k] - __half2float(h_Z[g])) *
                                                       __half2float(h_S[g]));
            }
        for (int c = 0; c < copies_f; ++c) {
            CUDA_CHECK(cudaMalloc(&d_F[c], n_w * 2));
            CUDA_CHECK(cudaMemcpy(d_F[c], h_F.data(), n_w * 2, cudaMemcpyHostToDevice));
        }
        half* d_Wdeq;  // scratch fp16 matrix for the unfused baseline
        CUDA_CHECK(cudaMalloc(&d_Wdeq, n_w * 2));

        const half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
        (void)alpha_h; (void)beta_h;
        const float alpha = 1.0f, beta = 0.0f;
        auto cublas_gemv = [&](const half* Wfp16) {
            // Row-major W[N][K] == column-major (K x N) with ld=K. y = op_T(A) * x.
            CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha, Wfp16, CUDA_R_16F, K,
                                      d_x, CUDA_R_16F, K, &beta, d_y, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        struct Kern {
            std::string name;
            std::function<void(int)> run;  // argument = rotation index
            double bytes;                   // bytes that must cross DRAM per launch
        };
        const double b_int4 = (double)bytes_int4 + K * 2.0 + N * 2.0;
        std::vector<Kern> kernels = {
            {"cublas_fp16", [&](int i) { cublas_gemv(d_F[i % copies_f]); }, (double)bytes_fp16 + K * 2.0 + N * 2.0},
            {"unfused_naive",
             [&](int i) {
                 launch_dequantize_int4_to_fp16_naive(reinterpret_cast<const uint8_t*>(d_W[i % copies_q]),
                                                      d_S[i % copies_q], d_Z[i % copies_q], d_Wdeq, (int)n_w, G, 0);
                 cublas_gemv(d_Wdeq);
             },
             b_int4 + 2.0 * bytes_fp16},  // read int4 + write fp16 matrix + read it back
            {"int4_v0_naive", [&](int i) { inferno::launch_v0(d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); }, b_int4},
            {"int4_v1_warp", [&](int i) { inferno::launch_v1(d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); }, b_int4},
            {"int4_v2_vec", [&](int i) { inferno::launch_v2(d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); }, b_int4},
            {"int4_v3_pad", [&](int i) { inferno::launch_v3(d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); }, b_int4},
            {"int4_v4_alg", [&](int i) { inferno::launch_v4(d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); }, b_int4},
            {"int4_v5_reuse", [&](int i) { inferno::launch_batch(1, d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); }, b_int4},
            {"int4_v6_lane16", [&](int i) { inferno::launch_batch16(1, d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); }, b_int4},
        };

        if (variants) {   // exploratory lane-count / unroll variants of the batch-1 kernel (not part of the ladder)
            const int LU[][2] = {{32, 1}, {32, 2}, {32, 4}, {16, 1}, {16, 2}, {16, 3}, {16, 4}, {8, 1}, {8, 2}, {8, 4}, {8, 6}};
            for (const auto& lu : LU) {
                const int L = lu[0], U = lu[1];
                kernels.push_back({"int4_v6_L" + std::to_string(L) + "_U" + std::to_string(U),
                                   [&, L, U](int i) { inferno::launch_lane1(L, U, d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G); },
                                   b_int4});
            }
        }
        fprintf(stderr, "\n%s  [int4 weights %.1f MB, fp16 %.1f MB, %d/%d rotating copies]\n", sh.name,
                bytes_int4 / 1048576.0, bytes_fp16 / 1048576.0, copies_q, copies_f);
        fprintf(stderr, "  %-15s %10s %10s %10s %9s %8s  %s\n", "kernel", "median_us", "p10_us", "p90_us", "GB/s", "%peak", "max_rel_err");

        std::vector<Result> results;
        for (Kern& kn : kernels) {
            if (!only_kernel.empty() && kn.name.find(only_kernel) == std::string::npos) continue;
            // correctness
            CUDA_CHECK(cudaMemset(d_y, 0, N * 2));
            kn.run(0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<half> h_y(N);
            CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * 2, cudaMemcpyDeviceToHost));
            double max_abs = 0, max_rel = 0, ref_scale = 0;
            for (int n = 0; n < N; ++n) ref_scale = std::max(ref_scale, std::fabs(ref[n]));
            for (int n = 0; n < N; ++n) {
                double err = std::fabs((double)__half2float(h_y[n]) - ref[n]);
                max_abs = std::max(max_abs, err);
                max_rel = std::max(max_rel, err / std::max(ref_scale, 1e-6));  // error relative to output scale
            }
            const bool ok = max_rel < 5e-3;  // fp16 output rounding dominates (~5e-4 relative)

            // timing: warmup, then `reps` batches of `iters` back-to-back launches
            for (int i = 0; i < 20; ++i) kn.run(i);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<double> samples;
            int idx = 0;
            for (int r = 0; r < reps; ++r) {
                CUDA_CHECK(cudaEventRecord(ev0));
                for (int i = 0; i < iters; ++i) kn.run(idx++);
                CUDA_CHECK(cudaEventRecord(ev1));
                CUDA_CHECK(cudaEventSynchronize(ev1));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                samples.push_back(ms * 1e3 / iters);
            }
            Result res{kn.name, percentile(samples, 0.5), percentile(samples, 0.1), percentile(samples, 0.9), 0, 0, max_abs, max_rel, ok};
            res.gbps = kn.bytes / (res.us_median * 1e-6) / 1e9;
            res.pct_peak = 100.0 * res.gbps / peak_gbps;
            results.push_back(res);
            fprintf(stderr, "  %-15s %10.2f %10.2f %10.2f %9.1f %7.1f%%  %.2e %s\n", res.kernel.c_str(), res.us_median,
                    res.us_p10, res.us_p90, res.gbps, res.pct_peak, res.max_rel_err, ok ? "" : "  <-- INCORRECT");
        }

        printf("%s    {\"name\": \"%s\", \"N\": %d, \"K\": %d, \"int4_bytes\": %zu, \"fp16_bytes\": %zu, \"kernels\": [\n",
               first_shape ? "" : ",\n", sh.name, N, K, bytes_int4, bytes_fp16);
        for (size_t i = 0; i < results.size(); ++i) {
            const Result& r = results[i];
            printf("      {\"kernel\": \"%s\", \"us_median\": %.3f, \"us_p10\": %.3f, \"us_p90\": %.3f, \"gbps\": %.2f, "
                   "\"pct_peak\": %.2f, \"max_rel_err\": %.3e, \"correct\": %s}%s\n",
                   r.kernel.c_str(), r.us_median, r.us_p10, r.us_p90, r.gbps, r.pct_peak, r.max_rel_err,
                   r.correct ? "true" : "false", i + 1 < results.size() ? "," : "");
        }
        printf("    ]}");
        first_shape = false;

        for (int c = 0; c < copies_q; ++c) { cudaFree(d_W[c]); cudaFree(d_S[c]); cudaFree(d_Z[c]); }
        for (int c = 0; c < copies_f; ++c) cudaFree(d_F[c]);
        cudaFree(d_Wdeq); cudaFree(d_x); cudaFree(d_y);
    }
    printf("\n  ]\n}\n");
    cublasDestroy(cublas);
    return 0;
}
