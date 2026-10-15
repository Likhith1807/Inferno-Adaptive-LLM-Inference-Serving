/*
 * Small-batch benchmark: where does a CUDA-core fused int4 kernel stop beating fp16 cuBLAS as the number of
 * activation vectors B grows? Same measurement discipline as bench_int4_gemv.cu (rotating weight copies past L2,
 * double-precision CPU reference gate before any timing, median of batches of back-to-back launches).
 *
 * The byte-ratio ceiling for batch B counts the activations and outputs too:
 *   ceiling(B) = (2NK + B(2K+2N)) / (int4 bytes + B(2K+2N)).
 * Output: table on stderr, JSON on stdout.
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

#include "int4_gemv_batch.cuh"

#define CUDA_CHECK(x)                                                                             \
    do {                                                                                          \
        cudaError_t e = (x);                                                                      \
        if (e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)
#define CUBLAS_CHECK(x)                                                                \
    do {                                                                               \
        cublasStatus_t s = (x);                                                        \
        if (s != CUBLAS_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

struct Shape { const char* name; int N, K; };

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double urand() {
    g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
    return (double)((g_rng * 2685821657736338717ull) >> 11) / 9007199254740992.0;
}
static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(size_t)std::min<double>(v.size() - 1, std::floor(p * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
    int G = 128, iters = 50, reps = 21;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_clock_khz = 0, bus_bits = 0, l2_bytes = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
    const double peak_gbps = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    const size_t flush_bytes = std::max<size_t>(768ull << 20, (size_t)l2_bytes * 24);
    fprintf(stderr, "GPU: %s | peak DRAM %.1f GB/s | group %d\n", prop.name, peak_gbps, G);

    const Shape shapes[] = {
        {"qwen1.5B_gate_up   (N=17920,K=1536)", 17920, 1536},
        {"qwen1.5B_down      (N=1536,K=8960)", 1536, 8960},
        {"llama7B_mlp_up     (N=11008,K=4096)", 11008, 4096},
    };
    const int batches[] = {1, 2, 3, 4, 6, 8, 12, 16};

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    printf("{\n  \"device\": \"%s\", \"peak_dram_gbps\": %.1f, \"group_size\": %d,\n  \"shapes\": [\n", prop.name, peak_gbps, G);
    bool first_shape = true;
    for (const Shape& sh : shapes) {
        const int N = sh.N, K = sh.K;
        const size_t n_words = (size_t)N * (K / 8), n_groups = (size_t)N * (K / G), n_w = (size_t)N * K;
        std::vector<uint32_t> h_W(n_words);
        std::vector<half> h_S(n_groups), h_Z(n_groups);
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
        const size_t bytes_int4 = n_words * 4 + n_groups * 4, bytes_fp16 = n_w * 2;
        const int copies_q = (int)std::max<size_t>(2, (flush_bytes + bytes_int4 - 1) / bytes_int4);
        const int copies_f = (int)std::max<size_t>(2, (flush_bytes + bytes_fp16 - 1) / bytes_fp16);
        std::vector<uint32_t*> d_W(copies_q);
        std::vector<half*> d_S(copies_q), d_Z(copies_q), d_F(copies_f);
        for (int c = 0; c < copies_q; ++c) {
            CUDA_CHECK(cudaMalloc(&d_W[c], n_words * 4));
            CUDA_CHECK(cudaMalloc(&d_S[c], n_groups * 2));
            CUDA_CHECK(cudaMalloc(&d_Z[c], n_groups * 2));
            CUDA_CHECK(cudaMemcpy(d_W[c], h_W.data(), n_words * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_S[c], h_S.data(), n_groups * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_Z[c], h_Z.data(), n_groups * 2, cudaMemcpyHostToDevice));
        }
        std::vector<half> h_F(n_w);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) {
                size_t g = (size_t)n * (K / G) + k / G;
                h_F[(size_t)n * K + k] = __float2half((float)(q[(size_t)n * K + k] - __half2float(h_Z[g])) * __half2float(h_S[g]));
            }
        for (int c = 0; c < copies_f; ++c) {
            CUDA_CHECK(cudaMalloc(&d_F[c], n_w * 2));
            CUDA_CHECK(cudaMemcpy(d_F[c], h_F.data(), n_w * 2, cudaMemcpyHostToDevice));
        }

        fprintf(stderr, "\n%s\n  %3s %12s %12s %10s %9s %9s %9s  %s\n", sh.name, "B", "cublas_us", "int4_us", "speedup",
                "ceiling", "%ceil", "%peakBW", "max_rel_err");
        printf("%s    {\"name\": \"%s\", \"N\": %d, \"K\": %d, \"int4_bytes\": %zu, \"fp16_bytes\": %zu, \"batches\": [\n",
               first_shape ? "" : ",\n", sh.name, N, K, bytes_int4, bytes_fp16);
        first_shape = false;
        bool first_b = true;
        for (int B : batches) {
            std::vector<half> h_x((size_t)B * K);
            for (auto& v : h_x) v = __float2half((float)(urand() * 2.0 - 1.0));
            std::vector<double> ref((size_t)B * N);
            for (int b = 0; b < B; ++b)
                for (int n = 0; n < N; ++n) {
                    double acc = 0;
                    for (int k = 0; k < K; ++k) {
                        size_t g = (size_t)n * (K / G) + k / G;
                        acc += ((double)q[(size_t)n * K + k] - (double)__half2float(h_Z[g])) * (double)__half2float(h_S[g]) *
                               (double)__half2float(h_x[(size_t)b * K + k]);
                    }
                    ref[(size_t)b * N + n] = acc;
                }
            half *d_x, *d_y;
            CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * 2));
            CUDA_CHECK(cudaMalloc(&d_y, (size_t)B * N * 2));
            CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));
            const float alpha = 1.0f, beta = 0.0f;
            // Row-major X[B][K] and Y[B][N] == column-major (K x B) and (N x B).  Y = W * X^T.
            auto run_cublas = [&](int i) {
                CUBLAS_CHECK(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, B, K, &alpha, d_F[i % copies_f], CUDA_R_16F, K, d_x,
                                          CUDA_R_16F, K, &beta, d_y, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            };
            auto run_int4 = [&](int i) {
                inferno::launch_batch(B, d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G);
            };
            auto run_l16 = [&](int i) {
                inferno::launch_batch16(B, d_W[i % copies_q], d_S[i % copies_q], d_Z[i % copies_q], d_x, d_y, N, K, G);
            };
            auto check = [&](std::function<void(int)> fn, double& max_rel) {
                CUDA_CHECK(cudaMemset(d_y, 0, (size_t)B * N * 2));
                fn(0);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<half> h_y((size_t)B * N);
                CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size() * 2, cudaMemcpyDeviceToHost));
                double scale = 0;
                for (double r : ref) scale = std::max(scale, std::fabs(r));
                max_rel = 0;
                for (size_t i = 0; i < h_y.size(); ++i)
                    max_rel = std::max(max_rel, std::fabs((double)__half2float(h_y[i]) - ref[i]) / std::max(scale, 1e-6));
            };
            auto time_it = [&](std::function<void(int)> fn) {
                for (int i = 0; i < 20; ++i) fn(i);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<double> samples;
                int idx = 0;
                for (int r = 0; r < reps; ++r) {
                    CUDA_CHECK(cudaEventRecord(ev0));
                    for (int i = 0; i < iters; ++i) fn(idx++);
                    CUDA_CHECK(cudaEventRecord(ev1));
                    CUDA_CHECK(cudaEventSynchronize(ev1));
                    float ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                    samples.push_back(ms * 1e3 / iters);
                }
                return samples;
            };
            double rel_c = 0, rel_i = 0, rel_16 = 0;
            check(run_cublas, rel_c);
            check(run_int4, rel_i);
            check(run_l16, rel_16);
            const bool ok_c = rel_c < 5e-3, ok_i = rel_i < 5e-3 && rel_16 < 5e-3;
            auto sc = time_it(run_cublas), si = time_it(run_int4), s16 = time_it(run_l16);
            const double us_c = percentile(sc, 0.5), us_i = percentile(si, 0.5), us_16 = percentile(s16, 0.5);
            const double act = (double)B * (2.0 * K + 2.0 * N);
            const double ceil_b = ((double)bytes_fp16 + act) / ((double)bytes_int4 + act);
            const double gbps = ((double)bytes_int4 + act) / (us_i * 1e-6) / 1e9;
            const double speed = us_c / us_i;
            fprintf(stderr, "  %3d %12.2f %12.2f %9.2fx %8.2fx %8.0f%% %8.0f%%  %.1e/%.1e  L16: %8.2f us (%.2fx) %s\n", B, us_c, us_i, speed, ceil_b,
                    100.0 * speed / ceil_b, 100.0 * gbps / peak_gbps, rel_c, rel_i, us_16, us_c / us_16, (ok_c && ok_i) ? "" : "  <-- INCORRECT");
            printf("%s      {\"B\": %d, \"cublas_us\": %.3f, \"int4_us\": %.3f, \"cublas_us_p10\": %.3f, \"cublas_us_p90\": %.3f, "
                   "\"int4_us_p10\": %.3f, \"int4_us_p90\": %.3f, \"ceiling\": %.4f, \"gbps\": %.2f, \"pct_peak\": %.2f, "
                   "\"int4_l16_us\": %.3f, \"int4_l16_us_p10\": %.3f, \"int4_l16_us_p90\": %.3f, "
                   "\"rel_err_cublas\": %.3e, \"rel_err_int4\": %.3e, \"correct\": %s}",
                   first_b ? "" : ",\n", B, us_c, us_i, percentile(sc, 0.1), percentile(sc, 0.9), percentile(si, 0.1), percentile(si, 0.9),
                   ceil_b, gbps, 100.0 * gbps / peak_gbps, us_16, percentile(s16, 0.1), percentile(s16, 0.9), rel_c, rel_i, (ok_c && ok_i) ? "true" : "false");
            first_b = false;
            cudaFree(d_x);
            cudaFree(d_y);
        }
        printf("\n    ]}");
        for (int c = 0; c < copies_q; ++c) { cudaFree(d_W[c]); cudaFree(d_S[c]); cudaFree(d_Z[c]); }
        for (int c = 0; c < copies_f; ++c) cudaFree(d_F[c]);
    }
    printf("\n  ]\n}\n");
    cublasDestroy(cublas);
    return 0;
}
