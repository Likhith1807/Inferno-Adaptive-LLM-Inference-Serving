/*
 * C API around the small-batch fused int4 kernel, built as libinferno_int4.so so that Python (ctypes) can launch it
 * on PyTorch's current stream with raw device pointers. No PyTorch headers are needed, which keeps this buildable
 * with the CUDA toolkit alone (the venv's torch is built against a newer CUDA minor than the local nvcc).
 *
 *   returns 0 on success, 1 if B is not an instantiated batch size, 2 on a CUDA launch error.
 */
#include "int4_gemv_batch.cuh"

extern "C" int inferno_int4_gemv(int B, const void* W, const void* S, const void* Z, const void* X, void* Y, int N,
                                 int K, int G, void* stream) {
    if (K % 32 != 0 || G % 32 != 0 || K % G != 0) return 3;
    const bool ok = inferno::launch_batch16(B, reinterpret_cast<const uint32_t*>(W), reinterpret_cast<const half*>(S),
                                          reinterpret_cast<const half*>(Z), reinterpret_cast<const half*>(X),
                                          reinterpret_cast<half*>(Y), N, K, G, reinterpret_cast<cudaStream_t>(stream));
    if (!ok) return 1;
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

extern "C" int inferno_int4_max_batch() { return 16; }
