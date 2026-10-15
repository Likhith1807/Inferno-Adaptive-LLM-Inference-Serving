/*
 * PyTorch C++ extension binding for the custom dequantization kernel.
 * Exposes it to Python as inferno_kernels.dequantize_int4_to_fp16(...)
 * so serving/harness.py can call it when config.use_custom_kernel is True.
 */
#include <torch/extension.h>

// Declared in dequant_kernel.cu
extern "C" void launch_dequantize_int4_to_fp16_naive(
    const uint8_t* packed_weights,
    const at::Half* scales,
    const at::Half* zero_points,
    at::Half* output,
    int num_elements,
    int group_size,
    cudaStream_t stream
);

torch::Tensor dequantize_int4_to_fp16(
    torch::Tensor packed_weights,
    torch::Tensor scales,
    torch::Tensor zero_points,
    int64_t num_elements,
    int64_t group_size
) {
    TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8, "packed_weights must be uint8");

    auto output = torch::empty(
        {num_elements},
        torch::TensorOptions().dtype(torch::kHalf).device(packed_weights.device())
    );

    // TODO: pass the real CUDA stream (at::cuda::getCurrentCUDAStream())
    // instead of the default stream once this is wired into the harness.
    launch_dequantize_int4_to_fp16_naive(
        packed_weights.data_ptr<uint8_t>(),
        reinterpret_cast<at::Half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(zero_points.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(output.data_ptr<at::Half>()),
        (int)num_elements,
        (int)group_size,
        0  // default stream — TODO
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_int4_to_fp16", &dequantize_int4_to_fp16,
          "Custom int4->fp16 dequantization (CUDA)");
}
