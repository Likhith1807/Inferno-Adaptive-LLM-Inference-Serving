"""
Build script for the custom CUDA kernel as a PyTorch extension.

Usage:
    cd src/inferno/kernels
    python setup.py build_ext --inplace

Then in Python: import inferno_kernels
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="inferno_kernels",
    ext_modules=[
        CUDAExtension(
            name="inferno_kernels",
            sources=["dequant_kernel.cu", "dequant_binding.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
