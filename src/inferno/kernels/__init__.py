"""
Custom CUDA kernels package.

The compiled extension (inferno_kernels) is a separate native module
built via `python kernels/setup.py build_ext --inplace` — it is not
importable until that build step has run (see scripts/build_cuda_kernel.sh).
"""
