# Upstream contribution ideas

A merged upstream change is the strongest signal to the people who maintain this software. Pick one that your
own measurements motivate; do not open a PR that pads a list.

## How to find a good one

1. Read the vLLM (or Triton / PyTorch) issue tracker for `good first issue` and `help wanted` items around
   quantization, benchmarking, and scheduling, and look for one you have reproduced with Inferno's harness.
2. Prefer changes with a measurement attached: a benchmark that exposes a regression, a doc fix backed by a
   test, a kernel edge case (odd group sizes, non-multiple-of-32 K) with a failing test.
3. Keep the diff small, include the exact command and hardware that reproduces it, and follow the project's
   contributing guide (DCO/sign-off, formatting, CI).

## Candidate areas suggested by this project (verify against the current tracker first)

- **Benchmark methodology:** open-loop arrival with latency measured from scheduled time, and L2-flushing kernel
  microbenchmarks. If vLLM's own benchmark scripts lack either, a PR that adds an option is small and valuable.
- **Quantization kernel edge cases:** any shape or group-size combination where your CPU-reference checker finds a
  discrepancy in a library kernel.
- **Documentation:** a worked example of routing across heterogeneous replicas, with the measured trade-off.
- **Observability:** per-request queue time exposure if your controller needed a signal the server did not export.

## Do not

- Claim a benchmark win without the paired method described in `docs/methodology.md`.
- Submit generated code you cannot explain line by line in review.
