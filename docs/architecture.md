# Architecture

```
              seeded workload generator (Poisson / bursty / bimodal / ShareGPT trace)
                                   |
                  open-loop replay, latency from scheduled arrival
                                   |
        +--------------------------+---------------------------+
        |                          |                           |
  HTTP routers               MPI dispatcher               Policy router (controller/live.py)
  (static LB baseline)       rank 0, isend/iprobe         Pinned | LeastLoaded | AdaptiveTier
        |                          |                           |
        |                    replica workers                   |
        |                    rank 1..N, asyncio               |
        +--------------------------+---------------------------+
                                   |
                     pool of resident vLLM replicas
             tier 0: low-latency decode   ...   last tier: throughput / long prefill
                                   |
             metrics (TTFT, TPOT, p99, goodput) + GPU power (nvidia-smi)
                                   |
                  analysis: paired stats -> report -> README block / dashboard
```

## Component boundaries

- **Policies are pure.** `controller/routing_policies.py` sees a `PoolState` snapshot and returns a replica index;
  no I/O, so decisions are unit-tested and auditable (every tick is logged with a reason).
- **Reconfiguration costs are real.** Changing batch size or quantization in vLLM needs a reload. The harness
  measures it (`InferenceHarness.reconfigure`), which is why the controller routes across pre-loaded configs.
- **MPI carries control and results, not tensors.** Point-to-point only; both hops are timestamped so overhead is
  measured directly rather than by subtraction. `mpi_router/routing.py` (policy + protocol) has no MPI import.
- **The kernel is standalone.** `kernels/bench_int4_gemv.cu` needs only nvcc and cuBLAS, so it builds on any CUDA
  machine and profiles cleanly under Nsight without a Python stack in the way.
- **Analysis reads files.** Everything downstream of an experiment is a pure function of `summary.csv` and the
  per-request JSONL, which makes results re-derivable and the dashboard a replay of real data.
