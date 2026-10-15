# Replica scaling: scaling_20260924_184645

MPI and HTTP routers replay identical seeded, saturating request streams (paired by repetition).

## Aggregate throughput and p99 (mean, ± 95% CI half-width)

| router | replicas | n | throughput | throughput_ci | p99 | p99_ci | errors | efficiency_vs_1x |
|---|---|---|---|---|---|---|---|---|
| http | 1 | 4 | 2.11e+03 | 31 | 74 | 7.12 | 0 | 1 |
| mpi | 1 | 4 | 2.12e+03 | 31 | 73.3 | 7.24 | 0 | 1 |
| http | 2 | 4 | 1.98e+03 | 35 | 82.2 | 6.63 | 0 | 0.47 |
| mpi | 2 | 4 | 1.98e+03 | 29 | 82.5 | 8.06 | 0 | 0.468 |

## MPI vs HTTP router (paired by repetition)

| replicas | metric | n_pairs | mpi_mean | http_mean | pct_diff_mpi_vs_http | p_ttest | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| 1 | throughput_tok_per_s | 4 | 2.12e+03 | 2.11e+03 | 0.582 | 0.0468 | 0.125 |
| 1 | e2e_p99 | 4 | 73.3 | 74 | -0.965 | 0.0479 | 0.125 |
| 2 | throughput_tok_per_s | 4 | 1.98e+03 | 1.98e+03 | 0.00928 | 0.945 | 1 |
| 2 | e2e_p99 | 4 | 82.5 | 82.2 | 0.323 | 0.635 | 1 |

`pct_diff_mpi_vs_http` > 0 means the MPI router's value is larger (higher throughput is better; higher p99 is worse). A difference is meaningful only if both tests agree and the effect exceeds the CI.

## Measured MPI hop overhead inside the real router (microseconds, mean over runs)

| replicas | mpi_dispatch_us_p50 | mpi_dispatch_us_p99 | mpi_result_us_p50 | mpi_result_us_p99 |
|---|---|---|---|---|
| 1 | 1.96e+03 | 1.8e+04 | 525 | 772 |
| 2 | 1.89e+03 | 6.34e+03 | 538 | 821 |

`dispatch` = dispatcher isend to worker seeing the job (includes the worker's 200 us poll interval); `result` = worker isend to dispatcher seeing the result. Compare with request latencies of seconds.
