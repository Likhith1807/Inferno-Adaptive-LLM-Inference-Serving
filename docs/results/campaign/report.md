# Results: campaign_20260924_154321

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, driver 566.07, vLLM 0.11.0, commit None.
Repetitions per cell: 8 (mean ± 95% CI, t-distribution).

## Load pattern: bursty

| config | n | e2e_p50 | e2e_p95 | e2e_p99 | ttft_p99 | throughput_tok_per_s | goodput_req_per_s | tokens_per_joule |
|---|---|---|---|---|---|---|---|---|
| adaptive | 8 | 8.54 ± 0.52 | 21.9 ± 1.2 | 24.3 ± 1.4 | 7.98 ± 1.4 | 1.34e+03 ± 37 | 2.97 ± 0.17 | 16.3 ± 0.44 |
| fixed-bulk | 8 | 4.06 ± 0.18 | 10.4 ± 0.4 | 11.4 ± 0.2 | 3.13 ± 0.87 | 1.42e+03 ± 45 | 3.81 ± 0.089 | 18.2 ± 0.54 |
| fixed-fast | 8 | 12.3 ± 2.2 | 18.6 ± 1.7 | 20.1 ± 1.6 | 16.5 ± 1.7 | 1.28e+03 ± 42 | 0.101 ± 0.15 | 16.2 ± 0.47 |
| static-lb | 8 | 8.53 ± 0.35 | 21.8 ± 1.4 | 24.2 ± 1.5 | 7.86 ± 1.6 | 1.35e+03 ± 46 | 3 ± 0.11 | 16.5 ± 0.55 |

Strongest fixed baseline on p99 latency: **fixed-bulk**.

## Load pattern: mixed_length

| config | n | e2e_p50 | e2e_p95 | e2e_p99 | ttft_p99 | throughput_tok_per_s | goodput_req_per_s | tokens_per_joule |
|---|---|---|---|---|---|---|---|---|
| adaptive | 8 | 4.2 ± 0.16 | 13.5 ± 0.84 | 18 ± 0.91 | 0.134 ± 0.0094 | 1.35e+03 ± 37 | 7.49 ± 0.26 | 16.8 ± 0.45 |
| fixed-bulk | 8 | 1.6 ± 0.055 | 4.94 ± 0.16 | 6.13 ± 0.15 | 0.0559 ± 0.0031 | 1.42e+03 ± 39 | 7.91 ± 0.28 | 18.2 ± 0.51 |
| fixed-fast | 8 | 2.05 ± 0.25 | 5.03 ± 0.2 | 5.93 ± 0.36 | 1.94 ± 0.46 | 1.41e+03 ± 34 | 4.44 ± 0.95 | 17.9 ± 0.44 |
| static-lb | 8 | 4.25 ± 0.16 | 13.5 ± 1 | 17.6 ± 0.83 | 0.132 ± 0.0081 | 1.35e+03 ± 34 | 7.47 ± 0.26 | 16.7 ± 0.42 |

Strongest fixed baseline on p99 latency: **fixed-fast**.

## Load pattern: steady

| config | n | e2e_p50 | e2e_p95 | e2e_p99 | ttft_p99 | throughput_tok_per_s | goodput_req_per_s | tokens_per_joule |
|---|---|---|---|---|---|---|---|---|
| adaptive | 8 | 6.78 ± 0.31 | 16.7 ± 1.2 | 18.2 ± 1.3 | 0.121 ± 0.0056 | 1.49e+03 ± 37 | 5.33 ± 0.15 | 18.4 ± 0.48 |
| fixed-bulk | 8 | 2.68 ± 0.098 | 5.68 ± 0.29 | 6.18 ± 0.43 | 0.0548 ± 0.0062 | 1.58e+03 ± 38 | 5.66 ± 0.16 | 19.9 ± 0.51 |
| fixed-fast | 8 | 4.22 ± 0.92 | 7.14 ± 1.1 | 7.98 ± 1.2 | 3.66 ± 1.1 | 1.55e+03 ± 28 | 1.41 ± 0.88 | 19.6 ± 0.4 |
| static-lb | 8 | 6.93 ± 0.32 | 16.5 ± 1.2 | 18.6 ± 1.4 | 0.122 ± 0.0092 | 1.49e+03 ± 34 | 5.32 ± 0.15 | 18.3 ± 0.41 |

Strongest fixed baseline on p99 latency: **fixed-bulk**.

## H1: adaptive vs best fixed (paired by repetition, p99 latency)

| pattern | best_fixed | n_pairs | treatment_mean | fixed_mean | pct_change | p_holm | cohens_dz | throughput_ratio | verdict |
|---|---|---|---|---|---|---|---|---|---|
| bursty | fixed-bulk | 8 | 24.3 | 11.4 | 112 | 0.0234 | 8.22 | 0.944 | adaptive WORSE: p99 +112.3% (Holm p=0.0234) |
| mixed_length | fixed-fast | 8 | 18 | 5.93 | 204 | 0.0234 | 12 | 0.956 | adaptive WORSE: p99 +204.3% (Holm p=0.0234) |
| steady | fixed-bulk | 8 | 18.2 | 6.18 | 195 | 0.0234 | 9.98 | 0.942 | adaptive WORSE: p99 +195.3% (Holm p=0.0234) |

`pct_change` < 0 means adaptive has lower p99. p-values are Wilcoxon signed-rank, Holm-corrected across load patterns; a result counts as significant only if the paired t-test agrees. `throughput_ratio` is adaptive / best-fixed mean throughput.

