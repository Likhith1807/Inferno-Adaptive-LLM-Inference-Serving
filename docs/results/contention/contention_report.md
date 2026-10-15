# GPU co-residency study: contention_20260924_175525

Same total load, same seeded streams. `single` = one replica with the GPU to itself; `split-idle` = two resident replicas, all traffic to one; `split-active` = two replicas sharing the traffic.

## Latency, throughput, efficiency (mean; CI half-width for p99)

| pattern | config | n | e2e_p99 | e2e_p99_ci | throughput_tok_per_s | tokens_per_joule |
|---|---|---|---|---|---|---|
| rho0.3 | single | 5 | 4.46 | 0.13 | 828 | 10.9 |
| rho0.3 | split-idle | 5 | 4.49 | 0.15 | 828 | 10.9 |
| rho0.3 | split-active | 5 | 9.25 | 0.17 | 792 | 10.1 |
| rho0.6 | single | 5 | 6.21 | 0.6 | 1.6e+03 | 20 |
| rho0.6 | split-idle | 5 | 6.26 | 0.67 | 1.59e+03 | 20.1 |
| rho0.6 | split-active | 5 | 13 | 0.79 | 1.5e+03 | 18.2 |

## Paired comparisons

| pattern | comparison | metric | n_pairs | a_mean | b_mean | ratio_a_over_b | p_ttest | p_wilcoxon |
|---|---|---|---|---|---|---|---|---|
| rho0.3 | split-idle vs single | e2e_p99 | 5 | 4.49 | 4.46 | 1.01 | 0.0215 | 0.0625 |
| rho0.3 | split-idle vs single | throughput_tok_per_s | 5 | 828 | 828 | 1 | 0.165 | 0.188 |
| rho0.3 | split-idle vs single | tokens_per_joule | 5 | 10.9 | 10.9 | 1.01 | 0.0302 | 0.125 |
| rho0.3 | split-active vs split-idle | e2e_p99 | 5 | 9.25 | 4.49 | 2.06 | 0 | 0.0625 |
| rho0.3 | split-active vs split-idle | throughput_tok_per_s | 5 | 792 | 828 | 0.957 | 0.0007 | 0.0625 |
| rho0.3 | split-active vs split-idle | tokens_per_joule | 5 | 10.1 | 10.9 | 0.92 | 0.0003 | 0.0625 |
| rho0.6 | split-idle vs single | e2e_p99 | 5 | 6.26 | 6.21 | 1.01 | 0.242 | 0.438 |
| rho0.6 | split-idle vs single | throughput_tok_per_s | 5 | 1.59e+03 | 1.6e+03 | 1 | 0.0728 | 0.0625 |
| rho0.6 | split-idle vs single | tokens_per_joule | 5 | 20.1 | 20 | 1 | 0.0382 | 0.0625 |
| rho0.6 | split-active vs split-idle | e2e_p99 | 5 | 13 | 6.26 | 2.08 | 0 | 0.0625 |
| rho0.6 | split-active vs split-idle | throughput_tok_per_s | 5 | 1.5e+03 | 1.59e+03 | 0.944 | 0.0002 | 0.0625 |
| rho0.6 | split-active vs split-idle | tokens_per_joule | 5 | 18.2 | 20.1 | 0.908 | 0 | 0.0625 |

`ratio_a_over_b` above 1 means the first condition's value is larger (worse for p99, better for throughput).
