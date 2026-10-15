| configuration | 1 seq | 2 seq | 4 seq | 8 seq |
|---|---|---|---|---|
| fp16 (no quantization) | 67 &plusmn; 0 (0.47x) | 132 &plusmn; 2 (0.47x) | 263 &plusmn; 3 (0.48x) | 515 &plusmn; 11 (0.49x) |
| AWQ-int4, vLLM classic kernel | 136 &plusmn; 4 (0.95x) | 264 &plusmn; 8 (0.94x) | 516 &plusmn; 8 (0.94x) | 968 &plusmn; 36 (0.91x) |
| AWQ-int4, vLLM Marlin (default) | 143 &plusmn; 2 | 280 &plusmn; 5 | 547 &plusmn; 7 | 1060 &plusmn; 6 |
| AWQ-int4, custom kernel (<= 2 tokens) | 143 &plusmn; 2 (1.00x) | 268 &plusmn; 11 (0.96x) | 512 &plusmn; 18 (0.94x) | 954 &plusmn; 68 (0.90x) |
| AWQ-int4, custom kernel (<= 8 tokens) | 144 &plusmn; 2 (1.01x) | 272 &plusmn; 7 (0.97x) | 370 &plusmn; 2 (0.68x) | 472 &plusmn; 6 (0.44x) |

Output tokens/s (mean of 3 process repetitions, +/- 95% CI); parentheses: paired ratio to AWQ-int4, vLLM Marlin (default).
