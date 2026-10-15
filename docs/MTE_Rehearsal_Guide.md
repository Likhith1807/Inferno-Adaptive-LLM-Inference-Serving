# MTE rehearsal pack: the two results slides

**Deck update:** the deck was restructured to 15 slides. Kernel and vLLM are now ONE slide ("Results: The CUDA Kernel", slide 9); routing and MPI are now ONE
slide ("Results: Adaptive Routing & MPI", slide 10). The Q&A content below is unchanged (the underlying numbers did not change) and is grouped to match the
two merged slides. These are the two places where an examiner can push hardest, so each section has
(1) the numbers to know cold, (2) a timed script per topic, (3) the questions most likely to come, with model answers, and (4) what you must **not** say.
Every number below is generated from `docs/results/` (same loaders as the deck and the report).

**How to rehearse:** read the script aloud with the slide on screen, timing yourself; then have a teammate read only the *Question* lines and answer from memory;
then do the drills at the end. Target: **~3 min for slide 9 + ~3 min for slide 10 = 6 minutes** for these two slides. Keep answers to 20–30 seconds.

---

## 0. The numbers card (memorize these)

| Topic | Number | Say it as |
|---|---|---|
| Kernel speedup, Llama-7B MLP, batch 1 | **3.67×** vs fp16 cuBLAS (98.3 µs vs 361.3 µs) | "3.7 times faster, kernel time, median of 5 runs" |
| Effective bandwidth | **244 GB/s = 95%** of 256 GB/s theoretical | "effective, calculated – not a profiler counter" |
| Measured streaming-read bandwidth | **252 GB/s** (98% of theoretical) → ours = 97% of it | "the honest ceiling on this GPU" |
| cuBLAS on the same layer | ~98% of peak | "both saturate bandwidth; we win by moving fewer bytes" |
| Byte-ratio ceiling | **3.76×** | fp16 bytes ÷ int4 bytes (scales + zeros included) |
| Other large layers | gate/up 3.61× (93%), attn 3.51× (90%), down 3.43× (87%) | "87 to 95 percent" |
| Small batches (gate/up) | B=1 3.6×, B=4 1.8×, B=8 1.2×, B=12 0.8×, B=16 0.5× | "advantage fades around B = 8" |
| vLLM, 1 sequence | custom **144** · Marlin **143** · classic AWQ **136** · fp16 **67** tok/s | "parity with Marlin (1.01×), 2.1× fp16" |
| vLLM ratio to Marlin | 1 seq 1.01 (range 1.003–1.019) · 2 seq 0.97 · 4 seq **0.68** · 8 seq **0.44** | "faster only in the sense of noise at 1" |
| Hybrid (≤ 2 tokens) | 143 / 268 / 512 / 954 tok/s (1.00 / 0.96 / 0.94 / 0.90 × Marlin) | "falls back to the classic kernel" |
| Verification | 112 patched layers, max rel. error 1.1e-3; identical 256-token output (1 prompt) | "a sanity check, not an accuracy study" |
| Routing campaign | 4 policies × 3 patterns × 8 reps = **96 runs**, 75 s each | |
| Adaptive vs best fixed p99 | steady 6.2→18.2 s (+195%) · bursty 11.4→24.3 s (+112%) · mixed 5.9→18.0 s (+204%) | "112 to 204 percent worse" |
| Statistics | Holm p = **0.023** (smallest attainable at n = 8), d_z 8–12, throughput 0.94–0.96× (guard failed) | |
| Contention (same total load) | 2nd replica active → p99 **×2.06 / ×2.08**; idle → **+0.8%**; throughput −4 to −6%, tokens/J −8 to −9% | |
| Headroom | best single tier (fixed-bulk) within **3.3%** of per-pattern best | |
| Load calibration | capacity 10.07 req/s; steady 6.04 (60%), bursts 15.1 (150%), mixed 8.27 req/s | |

---

## 1. Slide 9 – Results: The CUDA Kernel (kernel + vLLM, ≈ 3 min)

### Script

**(0:00)** "This slide answers: *is our kernel fast, and how do we know?* The left chart is the largest layer, the Llama-7B MLP, at **batch one**. Each bar is a version of the
kernel, speedup over fp16 cuBLAS. The naive version is *slower* than cuBLAS, 0.37×; each step adds speed, and we end at **3.67×**."

**(0:35)** "The four cards summarize it. 3.67× is kernel time, median of five runs. **244 GB/s** is our *effective bandwidth*: we count the bytes that must cross DRAM –
packed weights, scales, zeros, input, output – and divide by the measured kernel time. That is **95 percent of the 256 GB/s theoretical peak**, and 97 percent of the 252 GB/s that
a pure streaming-read kernel achieves on this GPU. It is a calculated number, not a profiler counter."

**(1:20)** "Why not 4×? int4 also stores scales and zeros, so the true byte ratio is **3.76×** – that is the ceiling, and we are at 98 percent of it. And every output matches a
double-precision CPU reference."

**(1:50)** "The right chart is the honest limit. For a few concurrent vectors the kernel still beats fp16 cuBLAS – 3.6× at B = 1, 1.8× at B = 4 – but around **B = 8** it breaks even,
and beyond that tensor cores win. So this is a **batch-one, small-batch** result."

**(2:25)** "Two honesty points. First, **Nsight Compute profiling is still pending** – GPU counters are blocked by a Windows driver setting – so our explanations of *why* each step helped
come from reasoning about access patterns, not from counters. Second, our first benchmark leaked about 6 percent through the GPU's L2 cache; we found it, fixed it, and re-ran everything."

### Likely questions

| # | Question | Model answer |
|---|---|---|
| 1 | **What does the kernel actually do?** | It computes y = W·x where W is stored as 4-bit integers. Each weight is decoded in registers and multiplied immediately, so the fp16 matrix is never written to memory. 16 lanes cooperate on a row, using 128-bit loads and a warp-shuffle reduction. |
| 2 | **Why 3.67× and not 4×?** | Bytes. fp16 = 2 bytes/weight. int4 = 0.5 bytes plus scale and zero (4 bytes per 128 weights = 0.031), i.e. 0.53 bytes. 2 ÷ 0.53 ≈ 3.76×. A memory-bound kernel cannot beat that ratio. |
| 3 | **cuBLAS is also near peak. What is your contribution?** | Right – both saturate bandwidth (cuBLAS ~98%, ours 95%). The gain is not better efficiency; it is that int4 moves 3.8× fewer bytes and we fused the dequantization so we do not pay for it in extra memory traffic. The unfused baseline (dequantize, then cuBLAS) is *slower* than plain fp16 (0.35×). |
| 4 | **Is 95% real? How was it measured?** | Effective bandwidth = bytes that must cross DRAM ÷ kernel time. Timing rotates 768 MB of weights so the 32 MB L2 cannot help; 5 fresh invocations, median. References: 256 GB/s theoretical from memory clock × bus width, and 252 GB/s measured by a streaming-read kernel. It is not from Nsight. |
| 5 | **You said you fixed a cache leak. What happened?** | One kernel first measured above 100% of theoretical, which is impossible. A sweep of the rotation footprint showed the same kernel was 93 µs at 192 MB but 98 µs at 768 MB or more; cuBLAS did not change. The L2 was keeping part of the working set. Default raised to 768 MB and everything re-run. |
| 6 | **How do you know the results are correct?** | Every kernel, every shape, every run is checked against a double-precision CPU reference before timing is trusted; worst relative error 4.4e-4 (fp16 rounding is ~5e-4). The first version of v4 failed this gate (wrong zero-point handling) and was fixed. |
| 7 | **Why does the advantage disappear at larger batch?** | The kernel runs on CUDA cores and costs roughly 3 instructions per weight per batch element, so it stops being memory-bound. Tensor-core kernels like Marlin keep scaling. That is why a tensor-core version for batches 8–64 is planned. |
| 8 | **Which version is "final", v5 or v6?** | They tie on this layer (98.3 vs 98.5 µs). v6 (16 lanes per row) helps the 1536-wide Qwen layers: o_proj about 15% faster, gate/up about 4%, in the same-invocation comparison. |
| 9 | **Why 16 lanes?** | A 1536-wide row has 48 chunks of 32 weights. With 32 lanes half the lanes do two chunks and half do one; with 16 lanes every lane does exactly three. It is a hypothesis consistent with the data – not confirmed by counters. |
| 10 | **Why is Nsight not done?** | Nsight needs access to GPU performance counters, and Windows blocks that by default (`ERR_NVGPUCTRPERM`). It is a one-time driver setting on the laptop; profiling is scheduled for 12–23 October. |
| 11 | **Why is the "down" layer only 87%?** | We do not know yet – that is exactly a question for Nsight. I would rather say that than guess. |
| 12 | **The small layers (qkv, o_proj) are only 44–47%. Why?** | They are 1–2 MB and take ~12–14 µs, so launch overhead and clock jitter dominate; only 3–4 of 5 runs agree within 2%. We do not draw conclusions from them. |
| 13 | **Why compare with cuBLAS, not the best library?** | cuBLAS is the fp16 baseline (what you get without quantization). The comparison with the best 4-bit library, Marlin, is the next slide, inside vLLM. |
| 14 | **What is a bank conflict, and did you prove yours?** | Threads in a warp reading shared memory in the same bank are serialized. From the access pattern we *predicted* a 16-way conflict in v2 and padded it away in v3 (about 9% faster). We have not confirmed it with counters, so we label it "predicted". |

### Do **not** say
- "We hit 95% of peak bandwidth" without the word *effective* or *calculated*.
- "Nsight shows / confirms …" – it has not been run.
- "We beat cuBLAS" without "fp16, batch 1".

---

### Continued on the same slide: the vLLM part (≈ 1.5 more min)

### Script

**(0:00)** "Kernel time is not token time, so we put the kernel *inside* vLLM and measured real decoding. Setup: Qwen2.5-1.5B, our RTX 4060 laptop, vLLM 0.11 with CUDA graphs, greedy
decoding, 256 tokens per sequence. The unit is **output tokens per second**, mean of three fresh processes. The baseline is **Marlin**, vLLM's default 4-bit kernel for this model on this GPU."

**(0:35)** "Read the leftmost group. With **one sequence – batch one** – fp16 gets 67 tokens per second, vLLM's classic AWQ kernel 136, Marlin 143, and **ours 144**. So we are about
**2.1 times faster than fp16** and **at parity with Marlin** – 1.01 times, within about 2 percent. I want to be clear: that is parity with a production library, **not a win**."

**(1:15)** "Moving right, with four or more sequences Marlin's tensor-core kernel pulls ahead – we reach 0.68 of Marlin at four sequences. That is expected: our kernel uses CUDA cores.
The fix is a tensor-core version, which is **planned and not started**."

**(1:45)** "How do we know it is correct? We checked all 112 patched layers against vLLM's own dequantization – maximum relative error 1.1 times ten to the minus three – and the generated
256 tokens are identical for every configuration on the test prompt. That is a sanity check, not an accuracy evaluation."

**(2:15)** "One more honest note: earlier in the project we believed we beat Marlin. A cross-check against these end-to-end numbers showed our Marlin timing was wrong – we retracted the claim
and corrected it. That is why the methodology has cross-checks."

### Likely questions

| # | Question | Model answer |
|---|---|---|
| 1 | **What exactly did you change in vLLM?** | We patched its AWQ linear layer. At load time we repack the weights once into our layout. During decode, steps with at most 8 tokens call our kernel through a torch custom op (so CUDA graphs can capture it); prefill and larger batches use vLLM's own path. Nothing else changes. |
| 2 | **Is 144 vs 143 a win?** | No. The ratio is 1.01 with a range of 1.003–1.019 over three repetitions, and the 95% interval on each mean is about ±2 tok/s. It is parity. |
| 3 | **Why not just use Marlin?** | For production, use Marlin. Our goal was to understand and build the kernel, reach the same place from scratch, and have a testbed. The value is the analysis and the method, not a claim of a better library. |
| 4 | **Why CUDA graphs?** | In eager mode Python launch overhead dominates: every configuration gets ~35 tok/s regardless of kernel, which hides the differences. Graph mode keeps the GPU busy like real serving. |
| 5 | **Why is the ≤ 8 config only 0.44× Marlin at 8 sequences?** | There our CUDA-core kernel is used for every decode step, but at batch 8 it is only at break-even with fp16 cuBLAS and far behind tensor cores. The hybrid that uses our kernel for ≤ 2 tokens and falls back to the classic kernel above that stays at 0.90–0.94× Marlin for 4–8 sequences. |
| 6 | **Why is fp16 2.1× slower – shouldn't int4 be 3.8×?** | The end-to-end token also includes work that is not quantized: the fp16 output projection over the 152k-word vocabulary (about 0.47 GB), attention, norms and sampling. So kernel-level gains dilute. |
| 7 | **Is the comparison fair?** | Same model, same process setup, same engine mode, three fresh processes each in rotated order; token streams compared. Limits: one model, one GPU, three repetitions. |
| 8 | **Did you check accuracy?** | Only that our kernel reproduces vLLM's own dequantized result to fp16 rounding and gives identical text on one prompt. Model quality of AWQ int4 itself is not evaluated – that belongs to the AWQ authors, and a task-accuracy check is planned. |
| 9 | **What was the retracted claim?** | An earlier in-situ Marlin timing was about 2× too slow, which made us look faster on every layer. End-to-end throughput disagreed; re-measurement in eager and graph mode gave the current numbers. A plausible cause is a GPU-wide slowdown in some profiling processes (2 of 18 recent ones showed every kernel 2–5× slower), but we cannot verify that for the old run. |
| 10 | **In-situ kernel times show you slower on gate/up (70 vs 60 µs). Why does standalone say 93% then?** | In-situ times vary between processes and engine modes (up to ~39% among normal processes), so they are noisier than the standalone benchmark. The end-to-end throughput is the more reliable summary, and it shows parity at one sequence. |
| 11 | **Why is Marlin the default?** | vLLM selects the Marlin path for AWQ checkpoints on Ampere-or-newer GPUs; we pinned it explicitly so the baseline is unambiguous. |
| 12 | **Would this hold on an A100/H100 or a 7B model?** | Unknown – single GPU, single model. The method transfers; the numbers may not. That is why a larger model on a cloud GPU is planned. |

### Do **not** say
- "We beat / outperform Marlin."
- "Faster than vLLM" (only faster than its *classic AWQ kernel* and fp16).
- "Accuracy is verified" (only numerical agreement with vLLM's dequantization).

---

## 2. Slide 10 – Results: Adaptive Routing & MPI (merged, ≈ 3 min)

### Script

**(0:00)** "Start with the banner, because it is the most important sentence on this slide: **H1 was rejected in the single-GPU setup, where both replicas share one physical GPU.
Multi-GPU validation is pending.**"

**(0:25)** "H1 was pre-specified: an adaptive router lowers p99 latency versus the best fixed configuration, at non-inferior throughput. The pool has two vLLM replicas – a low-latency one and
a high-throughput one – on the same GPU. We ran four policies on three load patterns, eight paired repetitions each: 96 runs, 75 seconds per run. Load was set at 60 percent of measured capacity,
with bursts at 150."

**(1:00)** "Left chart: p99 end-to-end latency in seconds. Blue is the best fixed policy, red is adaptive. Adaptive is **112 to 204 percent worse**; all eight paired repetitions agree, Holm-corrected
p = 0.023, and throughput was 94 to 96 percent, so the non-inferiority guard also failed."

**(1:35)** "Right chart: *why*. We held total load fixed and varied only how the GPU is shared. One replica: about 4.5 seconds. Two replicas with one **idle**: the same – idle residency is free.
Two replicas **both active**: 9.3 seconds – **2.1 times** worse. Two processes time-slice one GPU, and each sees smaller batches."

**(2:15)** "So the conclusion is narrow: adaptive routing did **not help when replicas share one GPU**. We do *not* claim routing is useless. A post-hoc check shows that even a perfect choice between the two tiers could
have gained only 3.3 percent here. The real test needs one replica per GPU – that is the cloud experiment planned for November, and it has **not been run**."

### Likely questions

| # | Question | Model answer |
|---|---|---|
| 1 | **What exactly was H1?** | For each load pattern, adaptive p99 latency is lower than the fixed policy with the lowest p99, at ≥ 98% of its throughput. It counts only if a paired t-test *and* a Wilcoxon signed-rank test both agree at Holm-adjusted α = 0.05. |
| 2 | **What is the adaptive policy?** | A 3-level controller. Level 0: all traffic to the fast tier. Level 1: long prompts (≥ 512 tokens) go to the bulk tier. Level 2: least-loaded across both. It escalates when the fast tier's utilization ≥ 0.70 or TTFT p95 exceeds 0.5 s, and de-escalates after 5 calm seconds. Thresholds were fixed before the campaign. |
| 3 | **What are the two replicas?** | Same int4 model. Fast: max 16 sequences, 512-token prefill chunks. Bulk: max 64 sequences, 4096-token chunks. Each gets 36% of GPU memory. |
| 4 | **Why is p = 0.023 not smaller?** | With 8 pairs the exact Wilcoxon test cannot go below 2/2⁸ = 0.0078; Holm over three patterns turns that into 0.023. It is the smallest value attainable, reached because all 8 pairs point the same way. |
| 5 | **Cohen's d_z of 8–12 looks unrealistic.** | It is a *paired* effect size: identical seeded request streams in each pair remove between-stream noise, so the differences are very consistent. It says "very consistent", and I would not read it as a general-purpose effect size. |
| 6 | **Did you tune the controller after seeing results?** | No. Thresholds were fixed before the campaign. The contention experiment and the headroom analysis were designed *afterwards* and are labelled explanatory / post hoc. |
| 7 | **So is adaptive routing useless?** | No. We tested it in one configuration where its precondition – independent capacity per replica – fails. On separate GPUs the penalty should disappear; that is a prediction we have not yet tested. |
| 8 | **Why not test with two GPUs now?** | We have one 8 GB laptop GPU. The cloud multi-GPU setup is written in Terraform but not applied; it needs an account and budget and is scheduled for 26 Oct – 27 Nov. |
| 9 | **Why not change the batch size dynamically?** | vLLM cannot change batch cap or precision on a running server; a restart takes about 48 seconds, longer than a 10-second burst. So adaptivity had to be routing across pre-loaded replicas. |
| 10 | **What does the 3.3% headroom mean?** | Best single fixed tier (bulk) vs the per-pattern best tier: bulk wins steady and bursty; fast wins mixed by 3.3%. So on this pool a perfect switch between them could gain at most 3.3% – the test was close to unwinnable regardless of contention. |
| 11 | **Are the workloads realistic?** | They are seeded synthetic: Poisson arrivals, a burst pattern, and log-normal lengths plus a mixed short/long profile. They are repeatable but not real traces; real conversation traces are a planned improvement. |
| 12 | **Why 60% of capacity?** | Rates come from measured capacity (10.07 req/s), not guesses; a guessed rate in our first smoke run saturated the GPU. Steady = 60% (6.04 req/s), bursts 150% (15.1 req/s). |
| 13 | **What about energy?** | In the contention study the second active replica lowered tokens per joule by 8–9% and throughput by 4–6%. Adaptive also had lower tokens/J than the best fixed policy. |
| 14 | **Why did adaptive track the static load balancer?** | After escalation the controller spreads traffic over both replicas – the same behaviour as least-loaded balancing – so it inherits the same contention penalty. |

### Do **not** say
- "Adaptive routing does not work" (say: *did not help on a single shared GPU*).
- "We tested multi-GPU" or "cloud results" – none exist.
- "p < 0.001" – the minimum attainable is 0.0078 raw / 0.023 Holm.

### MPI addendum (same slide, right-hand panel)

| Question | Model answer |
|---|---|
| **What does the MPI router do?** | A rank-0 dispatcher receives requests and sends each to a worker rank over MPI (non-blocking `isend`, `iprobe` polling); each worker owns one replica connection. |
| **What did you find?** | Overhead versus a plain HTTP least-outstanding router is below 1%, with no statistically significant improvement established (4 paired repetitions, both t-test and Wilcoxon checked). |
| **So why build it at all?** | The project explicitly asks for an MPI-coordinated system; it is also the honest baseline needed before claiming any coordination-layer benefit. We built it, measured it fairly, and reported that it is not the bottleneck. |
| **Why no significant result?** | Only 4 paired repetitions (fewer than the 8 used for routing), so the test has less power; and the effect itself is small because the GPU, not the network hop, is the constraint. |
| **What is next for MPI?** | Multi-GPU evaluation, where coordinating across separate physical GPUs is where MPI's distributed-memory design should start to matter more than on one shared GPU. |

---

## 3. Cross-slide traps and safe phrasing

| Risky sentence | Safe replacement |
|---|---|
| "95% of peak bandwidth" | "244 GB/s effective bandwidth, 95% of the theoretical peak, 97% of the measured streaming bandwidth" |
| "We beat Marlin" | "parity with Marlin at one sequence; Marlin is faster from four sequences" |
| "Nsight shows…" | "Nsight is pending; this is reasoned from access patterns" |
| "MPI speeds up the model" | "MPI coordinates requests; CUDA does the GPU math" |
| "Adaptive routing failed" | "H1 was rejected in the single-GPU setup; multi-GPU validation is pending" |
| "Everything is done except a few tasks" | "Kernel, vLLM integration, single-GPU campaign are done; Nsight, cloud multi-GPU and the tensor-core kernel are planned" |

## 4. Who answers what (rubric: "answer questions collectively")

| Question type | Lead | Backup |
|---|---|---|
| Kernel design, bandwidth, Nsight, batch limit | R1 – Likhith | R4 – A. Karthik |
| vLLM integration, verification, end-to-end | R4 – A. Karthik | R1 – Likhith |
| Statistics, calibration, workloads, p-values | R2 – Karthikeya | R3 – Sandeep |
| Routing policy, contention, MPI | R3 – Sandeep | R2 – Karthikeya |
| Plan / timeline / scope | Whoever is presenting – say "our plan is…" | any |

(Role assignment follows the order the names were given – swap it if the real split differs.)

Rule: the person asked answers first; a teammate may add one sentence. Never contradict a teammate on a number – use the numbers card.

## 5. Drills

**Rapid fire (answer in ≤ 5 seconds each)**
1. Speedup on the Llama MLP layer? → 3.67× vs fp16 cuBLAS.
2. Effective bandwidth? → 244 GB/s, 95% of 256 theoretical.
3. Byte-ratio ceiling? → 3.76×.
4. Break-even batch size? → about 8.
5. Tokens/s at one sequence, ours vs Marlin? → 144 vs 143.
6. Ratio to Marlin at four sequences? → 0.68.
7. Number of runs in the routing campaign? → 96 (4 × 3 × 8).
8. Adaptive p99 change? → 112 to 204% worse.
9. Effect of an idle second replica? → +0.8% (free).
10. Effect of an active second replica? → 2.1× p99.
11. Is Nsight done? → No, pending; counters blocked.
12. Has the multi-GPU cloud run happened? → No, planned for November.

**Hostile questions (practice staying calm and precise)**
1. "So your kernel is not better than Marlin. Why should we care?" → Parity from scratch, plus the methodology that exposed our own errors, plus a batch-1 and small-batch analysis; and a concrete plan to close the gap at higher batch.
2. "Your main hypothesis was rejected. Isn't the project a failure?" → A pre-specified hypothesis that could fail did fail; we found the mechanism and quantified the headroom; the multi-GPU test of the premise is next.
3. "How can you claim anything on a laptop GPU?" → We claim only what the setup supports: memory-bound batch-1 behaviour and single-GPU contention; the cloud run tests generalization.
4. "You changed your numbers during the project. Why should I trust them?" → Because we found the errors with cross-checks, corrected them, re-ran everything, and documented them; every number is generated from raw files.
5. "Which of these results would you bet on holding on an H100?" → The byte-ratio reasoning and the co-residency mechanism; not the exact numbers.

## 6. If time is cut (30-second versions)

- **Kernel:** "Fused int4 kernel: 3.67× faster than fp16 cuBLAS at batch 1, 244 GB/s effective bandwidth (95% of theoretical peak), advantage fades around batch 8; Nsight profiling pending."
- **vLLM:** "Inside vLLM at one sequence: 144 tokens/s – 2.1× fp16 and parity with Marlin, not a win; Marlin is faster from four sequences, so a tensor-core kernel is planned."
- **Routing:** "Adaptive routing was 112–204% worse on a single shared GPU because a second active replica doubles p99; the idle one is free. Multi-GPU validation is pending."

## 7. Delivery checklist
- Point at the chart element as you name it (the bar, the card, the banner).
- Say the unit every time: *µs, GB/s, tokens per second, seconds (p99)*.
- Say "batch one", "single GPU", "effective", "pending", "planned" out loud – the slides depend on those words.
- If you do not know: "We have not measured that yet; our hypothesis is … and it is scheduled for …" – never guess.
