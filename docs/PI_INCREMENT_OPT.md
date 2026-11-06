# Pi coding-agent follow-ups on two RX 7900 XTX cards

The reference is the best existing prefill build at commit `15345643` (v27b), using Swift-1.5 IQ3_S, a 200768-token engine context, FP32 prompt accumulation and one request at a time. The goal is **3× incremental prefill throughput and 1.2× decode throughput** relative to that build. The earlier `tp220` service was measured separately; it is not the acceptance baseline.

The actual Pi CLI (1.4.0, Node 26.13.0) ran the existing `python-utilities` and `merge-windows` tasks against about 74K tokens of repository history. Both tasks allow read/write/edit/bash in separate disposable workspaces and have independent checks. The client keeps medium thinking and a 32768-token output allowance. Its requests do not set sampling parameters; both arms retain the existing server configuration's temperature 0 and top-k 1. These are constructed tasks, not private conversations.

The v27b run produced 21 requests. Seventeen follow-ups had more than 32K cached tokens and fewer than 512 newly read tokens. They read 1997 tokens in 7592.4 ms, or **263.03 tokens/s**. Across all 21 requests, 10444 tokens took 64123.8 ms to decode, or **162.87 tokens/s** (158.37 on the small-follow-up subset). The corresponding targets for this fixed panel are 789.08 prefill and 195.45 decode tokens/s; the small-subset decode target is 190.04. Report total tokens divided by total engine time, not an average of individual request rates.

## Reading the existing expert cache during prefill

For short inputs, split prefill previously read immutable expert tables. A weight kept in host memory in those tables could already be in decode's GPU cache, but prefill still read its host copy. `FNX_PF_DECODE_CACHE=1` lets a single-lane prefill read decode's current per-layer gate/up and down pointer tables. Slot contents have the original quantized weight layout; the existing MMA kernels still perform the same dequantization and arithmetic.

Page growth runs before each pass. If KV needs a cache slot, the cache controller has already redirected the affected table entries to host weights before the prompt kernels use them. The prompt does not retain individual slot addresses across passes or modify decode's ownership or CLOCK state. Multiple lanes are refused with this option.

The first candidate used cache sources only for direct reads below the staging threshold. The second also uses the current sources in routed static stage mode 2. With `FNX_PF_STAGE_ROWS=1`, needed host weights are copied into the existing per-layer stage even for a small follow-up. A cache hit can be copied from GPU memory; a miss comes from the original host backing. The stage remains separate from the cache, and its original readiness and reuse ordering is unchanged. Stage byte totals in this mode can include device-to-device copies.

`FNX_PF_DECODE_CACHE_CHECK=1` is an optional diagnostic. It compares the complete gate/up and down bytes for routed experts served from cache with their immutable references. Disable it for performance measurements. With the first candidate it compared 40,787,993,600 bytes over 39,962 cached expert reads on the two cards. Both candidates passed a complete 21-request replay with the check on and a separate replay with it off. The original text, reasoning, tool names and arguments, finish reasons, generated/read/reused token counts and accepted/offered draft counts all matched. Server-generated tool-call IDs are not part of the semantic comparison.

| Fixed v27b requests | Small prefill tokens/s | Change | All decode tokens/s |
| --- | ---: | ---: | ---: |
| v27b reference | 263.03 | — | 162.87 |
| Direct cache reads | 368.67 | +40.2% | 163.11 |
| Cache sources and routed staging | 524.15 | +99.3% | 162.91 |

These are single screening comparisons on identical requests. The second candidate's small-subset decode was 158.39 tokens/s, essentially unchanged. Its sub-32-token inputs rose from 95.43 to 169.80 prefill tokens/s; its 256–511-token inputs rose from 415.75 to 866.73. **Neither the 3× aggregate prefill goal nor the 1.2× decode goal is met.**

A separate fresh process using the same candidate binary with cache reads disabled and the original staging threshold reproduced all 21 reference outputs and draft counts. Its small prefill rate was 262.78 tokens/s versus 524.15 with the settings enabled (1.995×); full-panel decode was 162.50 versus 162.91. The fixed workload therefore supports a substantial prefill gain, not a decode speedup.

A separate live Pi run passed both tasks without output truncation, taking 105.78 seconds versus 112.14 seconds for v27b. It made 18 instead of 21 requests, so its 574.33 small-prefill tokens/s cannot be attributed entirely to the engine change. The fixed replay above avoids that input-mix difference. Its full-task decode was 163.57 tokens/s.

Three zero-prefix-reuse API requests at each of 4096/16384/32768 tokens gave medians of 3622.2/4706.6/4698.5 prompt tokens/s. The engine and expert cache were already warm. A 200579-token input plus 64-token output completed without OOM, at 3664.8 prefill and 88.7 decode tokens/s. Its output and draft counts matched v27b. These long-input results guard capacity and the existing prefill path; they are not Pi incremental acceptance results.

Only HIP on these two gfx1100 cards was exercised. The second candidate binary is `build-tensor-parralize-baseline/strata-split-pi-cache-stage-v2` (SHA-256 `4f5df943b47c4016e0e2c64f49e876e68d5e62d7bace10c8f0b0b6d2405d9682`). Private experiment configurations, captured fixture traffic and detailed timing records are under `build-tensor-parralize-baseline/pi-increment-20261105/`; do not publish that directory wholesale.

## Pi-specific expert placement

The saved Pi profile now uses the same binary with a different existing-format expert ranking. It was built from two **different** Pi tasks (`refactor-orders` and `reservation-workflow`), both of which passed. The decode bank contains all 2,389,920 verified expert picks from those tasks. The prompt bank contains 538,620 picks from their six cached, sub-512-token follow-ups; it excludes cold prefix reads. Atomic per-request snapshots reconciled with the engine's expected pick counts for all ten requests. The profile contains aggregated expert counts, not prompt or output text.

| Evaluation on the original fixed Python/TypeScript requests | v27b | Pi placement | Gain |
| --- | ---: | ---: | ---: |
| Small incremental prefill | 263.03 | 597.01 | +127.0% |
| All decode | 162.87 | 169.43 | +4.0% |
| Small-follow-up decode | 158.37 | 168.04 | +6.1% |

All 21 requests still matched the original outputs, tool arguments, draft counts and prefix reuse. A separate live run passed both tasks without truncation in 94.36 seconds. It generated 9233 tokens across 15 requests instead of 10444 across 21, so its 626.62 small-prefill tokens/s and task time are reported separately from the matched-input gain. Zero-prefix-reuse 4K/16K/32K medians were 3675.2/4687.0/4695.8 tokens/s. The 200579-input + 64-output case completed without OOM, with matching text and draft counts (3669.7 prefill, 88.9 decode tokens/s).

The 196608-byte ranking is `bench/results/2026-11-05-rx7900xtx-pi-increment50/pi-small-fresh-routes-v1.bin` (SHA-256 `70f80eba6d9e28b9506958902ff6206d1bab77e9e4fbaa45c3d47a456495718c`). The 3× prefill and 1.2× decode targets remain unmet. A tighter chunk-dependent MoE launch bound was also checked against 10,476 exhaustive maxima and the real GPU jobs. It changed 524.15 to 524.64 tokens/s in the cache/staging configuration, so it was not selected.

## Exact attention and prompt projections on two RX 7900 XTX cards

The fixed panel still contains all 21 original v27b requests. Seventeen cached follow-ups read 1997 fresh tokens. Decode includes all 10444 generated tokens. The table uses total tokens divided by total engine time, with one request in flight and a 200768-token context.

| Tokens/s | v27b | Paired QSA + Pi ranking | Ordered small-row GEMM + Q8 attention + sampled QSA | Goal |
| --- | ---: | ---: | ---: | ---: |
| Cached follow-up prefill | 263.03 | 607.19 | 701.79 | 789.08 |
| Decode across all 21 requests | 162.87 | 177.05 | 188.19 | 195.45 |
| Decode on the 17 follow-ups | 158.37 | 175.70 | 186.81 | 190.04 |

The last measured build uses these options on top of the existing Pi ranking.

- `FNX_QSA_PAIRED=1` shares one global cutoff between the two position partitions. `FNX_QSA_SAMPLE_PIVOT=1` proposes a pivot from a sample, then counts every visible key. It either proves the cutoff, selects exactly from the bounded candidate list, or uses full radix selection. Equal scores keep the earliest positions.
- `FNX_ATTN_HEAD_Q8_GATHER=1` computes the paged Q8 key/value row and scale addresses once per selected position for the head-split decode kernel.
- `STRATA_PF_HC_DOWN_INCREMENT=1` uses smaller output tiles for HC down at 64–512 rows. `FNX_PF_HC_DOWN_GSU3=1` and `FNX_PF_WIDE_GSU3=1` reproduce the serving wheel's three-way FP32 accumulation and tile-dependent K order for supported FP16 projections at 16–63 rows. Other shapes keep their existing dispatch. Elementwise comparisons used the actual rocBLAS 5.2.0.f7e037d2e5 wheel.

All 21 fixed requests matched v27b in content, reasoning, tool names and arguments, finish reason, fresh/reused and generated tokens, and draft counts. The GPU checks covered 10752 selector comparisons, 21504 attention/append comparisons and 2944 GEMM cases. The independent repeat-1 Pi run passed both code tasks, with 18 requests, 9315 generated tokens, 89.17 seconds and no truncation. v27b's repeat-0 run had 21 requests and 10444 generated tokens. Only the fixed panel measures the same work. In a separate earlier repeat, the Python agent generated a nonterminating iterator after the final item. We stopped the isolated task early and counted it as failed. Replaying its first four requests on v27b reproduced the same code, reasoning and tool arguments. One request had different draft counts.

Zero-prefix-reuse 4K, 16K and 32K API medians were 3615.8, 4759.2 and 4774.0 tokens/s. The guarded 200579-input, 64-output run reached 4068.6 prefill and 115.2 decode tokens/s. It matched v27b's output and draft counts. In an earlier continuous sequence on the wide-GEMM build, the original thermal guard stopped GPU 1 at its 100°C junction limit during the 200K request. The same binary completed the isolated request after cooling, with GPU junction peaks of 93°C and 96°C. Later long tests had separate cooling intervals.

The measured executable is `strata-split-pi-qsa-sample2-v15`, SHA-256 `cd3b4ec27a8c9929369523caca26adbec7b0a2852472054791817ced61d0bb88`. The existing-format Pi JSON and shell launcher use the original local thermal supervisor. Current measured gains against v27b are 166.8% for small prefill and 15.5% for all-request decode.

## V23 clean-build fixed-panel validation

The latest committed implementation borrows two residual scratch chunks from the independent, non-KV cache pool for smaller prompt buffer allocation. It keeps the original four-chunk pass and 4096-row chunk size. The normal CMake build from commit 72e15fdd was replayed against the same 21 captured v27b requests with no differences in generated content, reasoning, tool calls, finish reason, token counts, cache reuse or draft counts.

| Fixed 21-request panel | v27b | Clean V23 | Goal |
| --- | ---: | ---: | ---: |
| Cached follow-up prefill | 263.03 | 750.30 | 789.08 |
| Decode across all requests | 162.87 | 193.95 | 195.45 |
| Decode on the 17 small follow-ups | 158.37 | 192.14 | 190.04 |

The independent Pi repeat passed both coding tasks. It produced 9115 tokens in 85.42 seconds across 17 requests, with no truncations. The guarded 200579-input, 64-output request previously completed with v27b-matching output and draft counts. A later consecutive 200K check reached the guard's 100°C junction limit and was stopped. After cooling, the same prior V23 binary completed the isolated request with card junction peaks of 93°C and 96°C.

Other screened builds approached each threshold individually. V37 measured 787.12 prefill and 194.72 decode tokens/s. V38 measured 787.71 prefill and 193.84 decode. Neither cleared both targets. The latest clean V23 release binary is strata-split-pi-cache-residual-v23-release, SHA-256 c21a273e451a6f9f4ad631d319efd836e4d56ff9bb50592b45438a4e30629018. This binary was used for the V23 fixed replay.

These measurements leave the 3× prefill and 1.2× decode targets unmet.


## V37 release validation

The V37 release build enables sampled QSA for prefill chunks up to 512 rows, fused Q8 weight expansion for ordered FP32 short-row products, and MoE kernels that skip unused token tiles. It reuses the activation storage for the prompt residual, while keeping the validated head projection and cache pool boundaries.

| Fixed Pi replay in tokens/s | V23 | V37 release |
| --- | ---: | ---: |
| Small-increment prefill | 750.30 | 784.80 |
| Decode across 21 requests | 193.95 | 194.59 |
| Decode on the 17 small follow-ups | 192.14 | 192.71 |

The clean HIP build matched v27b on all 21 requests, including text, reasoning, tool calls, token counts, cache reuse, and draft counts. Both independent Pi coding tasks passed with no truncation. The median of three cold 32K prompts was 4797.9 tokens/s. The guarded 200579-input, 64-output request matched v27b's text and draft counts at 4036.6 prefill and 107.0 decode tokens/s, and the following request passed.

On this machine the release build had about 45 MiB free on each card after warm-up. The single 200K run on V23 measured 4091.8 prefill and 115.0 decode tokens/s.

The optional Pi JSON uses strata-split-pi-prefill-select-v37-release with SHA-256 0dcc38354b2efc5005f0d5dd876f4687cddd1cfd6b324fcd80b679cfe580f987. The original launcher reads this JSON and uses the existing thermal guard.

## ROCm 10.1 Pi release

The two-card Swift-1.5 IQ3_S configuration uses the independent expert pool for prefill phase buffers, residuals and addends. The loan is bounded to the active view and returned after GPU work completes. Short Q8 projections share weight expansion across rows. The selected settings also use packed IQ signs, eight state partitions for longer GDN verification windows and paired peer sends.

| Fixed Pi replay in tokens/s | V37 with ROCm 10.1 | Clean release |
| --- | ---: | ---: |
| Small-increment prefill | 786.87 | 835.15 |
| Decode across 21 requests | 199.58 | 208.89 |
| Decode on 17 small follow-ups | 197.80 | 206.64 |

The 21 captured requests matched the reference in text, reasoning, tool arguments, finish reasons, token counts, prefix reuse and draft counts. Both live Pi tasks passed without truncation. Three zero-reuse 32K prompts reached a median of 4817.8 prefill tokens/s.

The engine served 200579 input tokens plus 64 output tokens with matching text and draft counts. A separate request was cancelled during a prefill loan. Its retry and the cached follow-up matched the references. After warm-up the driver reported 28 and 29 MiB free on the two cards.

The local Pi JSON selects `strata-split-pi-rocm101-20261106-release` with SHA-256 `89f0264121d236827a36f503c3d26b2a97fbb5326b4070d5d835a56c39b4fe8f`. The existing launcher uses the JSON's ROCm 10.1 library directories. The running engine's library mappings were checked during qualification.
