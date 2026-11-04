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
