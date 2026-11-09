# Unsloth UD v3.1 long-context TP measurements

A later Pi task-completion measurement and the selected host conversation
cache setting are recorded in [UD31_PI_THROUGHPUT.md](UD31_PI_THROUGHPUT.md).
The earlier measurements below retain their original configurations. Later
sections record the selected two-card QSA query partition and HC row partition,
with the 512 MiB conversation cache. The November 9
[lookup and exact sampling comparison](#pi-lookup-and-exact-sampling-kernels)
uses model-card sampling and keeps the earlier accepted kernel options.

Measured on 2026-11-06 on two RX 7900 XTX cards (gfx1100), with a Ryzen 7 7700,
ROCm 10.1, Qwen3.8-Flash-Next-UD-IQ2_XS and a 262144-token context. The branch is
`ud3.1-tp-opt`, based at `f172b03f`. The reference already enables the accepted
`FNX_PF_Q8_K320=1` optimization. These results do not use the older Swift model
as a performance baseline.

## Selected long-prompt QSA changes

`FNX_PF_QSA_WIDE=1` uses four query rows and 256 pooled keys per score tile.
The reference uses 16 query rows and 64 keys. The new shape reduces shared
memory from 44 KiB to 18.5 KiB. Query conversion, FP32 WMMA accumulation and
the sum over heads retain their original order. Each query still scores every
visible key. The new dispatch requires a prompt selection stride above 512 and
at least 32 query rows; small increments and decode keep their existing path.

`FNX_PF_QSA_SAMPLE_LONG=1` lets the existing sampled-pivot selector handle long
prompt chunks. The chunk's output stride can be 4096 even though one selection
launch handles at most 512 queries. Sampling proposes a pivot; selection still
counts every visible key, retains the radix fallback and resolves ties in the
same chronological order. This option requires the existing paired and sampled
QSA controls. Both new options default off.

## Same-binary comparison

Both arms use the complete HIP build with SHA-256
`5a73e0cbf0d7dc77f85f09b0cf10e1c7a1b24da10d001845a027316436f19c32`.
Only the two options above differ. Each request starts with a cold prefix;
prefill cases generate 64 tokens. Inputs and API request bodies are frozen.

| Input tokens | Options off, tok/s | Options on, tok/s | Gain |
| ---: | ---: | ---: | ---: |
| 32768 | 4721.7 | 4804.8 | 1.76% |
| 65536 | 4619.2 | 4776.7 | 3.41% |
| 131072 | 4401.8 | 4637.1 | 5.35% |
| 204800 | 4054.9 | 4300.4 | 6.05% |

An earlier independent run measured 4807.2 / 4788.9 / 4649.4 / 4316.8 tok/s
with the options on. These are two measured runs, not a large-sample estimate.
Neither establishes 5500 tok/s at any of these lengths.

The three 200K code/prose/edit requests use a 2048-token API output limit.
The server can insert reasoning-close tokens, so decode throughput uses the
native engine's generated-token count and total decode time. The weighted
rate was 173.36 tok/s with the options off and 173.14 with them on. This is
effectively unchanged, not progress toward the requested 30% decode gain.
The original measured reference is 173.70 tok/s; its 30% target is 225.81.

All seven long requests matched in text, reasoning, tool arguments, finish
reason, actual native output count and cache/prompt/output/draft counters.
The options-off arm also matched the frozen pre-change executable. Formal
requests had zero major page faults.

## Pi replay and kernel checks

All 23 frozen Pi requests matched in request body, text, reasoning, tool calls,
finish reason and five token/cache/draft counters. The 19 small increments
measured 968.92 → 973.61 tok/s prefill and 232.31 → 232.38 tok/s decode.
Across all replay requests, decode was 227.72 → 227.71 tok/s. These results
show no material regression; the small differences are not claimed as gains.
The replay contains no 1–15-new-token request and does not establish performance
for that bucket.

The complete HIP build passed. Rebuilt tests passed 240 long-selector exact
checks with the real 4096 output stride, 336 score/graph/guard checks with the
real paged KV spacing, 220 K320 exact/graph/fallback checks and the existing
186 GEMM oracle/range/padding/fallback cases. Validation is for these two gfx1100
cards. CUDA and other GPUs were not tested; the split engine is not a SYCL target.

## Selected prompt attention and temporary buffers

`FNX_PF_ATTN_Q8_PACKED=1` expands four Q8 bytes with packed half operations
and reuses each lane's KV page address and K scale. It retains the direct
kernel's eight FP32 score partials, FP16 rounding and softmax order. The packed
product uses FMA with positive zero to match the reference's signed-zero
behavior. Earlier multiplication-only probes failed that check and were rejected.
Serving selects this path for at least 32 new rows on gfx1100; shorter chunks
retain the existing kernel.

`FNX_PF_BORROW_ADAPTIVE=1` can temporarily use the unused tail of the KV/cache
arena for eight prefill chunks when it cannot keep that loan for the entire
prompt. On the measured 200K request, it borrows 800 MiB per card through
131072 rows, then returns the pages before KV grows into them and continues
with four chunks. It preserves the configured context and the cache ownership
checks. Both options default off.

The same-binary comparison below keeps the two QSA options enabled. Only packed
attention and adaptive borrowing differ. Both arms use SHA-256
`7a43b199781a4d999416da78384ea832d9fcf0713e508e801fcb8a609f91dc1b`.

| Input tokens | New options off, tok/s | New options on, tok/s | Gain |
| ---: | ---: | ---: | ---: |
| 32768 | 4811.9 | 4901.7 | 1.87% |
| 65536 | 4786.5 | 4874.6 | 1.84% |
| 131072 | 4647.0 | 4732.5 | 1.84% |
| 204800 | 4310.9 | 4474.7 | 3.80% |

All seven long requests and 23 Pi requests matched, including draft counters.
Weighted 200K decode was 173.24 → 173.56 tok/s. Pi's small increments measured
973.65 → 972.96 tok/s prefill and 232.51 → 232.75 tok/s decode. These decode
and small-increment differences are within the observed noise.

The integrated complete HIP build has SHA-256
`acb2b66e4f8198094fbea6a5622b5bc22cf1b1190231a4d405723139928bd0cb`.
Repeating the full request sequence measured 4898.6 / 4866.4 / 4721.7 / 4467.8
tok/s prefill and 173.69 tok/s weighted 200K decode. Every output and counter
matched the experimental executable. Pi small prefill/decode measured
990.68 / 233.00 tok/s. Long requests had zero major page faults. An earlier
recheck shortened the requests preceding Pi: all Pi outputs matched, but two
requests' draft counters differed. Restoring the full sequence restored exact
counters; the shortened sequence is retained in the local evidence and is not
used for the exact-counter comparison. The GPU code and read-only data in all
21 device modules of the experimental and complete builds also match.

Rebuilt checks passed all 31744 finite nonnegative FP16 scales combined with
all 256 signed byte values on both cards, 1848 attention output/softmax/graph/
guard cases, the existing 21504 head-attention/append checks, and 38922 host
scratch ownership checks. A real cancelled prefill drained at 32768 rows and
returned its loan; retry output and draft counts matched. A 261901-token prompt
returned the expected marker, followed by a successful fresh short request,
with the same engine process throughout. This checks capacity and ownership,
not long-context retrieval quality. CUDA and other GPUs remain untested.

## Selected Q8 HC-up projection and mixer

`FNX_PF_HC_UP_Q8=1` reads the split Q8 weights directly into the HC-up
register-mixer tile. It avoids the full FP16 weight image and separate product
buffer. Weight conversion still rounds to FP16 first; the 20 ascending WMMA
steps and four-stream mixer retain the same order. The option defaults off and
requires the existing fused HC-up option, at least 64 rows, N=10240 and K=320
on gfx11. Shorter rows and unsupported shapes retain their previous dispatch.

Both arms below use SHA-256
`d08ee8ff2eb523e4fe75405d3a6016125e32a2e3ff23a5a55238695206572d2f`,
with all four preceding options enabled. Only HC-up Q8 differs.

| Input tokens | HC-up Q8 off, tok/s | HC-up Q8 on, tok/s | Gain |
| ---: | ---: | ---: | ---: |
| 32768 | 4895.6 | 4943.1 | 0.97% |
| 65536 | 4870.7 | 4909.9 | 0.80% |
| 131072 | 4726.6 | 4768.0 | 0.88% |
| 204800 | 4470.7 | 4507.6 | 0.82% |

All seven long and 23 Pi outputs and counters matched. Weighted 200K decode
was 173.76 → 173.55 tok/s, within the observed noise. Pi small prefill/decode
was 972.36 / 232.60 → 990.30 / 232.78 tok/s. The Pi prefill difference is not
claimed as a proven gain: untouched sub-64-row buckets improved too.

The complete HIP build has SHA-256
`971bc524633b9e9e7811bcfa8284417967e88a02b99de91c2bdde7ac6cc2fca2`.
Its complete seven-request and Pi replay matched the experimental executable,
including draft counters. Prefill measured 4941.8 / 4909.6 / 4768.6 / 4509.0
tok/s; weighted 200K decode was 173.43 tok/s. Pi small prefill/decode was
989.44 / 232.78 tok/s. Long requests had zero major page faults.

The rebuilt interface passed 112 Q8 exact output/graph/guard cases, 28 rejected
argument checks and 80 existing FP16 cases. The separate GEMM suite passed
220 K320 and 186 oracle/range/padding/fallback checks. Earlier direct-kernel
screens passed 176 and 224 cases. Tests cover both installed gfx1100 cards;
CUDA and other GPUs were not tested.

Three further probes were rejected. GDN coalesced state I/O passed 384 core and
5760 restoration checks, but made common three/four-row windows slower. Direct
Q8 staging in the HC-down BK64 tile passed 132 exact/graph/guard cases, but
smaller chunks regressed and the 4K benefit did not hold on both cards. Neither
probe is enabled or included in the serving dispatch. A third probe packs
two/four HC normalization lanes per thread while preserving the reduction
order. It passed 432 exact/graph/guard cases, but its improvement at 1024 rows
did not extend to the 4K serving chunks, which regressed on both cards. It was
also left out.

At this checkpoint, the private Unsloth configuration selected the HC-up
complete build and all five new options, retaining K320 fusion, both cards
and the 256K context.
Neither the 5500 tok/s prefill target nor the 225.81 tok/s decode target has been
reached. Local evidence and rollback configurations are under
`local-tuning/2026-11-06/ud31-long-opt/`. Private configurations contain
authentication settings and must not be published.

## Coalesced QSA selection with the Pi cache setting

Measured overnight on 2026-11-06 to 2026-11-07, on the same two RX 7900 XTX
cards and UD model. Both arms use the 512 MiB conversation cache, 262144-token
capacity and the same complete HIP binary:
`3ccaa9917fc966764f725d85f0377322860b8b0741556881ab7c1932ecf836f7`.
Only `FNX_QSA_COALESCED` changes between the arms. Each starts a new server,
executes the same seven long requests, then replays the same 23 Pi requests.
No builds or other GPU tests run during timing.

The opt-in selector reads adjacent scores with adjacent lanes. It stores the
comparison predicates as bitmaps, then forms chronological 64-key masks for
the existing compaction. Score calculation, the sampled cutoff and its full
radix fallback, global top-K membership and tie order remain unchanged.
The flag defaults off and applies only when the existing two-card HIP paired,
sampled selector is eligible.

| Input tokens | Flag off, prefill tok/s | Flag on | Change |
| --- | ---: | ---: | ---: |
| 32768 | 4821.3 | 4816.6 | -0.10% |
| 65536 | 4804.1 | 4799.1 | -0.10% |
| 131072 | 4696.1 | 4729.2 | +0.71% |
| 204800 | 4440.6 | 4547.6 | +2.41% |

The three additional 200K code/prose/edit requests improve prefill by
2.28%, 2.43% and 2.38%. Weighted 200K decode is 168.13 to 168.34 tok/s
(+0.13%); this does not establish a useful decode gain. These cache-enabled
rates must not be substituted for the earlier cache-disabled reference.
The original 5500 tok/s and 225.81 tok/s targets remain unmet.

All seven long outputs match: text, reasoning, tool calls, finish reason,
request hash, native generated count and five cache/prompt/output/draft
counters. All 23 Pi outputs and counters also match. Pi small-prefill rates
are 776.44 to 793.97 tok/s, and small-decode rates are 232.97 to 232.64 tok/s
(-0.14%). This is one complete pair; the small-bucket timing differences are
not a demonstrated improvement for every request. A subsequent live Pi
task-duration comparison on this QSA build measures the conversation-cache
setting, with QSA enabled in both arms: 121.08 to 94.26 seconds (-22.15%),
with all four task checks passing. That result is not a QSA kernel speedup;
generated trajectories differ and three shorter tasks slow down. See
[UD31_PI_THROUGHPUT.md](UD31_PI_THROUGHPUT.md).

The rebuilt selector test passes 17784 checks against a CPU top-K oracle,
including both GPUs, ties, sparse and dense selections, lane positions,
full-context boundaries, unused storage and graph replay. A near-limit API
request with 261888 input tokens returns `1517`; its follow-up returns
`1683`, reuses 261892 tokens and reads 31 fresh tokens. Both complete without
engine errors, OOMs or major page faults. CUDA and other GPUs were not tested.

This step selected this binary and `FNX_QSA_COALESCED=1`, retaining the other
qualified settings. The later query-partition step below replaces the binary
and retains this option. The rollback
configuration, build log, exact checks, full A/B captures, context check and
service verification are under
`local-tuning/2026-11-06/ud31-residual-opt/`; `qualification.json` records the
selection. Private configuration files contain credentials.

Two other probes remain disabled. Shared-expert fusion restricted to the
model's eight IQ2_XXS layers passes the full long/Pi comparison but changes
200K decode by only +0.10% and Pi decode by -0.10%. The extension covering the
other 40 IQ2_S layers passes 16384 GPU cache/output/graph checks and the full
seven-long/23-Pi same-binary comparison, with identical outputs and counters.
Measured on November 7 with the QSA option and 512 MiB conversation cache held
constant, it changes weighted 200K decode from 168.49 to 168.96 tok/s (+0.28%),
but Pi small decode falls from 232.87 to 229.29 tok/s (-1.54%). This variant
is also rejected; see `ud31-residual-opt/shared-all-decision.json`.
An aligned V-byte transpose passes 1848 GPU
attention checks, but does not improve the 4K-row kernel consistently. Neither
probe is included in the selected executable.

## Further probes on November 7

These experiments retain the QSA build and 512 MiB conversation cache on the
same two RX 7900 XTX cards. None changes the serving configuration.

HC loop unrolling passes 2352 exact output, guard, graph and barrier-wrap
checks on both cards. The common three- and four-row cases show small or
inconsistent gains between cached and flushed timing. Six-row down unrolling
spills and regresses cached timing. It was not taken to a model-level claim.

Expert-cache repartitioning is constrained by prefill scratch space. Moving
pool 1 from 288 to 768 slots fails the startup scratch-capacity check. Turning
off residual carry adds 400 MiB of independent buffers and fails with OOM
before requests run. The restored service has about 18 MiB of free VRAM per
card after all allocations; the earlier initialization figure overstates the
remaining capacity. Both failed experiments were isolated and rolled back.

A smaller 320-slot pool-1 partition fits the full-chunk scratch budget. A
same-binary pair of 200K code/prose/edit requests matches every output and
counter, with weighted decode changing from 168.3870 to 168.6933 tok/s
(+0.18%). This single pair does not establish a useful gain, so the existing
288-slot partition remains selected. Evidence is in
`ud31-residual-opt/hc-unroll-summary.json`, `cache-residual-pipeline.json`
and `cache320-pipeline.json`.

An isolated phase-scratch prototype computes MoE down columns in slices with
the existing WMMA kernels. It keeps the ten-expert FP32 sum order and reuses
attention temporaries after their last consumer. The prototype passes 1080
exact GPU comparisons covering the installed down formats, short and full
chunks, pointer tables, guards and graph replay. Only the changed prefill
object was rebuilt and linked against the qualified HIP objects.

The same executable, SHA-256
`42d82529ea99b2add09d00c5ea2fac3a1ccd1eed54479bcf4661955cecbd3a91`,
was measured with the options disabled, then with 1536- and 512-column slices.
Both candidates also alias the attention phases and repartition pool 1 from
288 to 480 slots. Each arm uses a new server, seven long requests and the same
23 Pi requests; all outputs and counters match. All long requests have zero
major page faults.

| Measurement | Options off | 1536 columns | 512 columns |
| --- | ---: | ---: | ---: |
| Full phase scratch, MiB | 477.5 | 317.5 | 304.875 |
| 32K prefill, tok/s | 4803.4 | 4807.7 | 4844.3 |
| 64K prefill, tok/s | 4776.6 | 4784.6 | 4816.3 |
| 128K prefill, tok/s | 4719.5 | 4716.9 | 4755.8 |
| 200K prefill, tok/s | 4543.2 | 4537.7 | 4577.4 |
| Weighted 200K decode, tok/s | 168.416 | 168.815 | 168.001 |
| Pi small prefill, tok/s | 798.62 | 795.43 | 777.04 |
| Pi small decode, tok/s | 232.85 | 233.04 | 232.98 |

Neither variant is selected. The 1536-column variant saves 160 MiB but does
not establish useful throughput gain. The 512-column variant improves long
prefill by 0.75–0.85% in this single comparison, while Pi small prefill falls
2.70% and long decode does not improve. No task-duration improvement is claimed
for these variants. Near-capacity and cancellation checks were not run for
this rejected prototype. The selected service and private configuration are
unchanged. See `ud31-residual-opt/prefill-slice-decision.json` for the evidence.

A separate down-projection probe decodes each IQ4_NL or Q5_0 weight block once
for all tokens of an expert job. It preserves the two original integer dot
partials, FP32 scaling and addition order, and retains the shared partial
buffer. The opt-in dispatch leaves single-token windows unchanged. Earlier
IQ4 reuse/direct-partial evidence is under `ud31-pi-opt2/iq4-down-v1`; this
comparison adds Q5_0 and tests the current QSA/cache configuration at 200K.

The new probe passes 1015808 exact GPU dot comparisons, covering all finite
FP16 weight scales, and 6912 output, guard and graph cases across both cards.
Cached three/four-row jobs using the same experts improve the IQ4_NL kernel
by about 11–17%, but distinct experts and flushed weights benefit little.

The same-binary full seven-long/23-Pi model comparison matches all outputs and
counters. Weighted 200K decode changes from 168.5813 to 168.5986 tok/s
(+0.0102%). Pi small prefill changes from 784.21 to 786.36 tok/s (+0.27%), and
small decode from 232.67 to 233.01 tok/s (+0.15%). This does not establish a
useful final-throughput gain, so the probe is not selected. All long requests
have zero major page faults. The experimental executable has SHA-256
`6bd37a68a925d5b3f7d3a57d17cc6fcd01fdb582b25996e7dca8e3174b00588d`;
only its changed HIP translation unit was rebuilt. Other backends, near-limit
capacity and cancellation were not tested for this rejected candidate.
See `ud31-residual-opt/moe-down-reuse-decision.json`. The qualified 8081
service was restored and passed ordinary and thinking output checks.

An isolated HC prefetch probe reads the next HC's immutable Q8 weights on
the shared-expert stream while the existing residual ring runs. It joins that
stream before HC so the resident-grid barrier retains exclusive execution.
No weight, activation, FP16 ring order or HC arithmetic changes. The option
defaults off. The two measured variants prefetch both down/up with 16 blocks,
or only down with 32 blocks. The latter avoids prefetching the up weights that
HC itself already loads ahead of its reduction barrier.

Both RX 7900 XTX cards pass 8736 exact output, guard, graph-replay and barrier-wrap
checks, covering T=1..12 and the one/two-barrier forms. Only the experimental
HIP engine object was rebuilt, against the qualified QSA build's other objects.
The binary SHA-256 is
`94677376559dc6c3add712e05b10b25cc727a8ad268d164caada284fd4d3d900`.
Each fresh process runs all seven long requests followed by the same 23 Pi
requests. Disabled versus production, and both candidates versus disabled,
match every output and counter. All long requests have zero major page faults.

| Measurement | Disabled | Both, 16 blocks | Down only, 32 blocks |
| --- | ---: | ---: | ---: |
| Weighted 200K decode, tok/s | 168.39 | 165.49 | 163.84 |
| Pi small decode, tok/s | 232.53 | 228.60 | 227.02 |
| Pi 23-request API time, seconds | 74.108 | 74.797 | 74.943 |

The API time is the sum of measured HTTP request durations, excluding tools
and agent work; it is not the separate four-task completion panel. Long decode
falls 1.72% and 2.70%. Pi small decode falls 1.69% and 2.37%, and API time grows
0.93% and 1.13%. Cold standalone HC timings improve for some grids, but neither
model variant is selected. These are single full comparisons; no other backend,
near-capacity or cancellation qualification was run for these rejected probes.
The unchanged 256K, two-card service was restored and ordinary/thinking checks
returned 323 and 1683. Evidence: `ud31-residual-opt/hc-prefetch-decision.json`
and `hc-prefetch-{pipeline,down-pipeline,summary}.json`.

## Selected two-card QSA query partition

Measured on November 7 on the same two RX 7900 XTX cards, UD model and 256K
configuration. `FNX_PF_QSA_QUERY_SPLIT=1` assigns alternating groups of four
queries to the cards. Each owned query still scores every visible key using
the original arithmetic and global selection. The cards exchange the selected
position lists, packed losslessly as 16-bit block IDs. This avoids duplicate
scoring and selection. It does not approximate top-K or discard remote keys.

The option defaults off. It requires the qualified two-card HIP peer/head-split
path and existing wide, paired, sampled and coalesced QSA options. It applies
only to prefill chunks larger than 512 tokens, query batches of 256–512 rows,
and positions at or beyond 24576. Decode and small increments use their
existing dispatch. Each card needs 1092 KiB of additional mailbox storage,
allocated after the ordinary engine buffers. Separate push/pull kernels,
alternating generations and bounded waits protect the exchange.

The final same-binary comparison uses executable SHA-256
`73056cb915bea58057511709761323a92d016fe06cc1996b589bdfcee80b31c5`.
Both fresh processes run the same seven long requests followed by the same
23 Pi requests, with the 512 MiB conversation cache enabled. Only the new
option changes between arms. No builds or other GPU tests run during timing.

| Input tokens | Flag off, prefill tok/s | Flag on | Change |
| --- | ---: | ---: | ---: |
| 32768 | 4804.8 | 4829.3 | +0.51% |
| 65536 | 4783.1 | 4821.1 | +0.79% |
| 131072 | 4715.9 | 4812.7 | +2.05% |
| 204800 | 4547.0 | 4670.5 | +2.72% |

The additional 200K code/prose/edit requests gain 2.85%, 2.64% and 2.77%
prefill. Two earlier complete pairs, with an earlier mailbox allocation
position, measured 2.70% and 2.81% at 200K. Weighted 200K decode in the final
pair is 168.819 to 168.677 tok/s (-0.08%); no useful decode gain is established.
The original 5500 tok/s prefill and 225.8145 tok/s decode goals remain unmet.

All seven long outputs and native generation/cache/draft counters match;
all 23 Pi outputs and counters also match. Pi small prefill is 791.56 to
800.95 tok/s and small decode is 232.778 to 232.831 tok/s. Total Pi API time
is 74.034 to 73.733 seconds (-0.41%), excluding tools and agent work.
The 64–255-token prefill bucket falls 1.53%; the other populated buckets
improve. Earlier pairs varied by about 1% in aggregate small prefill. These
results do not establish a small-increment speedup.

The separate four-task Pi panel completes in 95.16 versus 91.37 seconds
(-3.98%), with 4/4 final acceptance and zero HTTP errors in both arms.
Both issue 35 requests, but generate 14753 versus 13998 tokens, and the enabled
arm retries the reservation task. After ignoring generated tool-call IDs,
the first differing request is request 20 (zero-based), where the tool's test
output contains different elapsed times. Subsequent model trajectories differ.
This is not evidence of a further 5% gain attributable to the kernel.

The complete HIP build passes. The score, selector and exchange tests pass
19758 exact GPU comparisons and eight bounded timeout/malformed-payload checks
on both cards. The final allocation-order change rebuilds the engine object;
all tested kernel and dispatch sources are unchanged. A 261888-token request
returns 1517, and its follow-up returns 1683 while reusing 261892 tokens.
There is no OOM. After cancelling a cold 200K request following its first full
32768-row pass, the same engine answers a short request and reproduces the
qualified 32K output. CUDA, SYCL and other GPUs were not tested.

This selection enabled the new flag and executable, retaining K=320 fusion,
both GPUs, 256K context and the 512 MiB host cache. The HC selection below
subsequently updates the executable and retains the QSA option. Qualification, captures, build logs
and rollback configuration are under `local-tuning/2026-11-06/ud31-residual-opt/`:
`qsa-query-decision.json`, `qsa-query-late-pipeline.json`,
`qsa-query-late-pi-tasks-pipeline.json`, and
`qsa-query-late-capacity-pipeline.json`. Private configurations contain credentials.

## Output-projection peer push: not selected

The next isolated experiment sends each completed Q8 output tile to the peer
before the whole projection finishes. A separate pull retains the original
FP16 rounding and card summation order. It applies only to main-model mixer
projections at N=2560, K=3072 and 1–8 verification rows. Prefill, the FFN ring
and MTP dispatch are unchanged. The experiment adds no device allocation.

On the same two RX 7900 XTX cards, 3840 exact GPU checks pass, including both
activation layouts, guards, graph replay and tag wrap. With distinct matrices,
the projection-plus-ring probe saves about 1.5 microseconds at 3–4 rows.
An integer-WMMA alternative passes 1536 exact checks but is slower at these
common row counts, so it is not integrated.

The full-model comparison uses one binary, SHA-256
`f749a9a6a6529563be16540469e0e82e7c9945e8cd27482337aeb5805caefabd`,
with only `FNX_Q8_OUTPUT_PUSH` toggled. Each fresh process runs seven long
requests followed by 23 frozen Pi requests. All outputs and generation,
cache and draft counters match; neither arm reports an HTTP/engine error.
The disabled arm also matches the selected query-split binary.

| Metric | Disabled | Enabled | Change |
| --- | ---: | ---: | ---: |
| Weighted 200K decode, tok/s | 168.638 | 169.322 | +0.41% |
| Pi small decode, tok/s | 232.812 | 233.385 | +0.25% |
| Pi small prefill, tok/s | 790.19 | 780.36 | -1.24% |
| Total frozen Pi API time, seconds | 73.801 | 73.759 | -0.057% |

This single pair shows little overall benefit. It does not measure live agent
task completion or establish a further 5% reduction. The prototype remains
unselected and the daily configuration is unchanged. Evidence is in
`local-tuning/2026-11-06/ud31-residual-opt/q8-output-push-decision.json`,
`q8-ring-summary.json` and `q8-integer-wmma-summary.json`.

## Selected two-card HC row partition

Measured on November 7 on the same two RX 7900 XTX cards, UD model and 256K
configuration. `FNX_PF_HC_ROW_SPLIT=1` divides main-model HC-down token rows
between the cards for prefill chunks of 3072–4096 rows. Each card runs the
original FP32 GEMM on its rows, then exchanges the original rounded FP16
activation values. Weight conversion and the GEMM accumulation order stay
unchanged. The partition boundary follows the existing 64-row tile. Smaller
increments, decode and MTP keep their existing dispatch.

The option defaults off. It requires two probed HIP gfx1100 peers, one lane,
head splitting and native FP32 prefill GEMM. Each card allocates about 2562 KiB
after its ordinary buffers and the QSA mailbox. Finite push kernels precede
bounded pulls. Consumed ready tags are cleared, including slots temporarily
unused by shorter chunks. A GPU test reproduced stale data with the earlier
uncleared-tag prototype after generation wrap; the corrected protocol passes
the same delayed-peer, variable-size test. That prototype was never selected
for the daily service.

The final same-binary comparison uses SHA-256
`832ed3729d89738bd21293cd47219d90ce21b0e1b14c168ed64640a198ed5a37`.
Only the HC option changes. Each fresh process runs seven frozen long requests
followed by 23 frozen Pi requests, with QSA partitioning and the 512 MiB host
cache enabled in both arms. No builds or other GPU tests run during timing.

| Input tokens | HC flag off, prefill tok/s | HC flag on | Change |
| --- | ---: | ---: | ---: |
| 32768 | 4796.0 | 4865.5 | +1.45% |
| 65536 | 4799.0 | 4878.3 | +1.65% |
| 131072 | 4793.0 | 4872.1 | +1.65% |
| 204800 | 4670.7 | 4728.9 | +1.25% |

An earlier isolated pair, measured in the reverse order before the ready-tag
correction, improved these four lengths by 0.96–1.19%. These are two complete
pairs, not a large-sample estimate. The corrected pair's weighted 200K decode
rate is 168.659 to 168.516 tok/s (-0.08%). Pi small prefill is 797.49 to
799.24 tok/s (+0.22%) and small decode is 232.752 to 232.506 tok/s (-0.11%).
The 256–511-token prefill bucket falls 1.70%; there are no 1–15-token requests.
Total frozen Pi API time is 73.771 to 73.488 seconds (-0.38%), excluding tools
and agent work. There is no demonstrated decode or small-increment speedup,
and this comparison does not establish a further 5% task-completion gain.

All seven long and 23 Pi requests match in text, reasoning, tool calls, finish
reason, request hashes, native generation counts and cache/prompt/draft counters.
Neither arm reports an HTTP or engine error. The disabled arm also matches the
previously selected QSA executable. The complete HIP build and 342 exact GPU
checks pass, as do six bounded-error checks. The old-protocol negative control
fails as expected. CUDA, SYCL and other GPUs were not tested.

A 261888-token request returns 1517; its continuation returns 1683 and reuses
261892 tokens. No OOM occurs, with about 13 MiB free per card. This near-limit
check uses short replies; the 200K throughput panel separately generates up
to 2048 tokens. Cancelling a cold 200K request after the first 32768-row pass
allows the same engine to answer 323 and reproduce the frozen 32K output.
The short recovery request takes 7.40 seconds while queued work drains, so
this does not establish immediate cancellation.

The local configuration now selects the corrected executable and HC flag,
retaining the QSA and K=320 options, both GPUs, 256K context and 512 MiB host
cache. Ordinary and thinking service checks return 323 and 1683. The original
5500 tok/s prefill and 225.8145 tok/s weighted decode targets remain unmet.
Build logs, captures, GPU tests, comparisons and rollback configuration are in
`local-tuning/2026-11-06/ud31-residual-opt/`, including
`hc-row-selected-qualification.json`, `hc-row-clear-performance-summary.json`,
`hc-row-clear-capacity-pipeline.json` and `hc-row-split-ready-clear-audit.json`.
Private configurations contain credentials and must not be published.


## Rejected decode QSA key partition

Two isolated prototypes were measured on November 7 on the same UD model,
two RX 7900 XTX cards and 256K configuration. Each card scores only its owned
pooled keys, keeps its exact local top 512, exchanges the original score bits
and global IDs, then selects the global top 512. The per-key FP32 arithmetic
and chronological tie order stay unchanged. Main-model decode at 192K or more
uses the split; MTP, batched lanes and prefill retain their existing path.
Each card adds 192 KiB after its ordinary allocations.

Both standalone versions pass 10368 GPU comparisons, 1296 independent CPU
top-K checks and six bounded-error checks. These cover ties, partial pools,
non-contiguous page maps, graph replay, delayed peers and generation wrap.
The second version also replays one graph across 192K in both directions.
At 200K, its cold score/selection/communication time falls from
106.079 to 93.360 microseconds for one row and 117.799 to 102.560 for three.
These component improvements did not carry through to model throughput.

| Same-binary comparison | Weighted 200K decode off → on | Frozen Pi API time off → on |
| --- | ---: | ---: |
| Host prefix dispatch, off then on | 168.501 → 166.919 tok/s | 73.416 → 73.766 s |
| GPU prefix dispatch, on then off | 168.503 → 167.540 tok/s | 73.365 → 73.501 s |

Each arm runs seven long requests followed by 23 Pi requests. All outputs,
request hashes and generation/cache/draft counters match; no HTTP error is
reported. The host version changes a shared graph key, which also invalidates
MTP graphs at the threshold. Its capture overhead was not separately measured.
The GPU version keeps the original host graph key and adds an empty merge
launch below the threshold. It still loses 0.57% weighted long decode; Pi
small-prefill and small-decode rates fall 0.47% and 0.37%. Its prefill rates
change by less than 0.14% at the four fixed lengths.

Neither prototype is selected. Formal source and the daily configuration
remain unchanged. Capacity, cancellation, live-task acceptance and other
backends were not tested for these rejected versions. The daily service was
restored and passed ordinary and thinking output checks. Records are
`qsa-key-host-decision.json` and `qsa-key-dynamic-decision.json` in
`local-tuning/2026-11-06/ud31-residual-opt/`. This experiment establishes no
additional complete-task improvement.


## Screened IQ2 codebook storage

An isolated HIP probe on November 7 compares the selected wide/fused gate/up
kernel with two versions that stage its codebook in local shared memory.
One keeps each 64-bit entry. The other stores eight exact high nibbles in
32 bits and restores the original byte values before the existing FP16
conversion. WMMA, split-K and SwiGLU arithmetic stay unchanged.

The probe passes 5792 exact GPU comparisons and checks all 2560 codebook
entries across both RX 7900 XTX cards. It covers both IQ2_S and IQ2_XXS,
partial output tiles, empty and partial jobs, split-K, trimmed token tiles,
shuffled expert pointers, untouched output bytes and graph replay. Sixty-four
three-way timing cases compare global, raw local and compressed local tables;
K=2560 and 640 stacked gate/up output rows match the model's expert shape.

For 512 cold IQ2_S jobs with eight token rows each, raw local storage reduces
the fused kernel time by 6.55–6.84%. With 64 rows per job it increases time by
0.14–0.75%; mixed 16/32/48/64-row jobs improve only 0.22–0.97%. IQ2_XXS raw
local storage is 1.98–3.52% slower in those cold 512-job cases, and compressed
storage is 7.72–15.85% slower. Eight rows per expert job does not mean an
eight-token request. These are synthetic component timings, not Pi throughput.

The broad change is not selected: it provides no substantial long-prefill
gain. A narrower IQ2_S thin-job version would need separate request-level
measurement. No model integration, capacity, cancellation or live-task checks
were run for these candidates. Formal source and daily configuration remain
unchanged. The record is `iq2-lds-decision.json` in
`local-tuning/2026-11-06/ud31-residual-opt/`.

## Short incremental GSU3 projections

Measured on November 9 with Unsloth UD IQ2_XS, two RX 7900 XTX cards and
262144-token capacity. `STRATA_PF_SHORT_GSU3=1` extends the existing eligible
HC-down and dense GSU3 products to 2–15 input rows. It retains their three
ordered FP32 partial sums and the original FP16 weight rounding. The existing
HC-down, wide and Q8 GSU3 options must also be enabled. This option defaults
off and checks for gfx1100, with the device result cached in the process.

One-row products retain rocBLAS: their reference sum order differs. The
K=320 short-row experiment also differs and is excluded. The selected medium
K=320 option keeps its previous range. No weights, context limit, generation
budget or sampling distribution change.

Conversation checkpoints can leave a short assistant-prefix segment at the
end of a read. The thinking Pi traces include four-row tails. Accelerating
these segments helps requests whose total new input is larger than 15 tokens.

The formal binary has SHA-256
`4ac155a41109fcf346826e28c70a72b16ea817fd7c6cabccf5328887fcdc70ba`.
Two repetitions of the matched 17-request panel use a 72539-token prefix and
3994 total fresh tokens. Each reply is the single token `READY`; cold seed
requests are excluded. Requests, outputs and generation/cache/draft counts
match with the option disabled and enabled. Times below sum engine prefill
latency, including its checkpoint work.

| Fresh input | Requests | Disabled | Enabled | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| All matched increments | 34 | 9.4942 s | 7.4775 s | 21.24% |
| Below 32 tokens | 20 | 4.4873 s | 3.1396 s | 30.03% |
| 256–511 tokens | 8 | 3.2727 s | 2.9031 s | 11.29% |

The GPU regression passes 3344 checks with the option disabled and 4632
enabled. These include comparisons with rocBLAS over 2–63 rows, mixed
magnitudes, Q8 scales with signed zeros and subnormals, head slices, padding,
fallback guards, and the existing K=320 graph cases. Three seeded model
requests also match the previously selected executable with the option off,
and match the enabled run in output and counters. Other GPU architectures and
backends were not tested for this option.

The formal build also passes a 261888-token request and a cached continuation,
returning `1517` and `1683`. The continuation reuses 261892 tokens. A separate
long request is cancelled after its first 32768-token pass. Ordinary and
thinking requests then return `323` and `1683` on the same engine process.
The local daily config now selects this binary and enables the new option;
its other settings are retained. The daily 8081 service was stopped before
this tuning run and remains stopped. Validation uses the isolated 18081 service.

The separate complete-task trial uses the isolated prototype
`f517dd7453f51be3f3406e7008bcb53b60b6e6b20766bea21755482d7ddc63c1`.
The formal version adds the cached gfx1100 gate and excludes the failed K=320
experiment. Each arm completes the same four live Pi tasks with all attempts,
tools and independent acceptance counted. Each pair shares its sampling seed;
temperature, top-p, top-k, medium thinking, 32768-token output ceiling, context,
task text and workspace paths stay fixed. Tool outputs retain their actual
timings. There are no separate warmup requests.

| Seed and order | Disabled | Enabled | Task-time reduction |
| --- | ---: | ---: | ---: |
| 101, disabled first | 233.235 s | 221.386 s | 5.08% |
| 202, enabled first | 106.643 s | 110.611 s | -3.72% |
| Combined | 339.878 s | 331.997 s | 2.32% |

All 16 task acceptances pass on the first attempt, with zero HTTP errors.
Generated output differs: 61819 versus 60701 tokens. The supplied Pi export
contains request metadata without the original task text; these fixtures
represent its request pattern. These two pairs do not establish a stable 5%
complete-task gain. At this stage, that goal remained open; the selected
increment was justified by the repeated matched-input prefill gain. The later
lookup comparison below records the next complete-task trial.

Evidence is in `local-tuning/2026-11-09/ud31-pi-task-opt/`:
`short-formal-check-pipeline.json`, `short-gsu3-gpu-regressions.json` and
`short-gsu3-task-pipeline.json`. Capacity, recovery and application are recorded
in `short-gsu3-qualification.json` and `short-gsu3-deployment.json`.
The earlier cache-hit staging prototype passes
320 exact GPU checks and preserves seeded model output, but copying 11.67%
fewer bytes does not establish a speed gain: matched prefill time increases
0.79%. It is excluded. The earlier fixed-three-draft trial changes complete
task time by only 0.63% and also remains excluded.

## Screened decode indexer and QKV overlap

An isolated November 7 probe schedules the original BF16 indexer projection,
pooling, scoring and selection alongside the original Q8S QKV projection and
Q/K normalization. It compares serial execution with three schedules: indexer
on a second stream, selection alone on that stream, or QKV on that stream.
The fork and join events are included in the timing. KV append, attention and
whole-engine cross-card scheduling are outside this synthetic screen.

All 8064 exact output and guard comparisons pass across both RX 7900 XTX cards,
packed and split activations, 1–16 rows, seven prefix lengths, and three graph
replays. Seventy-two four-way timing cases include warm and flushed caches.
Warm-cache improvements of about 5–15% do not persist with cold weights. At
200K and one row, serial GPU time is 159.882/167.082 microseconds on the two
cards; moving the whole indexer to the second stream gives 159.682/167.482.
At three rows those times are 175.962/182.482 versus 173.162/183.282. The other
two schedules also provide no consistent cold-200K gain across both cards.

No model integration is selected. These results establish no request-level
throughput or complete-task improvement. Formal source and daily configuration
remain unchanged. The record is `qsa-qkv-overlap-decision.json` in
`local-tuning/2026-11-06/ud31-residual-opt/`.

## Pi lookup and exact sampling kernels

Measured on November 9 with Unsloth UD IQ2_XS, two RX 7900 XTX cards,
262144-token capacity, six MTP drafts, a 15-token lookup limit and a
512 MiB conversation cache. Three new controls default off:

| Control | Default | Measured value | Purpose |
| --- | ---: | ---: | --- |
| `FNX_LOOKUP_SEARCH` | 0 | 32 | Search more matching context trigrams for a usable draft continuation |
| `FNX_TOPK_PARTS` | 0 | 16 | Find exact sampling candidates in smaller vocabulary partitions |
| `FNX_QSA_SCORE_GRID` | 0 | 128 | Reuse each indexer query across several key tiles |

The lookup reference stops at the nearest matching trigram, even when its
next token differs from the already sampled first draft. A nearby overlapping
match can also offer only one or two known continuation tokens. The new option
examines up to the configured number of matching trigrams, chooses the longest
usable continuation, and keeps the most recent occurrence on ties. The first
token must still match the sampled draft. Copies stop before image cells.
Values from 0 to 64 are accepted; 0 retains the original rule.

The target sampler, penalties and speculative rejection calculation are
unchanged. Changing the proposal chain changes random-number consumption,
so a fixed seed can produce different text when lookup search is enabled.
This option does not promise the same text as the nearest-match rule.

Partitioned top-k retains the original FP32 penalty expressions, score bits
and token tie order. It merges the best 64 candidates from each partition.
Supported partition counts are 2, 4, 8, 16 and 32; this path requires two-card
HIP Flash-Next on gfx1100. At 16 partitions, scratch uses 128 KiB per GPU
compute stream, is allocated on first eager use and is released with the
engine. A fresh stream inside graph capture falls back to the original kernel
until scratch is available. Small vocabulary slices also keep that fallback.

The local indexer score grid preserves each key's FP32 FMA and head-sum order
and the existing selector. Queries that need ranking still score every visible
key. The control accepts 0 through 1024; a positive value sets the target
number of score blocks across the active query rows. It adds no peer exchange
or GPU buffer. This is separate from the earlier rejected query-ownership probe.

The complete HIP build has SHA-256
`a1e3e8417002394ca6a6c27b2c0cde142cd77c98840d8b3704de11f7e7d08a7b`.
The ordered four-request diagnostic uses the same captured inputs, seeds and
1024-token API ceiling in each arm. With short GSU3 enabled throughout,
weighted native decode throughput is 241.01 tok/s with the three controls
disabled and 266.17 enabled, a 10.44% increase. Both arms generate 4062 native
tokens. Lookup changes the sampled text, so this is a request diagnostic,
not a measurement of a 10.44% gain over complete Pi tasks.

Both disabled repetitions match the earlier selected executable in text,
reasoning, tools, finish reasons and native counters. The enabled formal
build matches the combined prototype on those checks. The combined prototype
also matches lookup search alone, confirming that the two numerical kernel
changes preserve that proposal policy's output on this sequence.

The CPU suite passes 36002 compatibility and proposal checks. The two cards
together pass 2016 exact top-k cases in each of two runs, with the dispatcher
disabled and enabled, plus 80 total allocation, release and capture checks.
The local score-grid suite passes 2520 exact score, selected-position, count,
guard and graph checks. The complete HIP build passes. CUDA builds and other
GPU architectures were not tested; the split engine is not a SYCL target.

A cold 261888-token prompt and a sampled cached continuation return `1517`
and `1683`; the continuation reuses 261892 tokens. A separate 200K prefill is
cancelled after its first 32768-row pass. Ordinary and thinking requests then
succeed on the same engine process. All seven installed-Pi provider checks
pass, including tool calls, a long cached follow-up and repeated-thinking
recovery. These checks establish capacity and recovery, not retrieval quality
or long-generation throughput near the context limit.

Evidence is in `local-tuning/2026-11-09/ud31-pi-task-opt/`:
`lookup-search-screen.json`, `formal-pi-build.json`,
`formal-pi-kernel-checks.json`, `formal-pi-model-checks.json` and
`formal-pi-qualification.json`. The complete-task comparison is recorded in
[UD31_PI_THROUGHPUT.md](UD31_PI_THROUGHPUT.md).
