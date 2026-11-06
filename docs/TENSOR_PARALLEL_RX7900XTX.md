# Dual RX 7900 XTX decode experiments

Implemented on `tensor-parralize`, based on main
`ffa80d73133ea5c548aefe57889a16e8085a938f`. The defined single-request panel on
Swift-1.5 IQ3_S now reaches **222.38 tokens/s** across two independent engine
runs (**222.15** and **222.61**), meeting the follow-up 220 target. Each run
contains code, prose and edit prompts, 2048 output tokens each, repeated twice,
with context 32768 and warmup/prefill excluded. This is a weighted panel result;
individual prose prompts remain around 198 tokens/s. The original 200.88-token/s
checkpoint and its binary remain available below.

Main already contains tensor parallelism in `src/split`. These changes make
that engine usable with this model on two 24 GiB cards and a 64 GiB host, then
measure its decode bottlenecks. The deployed native engine and its config are
not replaced. All new execution paths are opt-in.

## Memory and transfer changes

- `FNX_STATIC_EXPERTS=1` gives hot experts compact VRAM storage and cold
  experts mapped host storage. Hot experts have no duplicate pinned host copy.
  The original cache pins the whole expert set, which caused heavy swapping
  on this host before a baseline could finish loading.
- Static placement disables expert eviction and the KV arena. It supports a
  plain model load, not the holder/worker path. An explicit `FNX_ARENA=1` is
  rejected. `FNX_EXPERT_RAM_GIB` and `FNX_EXPERT_PROFILE` select the cold set.
- `FNX_STATIC_CACHE_SLOTS=N` adds an optional decode cache for cold experts.
  It requires static placement and one lane; zero disables it (the default).
  Accepted nonzero values are multiples of 32 from 160 to 4096. There must be
  room for at least every possible job in a 16-row window. Each card uses one
  CLOCK pool with the maximum gate/up and down strides; transfers copy only
  the actual layer bytes. Fixed hot experts never take or reference a slot.
  Decode owns separate pointer tables, so prefill keeps the immutable model
  placement and needs no additional staging allocation. The cache persists
  across requests. Its VRAM allocation is additional to the fixed hot weights.
- `FNX_STATIC_CACHE_GROUPS=2` divides the cold cache into two pools, using
  the lower and upper halves of distinct expert byte sizes. Each pool retains
  enough slots for a complete decode window; extra slots are assigned in
  32-slot groups in proportion to the number of layers sharing the pool.
  Independent gate/up and down maxima determine each pool's stride. Total
  allocation cannot exceed the one-pool byte budget, and slot count is capped
  at twice the requested count. The default is one pool; a budget too small
  for two complete windows is rejected before allocation. This is a heuristic
  for these weights, not a universal cache-size recommendation.
- `FNX_MOE_STAGE=1` copies each selected expert matrix into reusable VRAM
  scratch before its quantized dot products. Gate/up and down reuse the same
  scratch in stream order. Copies preserve every weight byte. The normal
  `DW::ep` interface is used by the compute kernels.
- `FNX_DECODE_TIMING=1` reports the existing verify, draft and launch timers.
  `FNX_PROFILE_EXIT=1` lets profiler exit callbacks write their traces. It is a
  diagnostic option; normal operation keeps the existing fast exit.
- `FNX_IQ_REUSE=1` decodes an IQ2_S, IQ3_XXS or IQ3_S weight group once for
  all token rows in an expert job. Its integer divisions, scaling and reduction
  order match the existing Q8 dot product. Other weight types keep their path.
- `FNX_IQ2_XXS_REUSE=1`, with IQ reuse, also reuses IQ2_XXS gate/up weights.
  It preserves the original integer scaling and reduction order. With group 16,
  it supports the fused cold gate/up and down copies below. Zero keeps the
  original IQ2_XXS path and remains the default.
  HIP's existing `FNX_PF_MOE_WIDE=1` prompt path also uses packed FP16 pairs
  for IQ2_XXS dequantization, matching the original tile values exactly.
- `FNX_IQ_GROUP=16`, with IQ reuse, assigns 16 lanes to each output row.
  Each lane keeps two partial sums, then joins them in the original 32-lane
  reduction order. The default remains 32 lanes.
- On HIP, the IQ reuse path uses packed byte sign masks and subtraction from
  the existing compatibility layer. IQ grid values cannot saturate signed bytes,
  so this preserves the original values while avoiding the ggml bytewise
  implementation. CUDA keeps the original operations.
- `FNX_MOE_PRE_ROWS=1` computes the routed down activations once for every
  decode window, using the existing precompute kernel. The default threshold
  remains 9 rows; values 1..17 are accepted (17 disables it).
- `FNX_MOE_STAGE_COLD_ONLY=1`, with static placement and staging enabled,
  copies only host-resident experts. Hot experts keep their VRAM pointers.
  The main layer index is now passed explicitly to the FFN helper; MTP uses
  -1 and does not index the main model's residency table.
- `FNX_STATIC_HOST_CACHE=1` uses HIP's non-coherent pinned allocation for the
  immutable cold expert banks, allowing GPU caching. CPU writes finish during
  load, before the GPU commands read the banks; inference never modifies them.
  It does not change the allocation of communication buffers. This is HIP-only;
  the flag is not combined with `hipHostMallocMapped`, which would override it.
  See [HIP host-memory coherence](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.0/how-to/programming_manual.html#host-memory).
- `FNX_EMBED_WARM_AFTER_LOAD=1` refreshes the host embedding mappings after
  bulk expert reads, which can evict unlocked pages. It reports page residency
  before and after the refresh. This moves work to startup; it does not change
  the request, output or model arithmetic.
- `FNX_RELEASE_COPIED_WEIGHTS=1` releases owned GGUF conversion buffers after
  static plain loading has copied weights to each card and cold expert bank.
  Token/PLE embeddings and the output weight needed by MTP remain available.
  Metadata remains, but released tensors have null data and cannot be staged
  again without reloading. Staging scratch is destroyed before the optional
  embedding refresh, regardless of this flag.
- `FNX_KV_CONTEXT_ONLY=1` sizes the plain int8 KV pool for the requested
  context plus the existing page and lane margins. The default still expands
  the int8 pool to the FP16 byte budget. This option leaves the logical context
  limit and KV quantization unchanged and rejects the shared arena. At context
  32768 it saves about 195 MiB per card, making room for more fixed hot experts.
- `FNX_HC_ROW_GROUP=64` selects a 512-thread HC layout for windows of 1–12
  tokens on gfx1100. Larger windows retain the original 768-thread layout.
  The grids have separate barrier counters, including during graph replay;
  switching window size does not reset a counter or add a GPU launch. Startup
  checks residency across the selected compact kernels and caps their grid at
  80 blocks. The default remains 96-row groups. HIP HC timeouts use the device's
  wall-clock rate, since the previous cycle counter is unreliable on RDNA3.
- `FNX_Q8S_ROW_GROUP=2` gives each warp two output rows for Q8 split-layout
  matrices with at least 8192 rows, including segmented attention projections.
  Smaller projections keep their original dispatch. Zero (the default) keeps
  all original choices; 1, 2 and 4 are accepted overrides. Integer operations,
  scaling and each row's reduction order are unchanged.
- `FNX_GDN_REPLAY=1` replaces the recurrent layers' fifteen full rollback
  snapshots with the first state and the actual rank-one updates for later
  tokens. Partial acceptance restores that state and replays only the kept
  updates. The logged keys, deltas and decay retain the original gfx1100
  multiplication/FMA order; the default full-snapshot path is unchanged.
  This option requires the Flash-Next HIP build, gfx1100, one lane and
  `FNX_BATCH_DRAFTS=0`. Single-request MTP remains enabled. Batched multirow
  verification is rejected before using the smaller buffers. Prefill can
  borrow the allocation as before and may allocate additional scratch if
  needed. At the tested prefill size, net savings are 744 MiB per card.

`FNX_STATIC_CACHE_FUSED_DN=16` fills missing down weights inside the IQ gate/up
launch: 16 prefix blocks copy bytes while the remaining blocks compute the
already resident gate/up matrices. The preceding cache-copy launch fills only
gate/up, and kernel completion orders down reads. There is no additional stream,
event or grid barrier. This requires the static cache and 16-lane IQ reuse;
other expert types, prefill and the MTP layer keep their existing paths. Zero
(the default) disables it; 1–256 copy blocks are accepted. All original integer
operations, scaling and floating-point reduction order are retained.

`FNX_PF_ROWS=256` reduces prefill scratch compared with the existing default
4096. This allowed the cold allocation to fall from 16 to 14 GiB. With staging,
a 32768 context and MTP, only about 288 MiB remained free per card at startup.
This setting has not been validated for the configured 204800 context.

On this host, `HSA_FORCE_FINE_GRAIN_PCIE=1` allowed both directions of the
existing peer probe to pass. The runtime still checks the actual links before
using peer push. The startup log now reports host exchange only when it is
actually selected.

The existing `FNX_MTP_EXPERTS=q4_0` path can use draft experts quantized from
the installed BF16 tensors. `tools/mtp_experts.py` now prepares a separate
runtime directory, checking the input manifest hashes and Q4 bytes against the
pinned GGUF reference. Dense draft weights and the complete draft vocabulary
are reused. The target Swift IQ3_S weights are unchanged. Both draft formats
use Q4 blocks on the GPU: the default Q2 pack is expanded when loading, so the
Q4 experiment does not enlarge the expert allocations.

`FNX_MTP_HEAD_Q4=1` separately converts the draft output head's Q8 rows to
Q4_0 at load time. It keeps every draft vocabulary row, in the same order,
and leaves the main output head unchanged. The default is the original Q8
head. The Flash-Next HIP Q4 reader uses eight lanes per output row for large
matrices. This changes draft predictions and can change acceptance; it is
not an exact replacement for the draft head's arithmetic. On these cards,
each draft head shrinks from 137.87 to 72.99 MiB.

`FNX_MTP_LAST_HEAD=1` computes only the last output-head row in a single-request
MTP window. The earlier rows still update the complete draft hidden state;
only their unused vocabulary projections and argmax work are skipped. The
last row stays in its original buffers for penalties, sampling and host
readback. The option requires a Q4 draft head, defaults to zero, and leaves
the separate batched-lane path unchanged.

## Measurements so far

Hardware: two RX 7900 XTX (`gfx1100`, 24 GiB each), Ryzen 7700, 64 GiB RAM.
Model: the installed Swift IQ3_S GGUF, with its existing MTP pack and draft
vocabulary. These are short prompts, greedy generation, thinking disabled,
one request at a time, 2048 output tokens per task, max context 32768.

The native comparison uses the existing server config with an explicit context
override in the benchmark child. Its production config is unchanged. Split
uses MTP limit 6, lookup limit 15 and no conversation cache. Neither engine
uses a forced continuation.

| Engine / setting | Code | Prose | Edit |
| --- | ---: | ---: | ---: |
| Native engine, existing tuning | 137.1 | 100.3 | 154.8 |
| Native engine, same tuning, separate two-run comparison | 133.3–134.9 | 103.4–104.3 | 153.4–157.6 |
| Split, static 16 GiB cold, peer push | 71.9–77.0 | 69.2–75.1 | 70.1–72.4 |
| Same, with matrix staging | 99.1 | 92.3 | 100.8 |
| Staging, 14 GiB cold, prefill rows 256 | 109.0 | 99.6 | 110.8 |
| Same, calibrated decode residency, two runs | 129.6–134.3 | 113.7–115.8 | 136.1–137.8 |
| Same, IQ weight reuse, two runs | 130.0–141.4 | 110.2–120.1 | 139.3–145.2 |
| IQ reuse plus cold-only staging, two runs | 133.7–149.1 | 111.0–124.9 | 142.6–150.5 |
| Same, embedding refresh after load, two runs | 148.1–149.0 | 123.7–125.0 | 149.6–150.4 |
| Cached host weights, direct reads, embedding refresh, two runs | 155.7–156.1 | 131.5–131.6 | 158.3–158.7 |
| Same, 16-lane IQ groups, two runs | 156.8–158.7 | 133.3–133.5 | 161.4 |
| Same, precompute down activations from one row, two runs | 158.1–161.1 | 134.3–134.4 | 163.1–163.3 |
| Same, packed IQ sign operations, two runs | 163.2–167.1 | 137.0–138.8 | 167.9–169.0 |
| Same, 256-slot cold cache, 14 GiB cold, two runs | 169.9–171.2 | 149.6–150.0 | 174.5–174.8 |
| Same, 1024-slot cold cache, 16 GiB cold, two runs | 174.4–177.9 | 153.2–155.7 | 181.8–183.6 |
| Same, release copied GGUF conversions, two runs | 177.7–178.0 | 156.1–156.2 | 183.6–183.8 |
| Same, compact HC for 1–12 rows, two runs | 179.7–180.3 | 157.9–158.3 | 186.2–186.4 |
| Same, fused down-weight copy in IQ gate/up, two runs | 181.9–182.2 | 158.9–159.1 | 188.8–189.0 |
| Same, two cache pools within the existing byte budget, two runs | 184.2–184.5 | 160.6–160.7 | 192.9–193.2 |
| Same, context-sized KV pool and 15.5 GiB cold backing, two runs | 186.5–186.8 | 161.5–162.0 | 196.0–196.3 |
| Same, two-row groups for large Q8 projections, two runs | 189.0–189.7 | 164.7–165.0 | 198.6–198.7 |
| Same, BF16-derived Q4 draft experts, two runs | 196.0–196.4 | 167.5–167.7 | 200.7–201.3 |
| Same, logged GDN rollback and 14 GiB cold backing, two runs | 205.8–206.2 | 173.3–173.7 | 212.6–213.2 |
| Same, Q4 draft head and 13.875 GiB cold backing, two runs | 205.3–205.8 | 177.9–178.5 | 214.0–214.4 |
| Same, 32 copy blocks and last-row draft head, four passes in two independent runs | 207.1–208.4 | 179.6–180.0 | 216.7–217.4 |
| Same, 16 copy blocks, four passes in two independent runs | 208.1–209.0 | 179.9–180.4 | 217.8–217.9 |

Values are tokens/s from the engine's reported decode time. The benchmark
also records client time, time to first token, exact input/output token IDs,
draft acceptance, binary SHA-256, source diff fingerprint and environment.
Warmup is excluded. The calibrated split's weighted rate is 127.13 tokens/s;
with IQ reuse it is 129.76 tokens/s. This is a small end-to-end gain and needs
more runs to separate it from run-to-run variation. It does not establish a
general advantage over the native engine.
Cold-only staging with IQ reuse measured 133.75 tokens/s across its two runs.
A separate one-pass run measured 144.1 / 120.7 / 147.1 (weighted 136.22).
These differences between runs are material: for example, the code task's
verify GPU time was 12.24 s on the first pass and 12.16 s on the second, while
verify wall time fell from 13.78 to 12.21 s. The unlocked PLE table and host-side
first-use work need separate investigation. Both passes remain in the results.
Refreshing the embeddings measured 139.93 tokens/s across two passes. At the
end of bulk loading, the PLE mapping had 25914 of 27465 MiB resident; the refresh
restored almost all pages in 0.76 s. The first-pass code task then took 12.29 s
for verify, of which 12.16 s was GPU time. This supports page residency as one
source of the earlier first-pass overhead; it is not a GPU kernel speedup.

Cached host weights with decode staging disabled measured **147.60 tokens/s**
weighted across two passes. All six output token sequences match the reference.
Keeping cold-only staging with these same cache/refresh flags measured
148.7 / 124.6 / 150.0 tokens/s (weighted 140.07) in a separate pass, also with
identical token IDs. The reproduction recipe therefore uses direct reads.
The same short prompts' prefill times fell from about 590 / 702 / 888 ms to
200 / 208 / 285 ms on the second pass. These are measurements of this workload,
not a general prefill throughput claim. At this step the 200 tokens/s target was still open.
The separate two-pass native comparison measured 127.53 tokens/s weighted,
versus 147.60 for the cached split (15.7% higher for these tasks). Native output
and draft acceptance can vary across repeated greedy requests; split/native
arithmetic differs as described below, so this is not an equal-token comparison.
The 16-lane IQ groups measured 149.77 tokens/s weighted, a further 1.5% gain.
All six sequences match the cached 32-lane reference. The full engine gain is
much smaller than the roughly 15% kernel improvement at 24 expert jobs.
Precomputing down activations from one row measured 151.24 tokens/s weighted;
again all six output sequences match the reference. Its representative
six-row/24-job kernel pair took about 27 us versus 41–44 us for fused IQ4_NL
and 23–24 us versus 36–38 us for Q4_0. Kernel savings do not translate directly
to the full model because other work overlaps and remains on the critical path.
Packed IQ sign operations measured **155.87 tokens/s** weighted across two
passes, 3.1% above the down-precompute version. All six output sequences match.
This is 22.2% above the 127.53 tokens/s two-pass native comparison for these
tasks; the different-arithmetic caveat above still applies. The target remains 200 tokens/s.

A 256-slot cold cache (325 MiB per card) measured 164.27 tokens/s weighted,
5.4% above direct cached host reads. Increasing the cache to 1024 slots
(1300 MiB per card), with 16 GiB of cold backing to make room, measured
**170.19 tokens/s** weighted. All twelve output sequences from these two
configurations match the direct-read reference. The 1024-slot setting is
33.4% above the native comparison on these tasks, with the same arithmetic
caveat. It left 144 MiB free per card after startup/warmup; the 256-slot
setting left 102 MiB. The cache persists across requests, and both first and
second passes are retained. First-pass embedding page faults remain visible
in the 16 GiB run, so neither this nor the earlier trials are latency guarantees.

Releasing copied conversion buffers freed **7723 MiB** after loading. The
PLE refresh restored all 27465 MiB, and all six measured requests recorded zero
major faults during decode. This run measured **171.73 tokens/s** weighted,
with every output token identical to the cache reference. A separate recent
control measured 171.32 tokens/s, so the result primarily establishes reduced
memory pressure; it does not establish a GPU arithmetic speedup. A mid-run
RSS sample was 57092372 KiB. See `bench-release-owned` for the complete run.

Compact HC for 1–12 rows measured **173.92 tokens/s**, against **171.64** with
the flag off in the same binary (1.33% higher). All six measured output token
sequences match and decode major faults were zero. See `bench-hc-hybrid` and
`bench-hc-hybrid-control`. The complete-code tasks also matched their reviewed
token IDs and passed all 1326 checks. The 494-token varint task measured
203.95 tokens/s; at this step the broader 2048-token panel was still below 200.

Fusing down-weight copies into IQ gate/up measured **175.68 tokens/s** with
64 copy blocks, versus **173.80** with fusion disabled in the same binary
(**1.08%** higher). Every measured token matched and all six decode runs had
zero major faults. See `bench-cache-fused64` and `bench-cache-fused-control`.
The three complete-code tasks matched their reviewed token IDs and passed all
1326 checks (`quality-cache-fused64`); varint measured 206.22 tokens/s. The
broader panel still does not meet the target. Using the existing
`FNX_RING_GRID=80` alongside fusion measured 176.27 tokens/s with identical
outputs, only 0.33% above the default grid. This small difference is not treated
as an established improvement and the reproduction recipe keeps the default.
Increasing the fused copy prefix to 128 blocks measured 175.30 tokens/s,
with every token matching. It did not improve over 64 blocks, which remains
the recommended setting (`bench-cache-fused128`).

Two cache pools measured **178.27 tokens/s**, against **175.81** with one
pool in the same binary (**1.40%** higher). The requested 1024-slot byte budget
now holds 896 smaller and 288 larger slots (1184 total), using 1295 MiB per card
versus 1300 MiB. The smaller pool serves 37 layers with 627200 gate/up and
460800 down bytes per slot; the larger serves 11 layers with 870400 gate/up
and 460800 down bytes. Startup/warmup left 146 MiB free on each card. All six
output token sequences match and decode major faults are zero. The reviewed
code tasks also match and pass all 1326 checks; varint measured 208.30 tokens/s.
See `bench-cache-groups2`, `bench-cache-groups-control`, and
`quality-cache-groups2`. The broader 200 tokens/s target was still open at this step.

Sizing the KV pool to the requested context, then reducing cold backing to
15.5 GiB, measured **180.30 tokens/s** versus **178.37** with the preceding
recipe in the same binary (**1.08%** higher). The pool holds 172 pages (33024
positions), compared with 307 pages before. Startup/warmup left 86 MiB free
per card. All six output sequences match and decode major faults are zero;
the three reviewed code tasks also match and pass all 1326 checks. See
`bench-kv-context-ram15p5`, `bench-kv-context-control`, and
`quality-kv-context-ram15p5`.

A separate capacity check used an actual 32688-token input followed by 64
generated tokens, with a 32768 logical limit. Both recipes completed with
identical input and output token IDs, without decode major faults. The new
recipe took 47.59 s to prefill and 315.3 ms to decode; the control took 48.57 s
and 317.8 ms. This short list-generation check verifies operation near the
context boundary; its 202.98 tokens/s is not the acceptance panel result.
See `near-context-kv-context`, `near-context-kv-control`, and
`kv-context-checks.json`. The configured 204800 context remains untested.

Two-row Q8 groups measured **183.14 tokens/s**, against **180.71** with the
original dispatch in the same binary (**1.34%** higher). All six output token
sequences match and decode major faults are zero. See `bench-q8-group2` and
`bench-q8-group-control`. These gains are specific to the tested shapes and
hardware; the option does not change defaults on other backends.
The reviewed code tasks also retain their exact token IDs and pass all 1326
checks (`quality-q8-group2`); the varint task measured 212.60 tokens/s.

Q4 draft experts measured **187.03 tokens/s**, versus **183.14** with Q2 draft
experts in the same binary (**2.12%** higher). Accepted draft tokens increased
from 1536 to 1565 on code, 1350 to 1369 on prose, and 1593 to 1608 on edit.
All six target output sequences match and decode major faults are zero.
The completed-code tasks also match and pass all 1326 checks; varint measured
218.17 tokens/s. Two additional Chinese prompts, generating 1024 tokens each,
match the Q2 reference exactly: systems explanation measured 152.72 versus
151.73 tokens/s, and Unicode explanation/code 187.08 versus 181.74. These
held-out checks were not used to select expert residency. See `bench-mtp-q4`,
`quality-mtp-q4`, `heldout-mtp-q2`, `heldout-mtp-q4`, and `mtp-q4-conversion.json`.
Raising the Q4 draft limit from six to eight measured 186.70 tokens/s, with
identical target outputs; the recipe retains six (`bench-mtp-q4-spec8`). The
broader 200 tokens/s target was still open at this step.

Logged GDN rollback with the preceding 15.5 GiB cold backing measured
**189.85 tokens/s**. Startup/warmup left 828 MiB free per card. Moving another
1.5 GiB of expert weights from host memory to the two cards (14 GiB cold
backing) measured **195.89 tokens/s**, versus **186.89** with full snapshots
and 15.5 GiB backing in the same binary (**4.82%** higher). Both comparison
runs disable batched draft windows. All input/output token IDs match the Q4
reference; all six measured requests have zero decode major faults. The
14 GiB recipe leaves 68 MiB free per card. See `bench-gdn-replay-ram15p5`,
`bench-gdn-replay-ram14`, and `bench-gdn-replay-control`.
The three reviewed code tasks retain their token IDs and pass all 1326 checks
(`quality-gdn-replay`); varint measured 228.30 tokens/s. At this step the full panel was still
below 200 tokens/s.
A real 32688-token input plus 64 generated tokens also matches the same-binary
full-snapshot control exactly: prefill/decode took 45.66 s/271.0 ms, versus
47.73 s/285.6 ms. Both additional Chinese tasks retain every Q4-reference
token. These checks have zero decode major faults; see `near-context-gdn-replay`,
`near-context-gdn-control`, `heldout-gdn-replay`, and `gdn-replay-checks.json`.

The Q4 draft head, using the freed memory for additional fixed hot experts,
measured **198.09 tokens/s** versus **195.78** with the preceding Q8-head
recipe in the same binary (**1.18%** higher). All six target output sequences
match and decode major faults are zero. Accepted draft counts changed from
1565/1369/1608 to 1559/1391/1606 for code/prose/edit; the gain is not uniform
across tasks. Startup/warmup left 64 MiB free on each card. See
`bench-mtp-head-q4-ram13p875` and `bench-mtp-head-q4-control`. The reviewed
code tasks also keep their token IDs and pass all 1326 checks; varint measured
238.28 tokens/s (`quality-mtp-head-q4`). At this step the full panel was below 200.
The near-context input/output IDs and both additional Chinese replies also
match their earlier references, with zero decode major faults. See
`near-context-mtp-head-q4`, `heldout-mtp-head-q4`, and `mtp-head-checks.json`.

Retuning the fused-copy launch from 64 to 32 blocks measured **199.71 tokens/s**
versus **198.18** with the preceding setting in the same exploratory binary.
With 32 copy blocks, the final-row-only draft head then measured **200.25**
versus **199.70** with the option off in the same final binary (+0.28%). An
independent repeat measured **200.26**. All target IDs and draft acceptance
counts stayed identical; these are small gains on this hardware and workload.
See `bench-head-q4-fused32`, `bench-draft-cost-control`, `bench-mtp-last-head`,
`bench-mtp-last-head-control`, and `bench-mtp-last-head-repeat`.

The final recipe uses 16 copy blocks. Its two independently started panels
measured **200.93** and **200.83 tokens/s**, or **200.88** over all 24,576 output
tokens and 122.34 seconds of reported decode time. All twelve measured requests
retain the reference input/output IDs, finish reasons and draft acceptance
counts, with zero decode major faults. No compile or profiler ran alongside
these panels. See `bench-mtp-last-head-fused16{,-repeat}` and
`acceptance-200.json`. Prose remains about 180 tokens/s; the 200 target is the
aggregate panel rate, not a per-task lower bound. The margin over 200 is small.

The final recipe also retains the reviewed code outputs (1326 checks passed),
the 32688-token input plus 64-token generation, and both additional Chinese
1024-token replies. A same-binary sampled comparison with fixed seed 20261104,
temperature 0.7, top-k 20, top-p 0.9 and repeat/frequency/presence penalties
also preserves every output token and draft acceptance count. These checks
are functional evidence; their short decode rates are not part of acceptance.
See `quality-mtp-last-head`, `near-context-mtp-last-head`,
`heldout-mtp-last-head`, `sampling-mtp-last-head{,-control}`, and
`mtp-last-head-checks.json`.

A timing-derived adaptive-draft experiment was rejected. Five independent
calibration prompts gave round/row/draft-call costs of 10.91/1.273/0.598 ms.
Those values with draft limit eight measured 196.01 tokens/s, versus 198.18
with the existing costs and limit six in the same binary. Output tokens were
unchanged. The added cost knobs and instrumentation were removed; the patch,
fit and raw runs remain in `draft-cost-deferred.patch`, `draft-cost-fit.json`,
`draft-cost-calibration`, and `bench-draft-calibrated-k8`.

The exact token records and the calibration profile are in
[`bench/results/2026-11-03-rx7900xtx-tensor-parallel`](../bench/results/2026-11-03-rx7900xtx-tensor-parallel/).

The profile was collected from separate Rust, Go, SQL, Chinese systems and
graph-algorithm prompts, 1536 generated tokens each. The measurement prompts
were not used to train it. Decode counts select residency; prompt counts are
zeroed in the derived profile. Engine warmup counts are present in the
calibration. This is workload calibration, not a claim that the profile suits
every workload.

Staging and residency changes produced the exact same 2048 output token IDs
on all three measurement prompts, including both calibrated repeats. An early
trial that skipped staging for hot experts without IQ reuse was slower and was
removed. The retained cold-only option is evaluated together with IQ reuse.
Two gate/up kernel trials were also removed: a bounded persistent grid passed
3072 exact direct/graph checks but was slower; cooperative loading of job
descriptors passed 1024 checks but had no consistent benefit. Their experiment
logs are retained. The retained IQ reuse kernel passed 1024 bit-for-bit checks
on both cards, including T=1..16, direct launches, graph replay, changing job
counts and output guards. All six 2048-token full-model runs match the earlier
split outputs exactly. The kernel microbenchmark shows larger gains than the
full engine: at 24 jobs its three IQ formats take about 62–67 us instead of
118–126 us. The microbenchmark is not the decode acceptance result.

Increasing native MTP to five drafts with a 0.30 confidence cutoff was slower
(115.1 / 75.9 / 126.6 tokens/s). A fixed split draft limit of three produced
132.9 / 116.0 / 136.1 tokens/s, with the same output IDs, without a clear gain.
Neither setting is adopted by the reproduction recipe. With cached host weights,
a draft limit of eight measured 153.8 / 131.3 / 158.5 tokens/s, with reference
token IDs unchanged; it did not improve on the limit of six.
`FNX_RING_GRID=80` with cold-only staging measured 138.7–149.1 / 116.8–125.4 /
145.0–150.6. Its second-pass GPU verify times differ from the default grid by
less than 0.5%; it is not adopted as a demonstrated optimization.
The existing prefill FP16 matrix kernel was also slower than IQ reuse for
the representative small expert jobs in a separate synthetic trial. It is
not selected for decode.
A separate gfx11 small-window WMMA prototype tested 16/32/64-row tiles with
256/128/64-column steps. At 24 expert jobs the 32-row tile took about 81–91 us
versus 57–63 us for retained IQ reuse. The prototype is not integrated. Its
FP16 arithmetic also differs from the integer-dot reference (relative L2 error
about 0.00023–0.00026 on the synthetic inputs).

The split loader already converts 675 dense/small tensors to Q8_0/F32 for this
GGUF. Thus, split and the native engine do not have the same arithmetic or
necessarily the same output tokens. The checks above compare split variants.
They do not establish quality equivalence with the native BF16 path. The
2048-token performance replies hit their length limit.

A separate IQ dot-product prototype split the integer accumulation into two
or four independent chains. It passed 2304 exact direct/graph comparisons,
but timing did not improve consistently across expert counts. It is not
integrated; the source and microbenchmark log are retained in `iq-ilp/` beside
the other results. No full-model speedup is claimed for it.

Separate completed-code tasks cover interval merging (316 checks),
lexicographically smallest topological sorting (125 checks), and canonical
uint64 varints (885 checks). Both the native engine and the saved pre-IQ split
binary passed all 1326 checks. Their output code differs. These three tasks
are limited functional evidence, not a broad quality evaluation. The reviewed
outputs, prompts and reports are saved with the measurements; check them using
`python3 tools/tp_code_quality.py <quality-output-directory>`. The checker
executes Python and is intended only for reviewed outputs of these fixed tasks.
The quality runs overlapped CPU compilation and are excluded from speed claims.
The IQ-reuse/cold-only binary was then run on the same tasks. It produced
exactly the reviewed split token sequences and passed all 1326 checks too.
The cached direct-read/embedding-refresh binary also reproduced those exact
token sequences and passed all 1326 checks (`quality-hostcache`).
The 16-lane binary also matched them and passed all 1326 checks
(`quality-hostcache-iq16`).
The down-precompute binary passed the same checks with exactly the reviewed
tokens (`quality-hostcache-iq16-pre1`); route tracing was enabled in that quality
run, so its timing is excluded from performance claims.
The packed-sign binary also matched all reviewed tokens and passed all 1326
checks (`quality-hostcache-iq16-pre1-swar`).
The 1024-slot static cache again reproduced all reviewed token IDs and passed
all 1326 checks (`quality-static-cache1024-ram16`). This run overlapped a CPU
compile and is excluded from speed claims.

A four-second ROCprofiler sample showed matrix staging at 23.15% and the
window-sum peer kernel at 15.99% of summed kernel duration. This includes
overlap and waiting across both cards; it is not a wall-time speedup estimate.
The raw trace is retained with the experiment outputs. Profiler timings are
excluded from the performance table.
With IQ reuse, another sample still puts staging at 23.99% and peer window sums
at 16.38% of summed kernel time. The IQ gate/up calls average about 76–82 us for
seven-row windows; reducing this alone cannot close the remaining decode gap.
A later four-second sample with packed IQ signs puts gate/up at 19.52%, down
at 14.26%, dense Q8 at 17.55% and peer window sums at 19.63% of summed kernel
time. The different sample mixes prevent treating these percentages as an A/B
speedup. Raw trace locations and binary provenance accompany the summary.

Additional tuning experiments kept every benchmark output token unchanged but
did not improve the 170.19 tokens/s cache baseline: choosing the longest prompt
lookup match measured 168.85, fixed MTP K=3 measured 167.05, a top-logit margin
cutoff measured 167.95, and reducing HC to 40 blocks measured 170.18. These
changes are not enabled in the recipe. Fusing the shared expert into the IQ
router/gate-up launches measured 170.05 tokens/s and was also reverted; its
routed/shared outputs passed direct and graph checks and all benchmark tokens
matched. Raw runs and rejected source patches
are retained with the results. The margin experiment's first pass had major
page faults, so its timing also includes that variability.

An exact INT8 WMMA dense prototype matched the original Q8 reductions on both
cards, but was slower for the common 5–7-row windows, including with tile-packed
weights. It remains an experiment. The measurements use warm repeated kernels
and are not full-model speed claims.

A direct-read down prototype (precomputed activations read from device memory,
with each IQ4_NL/Q4_0 weight block decoded once per job) passed 1536 direct/graph
checks across three FF widths, including negative/zero scales, changed inputs,
partial output blocks and guards. Its first six-run result was 171.43 tokens/s.
A same-build repeat measured 171.32 with the flag off and 169.32 with it on;
the latter had 288/277/85 first-pass major faults after the first token. The
second passes differed by less than 0.4% on each task. This does not establish
a reliable full-model gain, so the source is retained as a deferred patch.
Its complete code-quality run matched all reviewed token IDs and passed all
1326 checks. The 494-token varint task measured **201.14 tokens/s** without a
concurrent compile/profiler, but this short individual task does not satisfy
the broader 2048-token panel's 200 tokens/s target.

IQ lookup tables staged in shared memory matched the reference but did not
improve the representative kernel timings. A separate HC prototype with
64-row groups and 80 blocks was about 16–17% faster in warm microbenchmarks;
that initial result did not cover every window. Broader checks found register
pressure at 5–6 rows and excessive LDS usage at 13–16 rows. The final compact
layout constrains register use and keeps the original layout above 12 rows.
The runtime checks all selected kernels' occupancy rather than relying on the
earlier finite-work occupancy probe. The original 768-thread layout cannot
safely use an 80-block grid on these cards.

A version forcing compact execution for all windows passed 5760 exact checks
but made the large windows slower. It measured 172.33 tokens/s versus 171.77
with the flag off in the same binary; all output tokens matched. It is retained
as `hc-all-windows-deferred.patch`, not the final dispatch policy.

The exploratory clock probe also confirmed that `clock64()` can go backwards
on these cards while `wall_clock64()` advances at 100 MHz. AMD documents
[the RDNA3 clock limitation](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_cpp_language_extensions.html).
The HC watchdog now uses the wall clock and has a bounded missing-arrival
test. Other device watchdogs still need a separate correction; reported decode
rates use host timing and GPU events, not those watchdog counters.

A diagnostic trace of the final HC/cache/release binary is summarized in
`profile-hc-hybrid-summary.txt`. In its first two seconds (decode only), summed
kernel durations were 28.3% ring waits, 17.6% Q8 projections, 12.1% cold-cache
copies and 8.4% HC. These percentages include overlapping kernels and waits,
and tracing substantially increases host launch overhead. They are not shares
of recoverable wall time. The existing `CardPool` already launches each card's
single-request and MTP graphs in parallel. Copy/compute overlap remains a
candidate for investigation. Simple two-hit/ghost admission simulations moved
more host bytes than the existing CLOCK policy, so those policies were rejected
before implementation (`cache-admission-estimates.json`).

A separate-stream cold-cache copy experiment passed 4608 byte-for-byte checks
but reduced the full panel from 173.8272 to 172.7733 tokens/s (-0.61%) in a
same-binary comparison. Every output token matched. Splitting gate/up and down
copies introduced extra graph events and substantially more host minor faults;
this implementation was removed. The patch, test log and both measurements are
retained as `cache-overlap-rejected.patch`, `cache-overlap-checks.txt`, and
`bench-cache-overlap{,-control}`. No overlapping-copy flag is enabled by the
accepted recipe.

## Reproduce

On the measured host, with the retained binary and converted MTP pack:

```sh
cd /home/john/Strata
bash bench/results/2026-11-03-rx7900xtx-tensor-parallel/reproduce-200.sh /tmp/strata-tp-new-run
```

The directory must be new. This runs the benchmark in a child process.
The script supports `TP_EXE`, `TP_MODEL`, `TP_MTP` and `TP_TOKENIZER` overrides.
The final binary SHA-256 is
`b051993f8a7793706ee848959226d99876263aa0c9d98630e981d88b2db5da3e`.
It is also retained at `build-tensor-parralize-baseline/strata-split-mtp-last-head`.

Build the split target using the installed ROCm SDK and its compiler:

```sh
PATH="$PWD/.venv/bin:$PATH" \
B="$PWD/build-tensor-parralize" \
HP="$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_devel" \
LIBW="$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu" \
GG="$PWD/third_party/llama.cpp" CARDS=2 ARCHS=gfx1100 JOBS=4 \
bash packaging/build_split_hip.sh
```

Set `MODEL` and `TOKENIZER` to the installed GGUF and tokenizer directory.
For the measured Q4 draft recipe, prepare a new runtime directory from the
existing MTP source directory (containing `mtp-manifest.json`, `tensors/`, and
`rt/`), then set `MTP` to that new directory:

```sh
python3 tools/mtp_experts.py --src "$MTP_SOURCE" --out "$PWD/build-tensor-parralize/mtp-rt-q4"
MTP="$PWD/build-tensor-parralize/mtp-rt-q4"
```

The output must be new. The tool writes 1415577600 bytes of Q4 expert data and
a provenance report; it symlinks the existing dense weights and draft vocabulary.
Set the SDK library paths for the child process:

```sh
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib"
HIP_VISIBLE_DEVICES=0,1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
FNX_STATIC_EXPERTS=1 FNX_EXPERT_RAM_GIB=13.875 FNX_STATIC_CACHE_SLOTS=1024 \
FNX_STATIC_CACHE_GROUPS=2 \
FNX_MOE_STAGE=0 FNX_PF_ROWS=256 \
FNX_IQ_REUSE=1 FNX_IQ_GROUP=16 FNX_MOE_PRE_ROWS=1 \
FNX_STATIC_HOST_CACHE=1 FNX_EMBED_WARM_AFTER_LOAD=1 \
FNX_RELEASE_COPIED_WEIGHTS=1 FNX_KV_CONTEXT_ONLY=1 \
FNX_HC_ROW_GROUP=64 FNX_STATIC_CACHE_FUSED_DN=16 \
FNX_Q8S_ROW_GROUP=2 \
FNX_GDN_REPLAY=1 FNX_BATCH_DRAFTS=0 \
FNX_MTP_EXPERTS=q4_0 FNX_MTP_HEAD_Q4=1 FNX_MTP_LAST_HEAD=1 \
FNX_EXPERT_PROFILE="$PWD/bench/results/2026-11-03-rx7900xtx-tensor-parallel/swift-decode-routes.bin" \
python3 tools/tp_decode_bench.py \
  --exe build-tensor-parralize/src/split/strata-split-2 \
  --model "$MODEL" --mtp "$MTP" --tokenizer "$TOKENIZER" \
  --context 32768 --tokens 2048 --repeats 2 --out /tmp/strata-tp-new-run
```

The output directory must be new. The recipe uses calibrated placement via
`FNX_EXPERT_PROFILE`, a measured split-format count file. It is different
from the native engine's `STRP` profile format.
The recorded profile is
`bench/results/2026-11-03-rx7900xtx-tensor-parallel/swift-decode-routes.bin`;
its prompts and provenance are beside it.

The native comparison uses the same prompts:

```sh
python3 tools/tp_decode_bench.py --config strata-swift-iq3_s.json \
  --context 32768 --tokens 2048 --repeats 2 --out /tmp/strata-native-new-run
```

Use `--cases prompts.json` for a JSON object mapping names to prompts, and
`--prefix-file context.txt` to test real longer inputs. Changing the allocated
context alone does not measure long-context decode.

## Validation and remaining work

- HIP build and model startup self-tests passed on both physical cards.
- HC compact dispatch: 5766 bit-exact comparisons passed on both cards,
  covering T=1..16, both barrier forms, optional writes/injection/quantization,
  changed and zero inputs, guard bytes, graph replay and barrier-counter wrap.
  A mixed-window sequence switches between 48- and 80-block grids on the same
  scratch, including inside a graph, without resetting counters between reads.
- GGUF ownership release passed a host-only regression and ASan/UBSan, covering
  mixed mapped/converted tensors, retained bytes, metadata, invalid retention,
  null released pointers and repeated release.
  The complete model also matched all reviewed quality token IDs and passed
  all 1326 checks after release (`quality-release-owned`).
- Static placement: all 256 partitions of eight experts passed, including
  empty hot/cold banks and invalid-input rejection; tested with `NDEBUG` too.
- Staging: 432 byte-for-byte direct/graph checks passed on both cards, with
  changing routes, mixed host/device pointer tables, contiguous weights,
  hot-pointer bypass, several strides and guard bytes. Full-model startup
  and nine cold-only benchmark outputs also match the reference token IDs.
- Extended host-memory checks: 1440 direct/graph checks passed on both cards,
  including coherent and non-coherent allocations, changing CPU contents,
  all-hot/all-cold masks and guard bytes.
- Existing split Python suite: 52 tests passed.
- Existing split serving/fallback Python suite: 25 tests passed.
- IQ gate/up reuse: 1024 exact direct/graph checks passed on both physical cards;
  the full engine builds and matches all six reference token sequences.
- Extended IQ grouping: 2048 exact direct/graph checks passed on both cards,
  covering both lane groups and the fallback. Guard initialization is explicitly
  ordered on the compute stream.
- Packed IQ signs: all 2048 exact direct/graph checks passed again on both cards.
- Fused down copy: 3584 exact gate/up comparisons passed on both cards, including
  1536 with down-byte and guard checks. These cover T=1..16, 1/32/64/256 copy
  blocks, coherent and non-coherent host backing, nonzero global expert/slot
  offsets, changing lists and host bytes, empty GU jobs with nonempty copies,
  empty copies, and graph replay.
- Cold decode cache: 2304 exact checks passed on both cards, with 160/256/1024 slots,
  four layer sizes, changing routes, eviction, zero/max jobs, all-hot/all-cold
  and mixed placement, coherent/non-coherent backing, guards and graph replay.
- Down precompute: 512 exact direct/graph comparisons passed on both cards,
  including T=1..16, changing routes/job counts, IQ4_NL and Q4_0, and guards.
- Grouped cache: 3840 byte-for-byte direct/graph checks passed on both cards,
  including single/two-pool layouts, nonzero pool offsets, evictions, hot/cold
  patterns, immutable backing tables and all pool/output guards. Host layout
  checks passed normally and under ASan/UBSan: 384 generated layouts, overflow
  and invalid-input rejection, full-window capacity, memory/slot bounds and
  order-independent layer assignment.
- Both engines passed the three completed-code tasks described above.
- Context-sized int8 KV: the complete panel and reviewed code tasks match
  their reference token IDs, with 1326 code checks passing. A real 32688-token
  input plus 64-token generation matches the original pool at context 32768.
- Q8 row groups: 6144 bit-exact direct/graph comparisons passed on both cards
  across settings 0/1/2/4, T=1..16, packed/split activations, segmented and plain
  matrices, uneven final rows, changed/zero inputs and output guards.
- Q4 draft pack: source tensor hashes verified; 24064 blocks match the pinned
  GGUF Q4 reference. Target token IDs match on the full panel, all three reviewed
  code tasks (1326 checks passed), and two additional Chinese prompts.
- Logged GDN rollback: 384 exact core and 5760 exact restoration comparisons
  passed on both cards. Tests cover T=1..16, every partial acceptance length,
  both history parities, changed/zero inputs, output guards, direct/graph
  execution, and two independent layer descriptors with rotated head order.
  The original full snapshots are the byte oracle. Uniform FMA contraction
  orders failed the prototype; the retained row-dependent order matches.
- Q4 draft head: 32768 converted blocks match the pinned GGUF reference
  byte-for-byte in both normal and ASan/UBSan host builds. Checks include
  signed scales, subnormals, zero blocks, maximum ties, row order/repetition
  and invalid metadata. The GPU reader passed 384 direct/graph product and
  guard checks on both cards, across T=1..16 and changed/zero inputs. The
  eight-lane sums are compared with the original 32-lane Q4 reader with a
  2e-5 relative/absolute tolerance; different reduction groups need not be
  bit-identical. Full-model target token comparisons are exact.
- Last-row draft head: 384 additional bit-exact last-row/full-window comparisons
  passed on both cards, T=1..16, two shapes, changed/zero inputs, direct/graph
  execution and guard bytes. Together with the original Q4 product checks,
  the GPU test now passes 768 checks. Full-model greedy and seeded sampled
  output/acceptance comparisons are exact; the final recipe passes all 1326
  reviewed code checks and the near-context/Chinese checks described above.
- CUDA has not been built or run here. No SYCL files are changed.
- Outside this acceptance: performance at the configured 204800 context, broader
  quality and workload coverage, concurrent serving, and integration into launch policy.

The implementation builds on Strata's existing sFNX split engine. Reference
projects considered include [vLLM's custom all-reduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py),
[Paiton](https://github.com/Eliovp-BV/paiton-vllm-plugin), and
[Strata dual 3090](https://github.com/q8atnight/strata-dual-3090).


## Exact attention head partitioning and shared KV cache (November 4)

The follow-up target is 220 tokens/s on the same single-request Swift IQ3_S
panel. It has **not yet been reached**. The opt-in candidates below preserve
the target model, full 32768 context capacity, generated tokens and draft
acceptance. Results are weighted generated tokens / engine decode time, with
warmup excluded; each panel has code/prose/edit, 2048 output tokens, two repeats.

| Configuration | Weighted tokens/s | Observation |
| --- | ---: | --- |
| Fresh cfc7 baseline | 200.35 | Original accepted recipe |
| Attention head partitioning, original RAM budget | 206.54 | All token IDs exact |
| Head partitioning, 13.5 GiB expert RAM | 207.57 | Saved VRAM holds more fixed experts |
| Head partitioning, later control | 208.10 | All token IDs exact |
| Shared KV suffix, automatic cache pool | 209.20 | Independent process, no decode major faults |
| Shared KV suffix, pool 1 | 209.24 | No material difference from automatic selection |
| V5 binary with both new features disabled | 201.14 | Original recipe; all IDs exact, no decode major faults |

`FNX_ATTN_HEAD_SPLIT=1` gives each card twelve query heads and one KV head.
Within that head, both original position partitions are retained, including
merge order. The existing page layout stores the two virtual partitions in
its two head slots, so page sizes and KV capacity are unchanged. The compact
Q/K/V projections save about 224 MiB per card and remove peer attention merge.
Prefill deliberately retains the original full-matrix GEMM shapes and leading
dimensions: compact GEMM shapes changed reduction order and failed token
identity. Unowned projection rows are zero-filled in the existing staging
buffer, then owned outputs are gathered. This costs no persistent extra buffer.
The feature requires the two-card Flash-Next HIP build, one lane and
`FNX_BATCH_DRAFTS=0`; other attention layouts retain the original path.

`FNX_STATIC_CACHE_ARENA=1` makes one allocation serve both the KV page pool and
one static expert cache pool. Initially 16 KV pages are reserved; unused suffix
pages hold additional host-backed cache entries. Before KV grows, existing
arena lending logic evicts intersecting entries back to their immutable host
pointers, waits for completion and zeros the newly exposed pages. Returning
pages leaves empty CLOCK slots that fill on demand. Even at maximum context,
the original cache slot budget remains available. The default placement adds
218 slots per card in this configuration. The model's original `FNX_ARENA`
placement remains disabled; static hot experts never move to host memory.
`FNX_STATIC_CACHE_ARENA_BASE_PAGES` can select a prefix from 1 to the allocated
KV capacity. `FNX_STATIC_CACHE_ARENA_POOL=-1` selects the pool with most slots;
0 or 1 select an existing pool explicitly. Static cache and one lane are required.

Evidence is under `bench/results/2026-11-04-rx7900xtx-decode220/`.
`candidate-v5.patch` records the tested implementation before further tuning.
`run-heads-experiment.sh` carries the same benchmark inputs and permits
`TP_EXE`/`TP_EXPERT_RAM_GIB` overrides; it does not replace the accepted
`reproduce-200.sh` recipe. The V5 binary is
`build-tensor-parralize-baseline/strata-split-static-kv-v5`, SHA256
`898a2a7f50be02dd3a40e38bcd041ec7f4ee73db7a429044ca9d0e2de2ca0c53`.
The automatic-pool V4 repeat uses SHA256
`1906db9a24ccb1838e9c0ca1ddeb736b78d6c974e1e06c2c59bafb2bda62a2a7`.
The first V4 panel had decode major faults after compilation and is not used
as performance evidence; the subsequent process had none.

Validation so far:

- 21504 exact attention/append comparisons on both cards: every T=1..16,
  both merge orders, F16/int8 KV, page boundaries and graph replay.
- 192 exact prompt GEMM comparisons preserving original shapes. Naive compact
  shapes failed 40 of 192 corresponding comparisons and were discarded.
- 21945 host shared-region layout checks; 6144 exact GPU cache comparisons,
  including 2304 lending/returning cases, both pools, non-aligned slot counts,
  coherent/non-coherent backing, fixed hot entries, guards and graph replay.
- The three reviewed code tasks match the reference output and pass all 1326
  checks. Both Chinese prompts (1024 outputs), 32688-input/64-output context
  boundary, fixed seeded sampling with penalties, and forced arena churn all
  match their reference tokens. The near-context run grows the pool through
  16, 80, 144 and 172 pages, lending all 218 extra slots. See
  `static-kv-v5-validation.json` and `validate-static-kv-v5.sh`.

Peer ring batches of 2 and 4 passed 2880 exact checks but measured 207.76 and
207.63 against the 208.10 original-path control, so the implementation was
removed. `HIP_FORCE_DEV_KERNARG=1` and `HSA_ENABLE_SDMA=0` also showed no material
gain. Speculation limits 4 and 5 measured 203.09 and 206.96, so the limit remains
6. These are measured local outcomes, not general recommendations for other
models or machines. CUDA, concurrent serving and the production 204800-context
configuration remain outside this new validation.

### Gate/up cache fill in the expert compute launch (V7)

`FNX_STATIC_CACHE_FUSED_GU=1` fills a cold IQ expert's gate/up rows in the
thread block that computes those rows. Coalesced reads from the immutable host
backing populate both the VRAM cache slot and shared memory; that block then
uses the shared copy. Hot jobs can start immediately. The original integer
dots, per-lane floating-point accumulation and reduction order are unchanged.
The existing down-copy prefix remains in the same launch. Kernel completion
publishes all copied bytes before subsequent down work or later cache hits.

This path requires the static expert cache, `FNX_STATIC_CACHE_FUSED_DN>0`,
`FNX_MOE_STAGE=0`, IQ reuse with 16 lanes per row, and two-card HIP Flash-Next.
IQ2_S, IQ3_XXS and IQ3_S gate/up weights are supported. Other quantizations and
MTP keep their existing paths. Deferred GU copy still runs the cache planner
when prefetch probing is enabled. The feature is disabled by default.

On the full six-request panel, the V7 candidate measured **210.93794 tokens/s**
versus **209.34024** with only this flag disabled in the same binary, a 0.76%
improvement. All six requests improved; all output IDs and draft acceptance
matched, and both processes had zero decode major faults. The 220 target
remains unmet. The binary is
`build-tensor-parralize-baseline/strata-split-gu-cache-v7`, SHA256
`3cad60618315c9fbe249f38731b655c821997246fc27b7244b40bdbaebd0cffe`.
See `run-gu-cache-v7.sh`, `gu-cache-v7-same-binary.json`,
`validate-gu-cache-v7.sh` and `gu-cache-v7-validation.json` in the November 4
evidence directory.

Validation includes 12288 exact GPU gate/up, cache-byte, down-copy and guard
comparisons across both cards, all T=1..16, all three IQ formats, coherent and
non-coherent host backing, partial final row groups, changing routes/weights,
and graph replay. The existing 6144 GPU cache checks also passed. The three
reviewed code tasks pass all 1326 checks and match their reference outputs.
The 32688-input/64-output context boundary, two 1024-token Chinese generations,
fixed seeded sampling with penalties, forced cache churn, and prefetch-probe
output prefixes all match the reference.

The following alternatives were measured and retained only as scratch
experiments: fusing cache copy with shared-expert down had no end-to-end gain;
reading newly copied GU rows from global memory had no consistent advantage
over shared memory; a per-block-flag HC barrier was slower than the existing
atomic barrier; raising compact HC's grid from 80 to 96 blocks was slower.
An exact INT8 WMMA dense-Q8 prototype passed 864 direct/graph comparisons,
but regressed the common seven-token shapes and was not integrated.

### Lossless packed down weights (V10)

`FNX_PACKED_Q2_DOWN=1` retains the GGUF's original Q2_0 routed-down storage
instead of expanding each 18-byte block into two Q4_0 blocks. Nine layers of
Swift IQ3_S use this format, saving 2123366400 bytes (1.97754 GiB) across both
cards' expert weight set. The weights are unchanged: the two-bit code maps to
the original Q4 nibble as `code + 7`, with the same scale. Decode expands a
half-block once into registers and reuses it for the job's tokens, retaining
both Q4 partial dots, the rounded Q8 activation-sum correction, and the original
floating-point accumulation order. Prefill expands into the original Q4 tile
registers and retains the same WMMA shapes and arithmetic. Its alternative Q8
BigJob path is also supported.

The option is disabled by default and requires plain static placement with the
two-card HIP Flash-Next engine. Column slices must align to 64 values. Other
GGUF readers retain default repacking; other tensors and MTP are unchanged.
The V10 recipe combines this option with V7 and sets expert RAM to 11.625 GiB
instead of 13.5 GiB, using the saved space to keep more experts resident. It
loads 5973 cold experts, has 142 MiB free per card after warmup, and preserves
the full 32768-position context allocation.

The complete six-request panel measured **214.14082 tokens/s**, against
**210.89848** with packed down disabled and the original 13.5 GiB RAM budget
in the same binary: **+1.54%**. All six requests improved; input/output tokens
and draft acceptance were identical. Both engines exited normally with zero
decode major faults. The aggregate remains below the 220 target. The immutable
binary is `build-tensor-parralize-baseline/strata-split-packed-q2-v10`, SHA256
`bf1e35c8d08850128c0a8eae2b1306cbea1c19f3e318614629831ae8ab30ef01`.

Validation covers 1048576 exact dot comparisons (including zero, signed zero,
subnormal and extreme finite scales), 3072 decode/guard comparisons, 648 Q8
prefill comparisons, and 648 WMMA prefill comparisons on both cards. The tests
exercise odd output widths, 64/256/320-column slices, empty jobs, partial token
tiles, both activation-precompute modes and graph replay. The host GGUF test
checks default conversion, packed metadata, unchanged other tensors, exact
values, and mapped-buffer lifetime. The three reviewed code tasks pass all
1326 checks; near-context, Chinese, seeded sampling, cache churn and probe
outputs match their existing references. See `packed-q2-v10-same-binary.json`
and `packed-q2-v10-validation.json` in the November 4 evidence directory.

Reproduce the panel with a fresh destination:

```sh
bash bench/results/2026-11-04-rx7900xtx-decode220/reproduce-214.sh NEW_OUTPUT_DIRECTORY
```

Build the focused tests with CMake targets `split_moe_q2_down_test`,
`split_moe_q2_mma_test` and `split_gguf_release_test`; the first two require both
cards. `split_moe_q2_down_test --bench` also runs diagnostic kernel timings.
Changing expert ranking to frequency alone, four-token GU tiling, a signed-IQ
lookup table, and host transparent huge pages did not improve their respective
controls and remain excluded from the selected recipe.

### Cache pool padding (V11)

`FNX_STATIC_CACHE_MIN_PADDING=1` considers two-pool partitions at cuts of
ascending gate/up bytes, down bytes, and total bytes. It chooses lower
layer-weighted pool padding within the original allocation budget; ties retain
the median-based default. The calculation uses only weight geometry, with
integer-overflow-safe cost comparisons and the existing minimum slots per pool.
This option requires `FNX_STATIC_CACHE_GROUPS=2` and is off by default.

For packed Swift IQ3_S, the selected 37/11-layer split increases base cache
slots from 1088 to 1184 and total slots with spare K/V borrowing from 1292 to
1402. Total cache plus borrowed bytes remain within the same budget. The
unchanged 11.625 GiB expert RAM recipe measured **214.73470 tokens/s**, versus
**213.84493** with only this option disabled in the same binary (+0.42%). All
six input/output and draft streams match, normal process exits are recorded,
and decode major faults remain zero. The 220 target remains open.

`split_static_cache_layout_test` checks 768 generated layouts, deterministic
pool assignment, budget constraints, default compatibility, the real packed
shape fixture and large integer costs. `split_static_kv_cache_test` passes
21945 layout checks. Code, near-context, Chinese, seeded sampling, forced churn
and probe outputs match their existing references. Reproduce with
`reproduce-cache-padding-v11.sh NEW_OUTPUT_DIRECTORY` in the November 4 evidence
directory. The tested binary SHA256 is
`006e535eb09e75b7a7fc61c781afd7f0c1d085e7aa1beacc6801f0aace1c4b2b`.

### Speculative draft-head screening (V13)

`FNX_MTP_HEAD_MODE=3` computes the first half of each Q4 draft-head dot and
retains four candidates in each group of sixteen vocabulary rows. It completes
those candidates with the original per-lane accumulation and reduction order;
other draft scores receive a finite negative sentinel. Candidate selection is
approximate, and a discarded row may have won after a complete dot. This affects
only speculative proposals: the unchanged main model verifies them. Sampling
and penalty requests use the original complete draft projection, with the mode
included in the graph key so transitions recapture the appropriate graph.

Mode 0 is the default. Mode 1 evaluates a specialized complete Q4 dot; mode 2
screens one quarter but lost some acceptance and is not the selected recipe.
Nonzero modes require the two-card HIP Flash-Next engine, a Q4 draft head, and
`FNX_MTP_LAST_HEAD=1`. Main output projections and model weights are unchanged.

Mode 3 measured **216.62750 tokens/s**, against **214.97927** for the same binary
in mode 0 (+0.77%). All six input/output token streams and draft acceptance
counts match, all six requests improve, decode major faults are zero and both
engines exit normally. This remains below 220 tokens/s. The tested binary SHA256
is `28e377099cf8c62facf1b73b897f9ee3a294de3844f4cba6185033b38c891328`.

`split_mtp_head_gemv_test` covers 216 bit-exact selected-score and guard cases,
survivor counts, stable ties, direct launches and replay on both cards. A retained
candidate's full score is exact; this test does not assert that pruning always
retains the global winner. The full quality suite includes the 1326 reviewed
code checks, near-context generation, Chinese prompts, seeded sampling, forced
cache churn and the prefetch probe. A same-process sequence alternates seven
greedy, sampled and penalized requests and matches the full-head control. One
initial sampling warmup had a major fault; the retained isolated recheck has
none, and both runs' outputs match. See the November 4 evidence README for raw
records and `reproduce-217.sh NEW_OUTPUT_DIRECTORY`.

### Exact recurrent projection fusion, router storage and cache credit (V26)

The unchanged Swift-1.5 IQ3_S single-request panel now reaches
**222.14549** and **222.61031 tokens/s** in two independent engine processes,
or **222.37766 tokens/s** across all 24,576 measured output tokens. Both
runs exceed 220. Each contains code, prose and edit prompts with 2048 outputs,
repeated twice; context allocation is 32768. The metric divides output tokens
by engine DONE decode time and excludes warmup and prefill. Individual prose
requests remain around 198 tokens/s; this is an aggregate panel target.

The retained additions since V13 are:

- `FNX_STATIC_CACHE_MAX_AGE=2`: bounded CLOCK hit credit. A cache hit increases
  credit up to the configured maximum; a hand pass consumes credit before
  eviction. New entries start at one, and K/V lending clears their credit.
  Values 1..4 are accepted; default 1 retains the previous policy.
- `FNX_ROUTER_BF16=1`: compact a main router matrix only when every FP32 value
  has zero low 16 bits. Any inexact matrix falls back to FP32, without rounding.
  Decode expands the exact bits and retains the original accumulation order;
  prefill restores FP32 into existing scratch before the same SGEMM. All 48
  routers on each card qualify in this model, saving 120.234375 MiB per card.
  This allows the selected expert RAM budget to decrease to 11.25 GiB.
  MTP routers keep FP32. The default is disabled. The holder image version
  changes to IMG2 to account for the additional router address.
- `FNX_Q8S_FUSED_AB=1`: use the existing R=2 fused Q8S/alpha-beta kernel for
  the two-card recurrent projection's 8192 rows. This removes its stream
  fork/join while retaining both dot products' arithmetic. It requires HIP,
  two cards, Flash-Next and `FNX_Q8S_ROW_GROUP=2`; default 0 keeps the old path.
- `TP_CPU_AFFINITY=0-3` in the benchmark recipe, with
  `HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0` so ROCr helpers inherit the process mask.
  The harness records actual main and helper-thread masks after load and after
  decode; every recorded thread used CPUs 0..3. This runtime behavior is
  documented in [ROCr environment variables](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html).

With only Q8S/alpha-beta fusion disabled, the same binary measured
**220.09909 tokens/s**. The first enabled panel improves by **0.93%**,
with all six requests faster. Input/output token IDs and accepted/offered draft
counts match the unfused control, both enabled runs, and the V13 checkpoint.
Every engine exits normally; measured decode major-fault deltas are zero,
including warmup. These counters start after the client receives the first token.

Production tests pass 1152 exact Q8S/AB direct-launch, graph and guard checks,
576 router GPU checks, 65536 host BF16 bit-pattern checks and 65535 rejection
checks, plus 18432 cache-policy checks against an independent CPU oracle
(6912 with K/V lending). Full final-recipe regression passes all 1326 reviewed
code checks; near-context generation, held-out Chinese replies, fixed-seed
sampling with penalties, cache churn and probe mode match their reference
outputs. Sampled draft counts also match. V13's approximate draft-head screening
remains selected only for unpenalized greedy proposals; main verification stays
enabled. These checks do not establish broad quality equivalence with another
engine or every possible prompt.

The tested binary is `build-tensor-parralize-baseline/strata-split-q8s-ab-v26`,
SHA256 `83edcdb64b0b620c186b759dac207e458019a5eb7b358e5944b169e77a91cf45`. From `/home/john/Strata`:

```sh
bash bench/results/2026-11-04-rx7900xtx-decode220/reproduce-220.sh NEW_OUTPUT_DIRECTORY
python3 bench/results/2026-11-04-rx7900xtx-decode220/acceptance-220.py
```

The first command requires a new directory. `TP_EXE` can select a newly built
`strata-split-2`, which needs independent validation. The second command audits
the retained acceptance artifacts. Build instructions above still apply;
focused targets are `split_q8s_ab_test`, `split_router_bf16_test`,
`split_lossless_bf16_test` and `split_moe_cache_test`.

The recipe leaves only **34 MiB free VRAM per card at load** on this host.
It is validated for the recorded dual RX 7900 XTX and 32768 allocation; the
deployed 204800-context configuration, concurrent serving, and CUDA performance
have not been accepted by these tests. Performance panels ran without concurrent
builds, profiling or other experiment GPU jobs. Runtime defaults and the existing
service configuration are unchanged; all new execution options are opt-in.

Earlier near-threshold results remain recorded: V17 reached 220.25611 but its
independent repeat was 219.28564; V21's four-core recipe reached 220.13339 but
repeated at 219.95017. Neither was accepted as 220. Optional GPU timer removal
and runtime wait changes had only noise-sized gains and were removed; interrupt
polling crashed during load. A 128-row prefill experiment changed output IDs
and was rejected. More aggressive draft-head and draft-dense alternatives lost
acceptance or end-to-end speed and are absent from the selected source.

See `acceptance-220.json`, `q8s-ab-v26-validation.json`, the comparison JSONs,
unit-test logs and raw `q8s-ab-v26-*` / `*-q8s-ab-v26` directories in the
November 4 evidence directory. The original 200-token/s binary is retained
unchanged for reproduction of that earlier checkpoint.
