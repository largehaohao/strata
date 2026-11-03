# Dual RX 7900 XTX decode experiments

Work in progress on `tensor-parralize`, based on main
`ffa80d73133ea5c548aefe57889a16e8085a938f`. The target is **200 tokens/s for one
request on Swift-1.5 IQ3_S**. This target has **not** been reached.

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

`FNX_STATIC_CACHE_FUSED_DN=64` fills missing down weights inside the IQ gate/up
launch: 64 prefix blocks copy bytes while the remaining blocks compute the
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
not a general prefill throughput claim. The **200 tokens/s target remains open**.
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
203.95 tokens/s; the broader 2048-token panel still falls short of 200.

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
`quality-cache-groups2`. The broader 200 tokens/s target remains open.

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
broader 200 tokens/s target remains open.

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
FNX_STATIC_EXPERTS=1 FNX_EXPERT_RAM_GIB=15.5 FNX_STATIC_CACHE_SLOTS=1024 \
FNX_STATIC_CACHE_GROUPS=2 \
FNX_MOE_STAGE=0 FNX_PF_ROWS=256 \
FNX_IQ_REUSE=1 FNX_IQ_GROUP=16 FNX_MOE_PRE_ROWS=1 \
FNX_STATIC_HOST_CACHE=1 FNX_EMBED_WARM_AFTER_LOAD=1 \
FNX_RELEASE_COPIED_WEIGHTS=1 FNX_KV_CONTEXT_ONLY=1 \
FNX_HC_ROW_GROUP=64 FNX_STATIC_CACHE_FUSED_DN=64 \
FNX_Q8S_ROW_GROUP=2 \
FNX_MTP_EXPERTS=q4_0 \
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
- CUDA has not been built or run here. No SYCL files are changed.
- Remaining: the 200 tokens/s target, long-context acceptance, broader quality
  and workload coverage, and integration into launch policy.

The implementation builds on Strata's existing sFNX split engine. Reference
projects considered include [vLLM's custom all-reduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py),
[Paiton](https://github.com/Eliovp-BV/paiton-vllm-plugin), and
[Strata dual 3090](https://github.com/q8atnight/strata-dual-3090).
