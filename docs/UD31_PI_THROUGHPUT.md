# UD 3.1 Pi task completion

The November 9 selected build adds
[bounded lookup and exact sampling kernels](UD31_TP_OPT.md#pi-lookup-and-exact-sampling-kernels)
to the earlier short GSU3 optimization. Two independent task pairs using
previously unused seeds complete in **274.57 seconds versus 301.43 seconds**,
an **8.91% reduction**, with all 16 task acceptances passing. This meets the
further 5% task-time target for the declared representative panel. The final section records all six pairs,
including the slower pair and changes in generated length.

On November 9, the daily service also received a Pi response-recovery fix.
Its forced greedy sampling defaults were removed, and a repeated token inside
thinking can trigger one bounded continuation when `reasoning_close_retry` is
enabled. The November 7 measurements below retain their original sampling
configuration; they do not measure throughput after that change. The final
November 9 comparison uses the corrected model-card sampling defaults.
The incident and validation are recorded in
`local-tuning/2026-11-09/pi-truncation/README.md`.

On two RX 7900 XTX cards, the November 7 selected build completed the four-task
Pi panel in **103.38 seconds with a 512 MiB conversation cache**, versus
**123.11 seconds with the cache disabled**. The **16.02% reduction** exceeds
the 5% target for this representative panel. Both runs passed all task checks.

Measured on 2026-11-07 with Qwen3.8-Flash-Next Unsloth UD IQ2_XS, a Ryzen
7700, 62 GiB system memory, and a 262144-token context. Measured binary
SHA256: `832ed3729d89738bd21293cd47219d90ce21b0e1b14c168ed64640a198ed5a37`.

The current build includes the selected QSA and HC row partitions, recorded
in [UD31_TP_OPT.md](UD31_TP_OPT.md#selected-two-card-hc-row-partition).
Both remain enabled throughout the latest cache comparison. Earlier pairs
and their binaries are kept below; none is an exact historical-task replay.

## Workload and acceptance

The supplied Pi export contains metadata for 7345 requests in 116 sessions,
with a 96.74% aggregate token-cache hit fraction. Median context, fresh
input and output lengths are 65716, 185 and 316 tokens. Its Strata subset
has 4554 requests in 49 sessions, with median context 70915 tokens and median
fresh input 172 tokens. Original task text is absent, and the export covers
other models, so it cannot provide an exact historical-task replay or a
same-day UD baseline.

The executable panel uses real Pi read, write, edit and bash calls in one
conversation. It starts with about 74K tokens of repository reference data,
then completes Python utilities, interval merging, an order-summary
refactor, and a reservation workflow. Independent Python and TypeScript
checks validate the results. File-scope checks reject edits to tests and
unrelated files. Runtime Python bytecode is excluded from source edits.

Both arms use identical initial request bytes, workspace paths, task order,
greedy sampling, medium thinking, a 32768-token output ceiling, model and
context capacity. Failed tasks receive acceptance feedback and may retry,
up to three attempts. Total time includes failures, retries, tool execution
and final acceptance. Server startup and the same two warmup requests are
excluded. No builds or GPU tests ran alongside the timed panels.

## Results and limits

Each pair uses the same binary with only `--conversation-cache-mib` changed.
The second and fourth pairs run cache enabled first. The third uses the
coalesced-QSA binary and runs cache disabled first.

| Pair | Cache disabled | Cache 512 MiB | Reduction | Final acceptance |
| --- | ---: | ---: | ---: | --- |
| Initial controlled pair | 122.93 s | 93.84 s | 23.66% | 4/4 in each arm |
| HC-up binary, reverse order | 124.47 s | 107.12 s | 13.94% | 4/4 in each arm |
| Coalesced-QSA binary | 121.08 s | 94.26 s | 22.15% | 4/4 in each arm |
| Current QSA + HC row partition, reverse order | 123.11 s | 103.38 s | 16.02% | 4/4 in each arm |

The initial pair used an experimental binary with shared-expert fusion
disabled in both arms. The second pair uses the qualified HC-up binary
`971bc524633b9e9e7811bcfa8284417967e88a02b99de91c2bdde7ac6cc2fca2`.
The cache setting itself introduces no GPU kernel changes.

Generated text and tool trajectories changed. This is a complete-task
measurement, with no claim of output identity or raw decode acceleration.
The coalesced-QSA pair issued 28 and 35 requests, generated 20944 and 14660
tokens, and achieved token-cache hit fractions of 93.79% and 97.23%.
Cache-enabled median fresh input and output were 139 and 317 tokens.
Both arms had zero HTTP errors and identical initial request hashes.

In the coalesced-QSA pair, the aggregate benefit is concentrated in the Python
task. The three shorter tasks became slower:

| Task | Cache disabled | Cache 512 MiB |
| --- | ---: | ---: |
| Python utilities, including retries | 101.75 s | 58.76 s |
| Interval merging | 4.77 s | 8.47 s |
| Order-summary refactor | 6.03 s | 12.93 s |
| Reservation workflow | 8.53 s | 14.09 s |

Without checkpoints, a serialized conversation that differs from the held
sequence can force a full prefix read. One baseline request reread about
76K tokens. Checkpoints recover an earlier matching prefix, but saves and
additional prefill segments cost time. Model trajectories also differ.
These pairs do not establish a speedup for every task or every Pi session.

## Context and output validation

On the earlier HC-up build with the selected cache setting, cold prompts of
204800 and 261888 tokens both
returned `1517` for 37 × 41. Their follow-ups returned `1683` for 17 × 99
and reused 204804 and 261892 tokens. All four requests completed without
engine errors, OOMs or major page faults.

The near-limit follow-up had 261923 prompt tokens and a 64-token output
ceiling, within the configured 262144-token capacity. This checks capacity
and continuation; long-generation throughput at that length was not tested.
Peak engine resident memory during validation was about 46.6 GiB, including
mapped model data, with no engine swap. The cache budget governs retained
host states; parking a full conversation can temporarily allocate more
before eviction.

The coalesced-QSA build separately passed the 261888-token request and its
follow-up with the same answers and reused-token count. After the latest
task A/B, the restored daily service passed ordinary and thinking requests,
returning the correct answers `323` and `1683` at 262144-token capacity.

## Applied setting and experiments

For this cache tuning step, the local `strata-unsloth-ud-iq2_xs.json` changed
only `--conversation-cache-mib` from `0` to `512`. The model, two GPUs, 256K
context, K=320 fusion, HC-up optimization and sampling settings are retained.
Set that argument back to `0` to revert this tuning choice.

An isolated shared-expert/cache fusion experiment passed 8192 GPU byte,
cache-copy, guard and graph checks. It was not selected or applied. A
4096-token checkpoint-stride experiment improved short-prefill throughput,
but its completed-task total was 112.51 seconds, slower than the ordinary
512 MiB cache runs. It also remains an isolated experiment.

Evidence, API captures, acceptance logs, pilot failures and the local
rollback config are in `local-tuning/2026-11-06/ud31-pi-throughput/`.
`qualification.json` records the first two A/B pairs and their limits. The
coalesced-QSA pair, complete captures and service restoration are under
`local-tuning/2026-11-06/ud31-residual-opt/`; `pi-current-pipeline.json`
records that result. `pi-hc-selected-pipeline.json` records the current build's
cache comparison, and `pi-workload-evaluation.json` records the attachment
hash and selected metadata. Private configs contain credentials and must not
be published. Initial pilots with
failed task scope or a corrected harness are retained and excluded from
successful task-completion totals. Failures and retries within the final
protocol are included.

## Later long-prefill query partition

The next selected binary also partitioned eligible long-prefill QSA queries
between the cards. Its SHA-256 was
`73056cb915bea58057511709761323a92d016fe06cc1996b589bdfcee80b31c5`.
The cache setting stays at 512 MiB. The same-binary four-task off/on comparison
finishes in 95.16 and 91.37 seconds, a 3.98% reduction. Both arms pass 4/4
acceptance checks, issue 35 requests and have zero HTTP errors. The enabled
arm needs a second attempt on the reservation task; that time is included.

Generated output differs: 14753 versus 13998 tokens. Ignoring generated
tool-call IDs, requests first differ at request 20 (zero-based), where the
test runner returns different elapsed times. Later model trajectories differ.
This result does not establish a further 5% task gain from the QSA optimization.
The separate frozen 23-request replay keeps every output and counter identical
and reduces API time by 0.41%; it excludes tool and agent time.

This selected optimization has a repeated 2.7–2.8% benefit on 200K cold prefill.
Its near-limit request, cached continuation, cancellation/recovery and ordinary/
thinking service checks pass. The cache comparisons, including the current-build
repeat below, provide the evidence for this representative panel's 5% target.
The cache comparison was not repeated at the QSA selection step. See the
[query-partition measurements](UD31_TP_OPT.md#selected-two-card-qsa-query-partition)
and `ud31-residual-opt/qsa-query-late-pi-tasks-pipeline.json`.

## HC row partition retained with the Pi setting

The current executable also partitions HC-down rows for main-model prefill
chunks of 3072–4096 tokens. The corrected full-build SHA-256 is
`832ed3729d89738bd21293cd47219d90ce21b0e1b14c168ed64640a198ed5a37`.
It retains the QSA options, K=320 fusion and 512 MiB host conversation cache.

The same-binary HC off/on comparison improves long prefill by 1.25–1.65% at
32K, 64K, 128K and 200K. All seven long and 23 frozen Pi requests match in
outputs and counters. Pi small prefill changes by +0.22%, small decode by
-0.11%, and frozen API time falls from 73.771 to 73.488 seconds (-0.38%).
This is not an additional 5% complete-task gain. The
[HC measurements and checks](UD31_TP_OPT.md#selected-two-card-hc-row-partition)
include the successful near-limit continuation and cancellation/recovery test.

## Cache comparison on the current selected build

The latest pair uses the corrected HC binary above, with both HC and QSA
enabled. It runs 512 MiB first, then disables the cache in a fresh process.
The first request hashes match, and model, workspace paths, task order,
thinking settings and output limits are unchanged. Each arm receives the
same two excluded warmup requests. No build or GPU test overlaps timing.

| Metric | Cache disabled | Cache 512 MiB |
| --- | ---: | ---: |
| Complete-task time | 123.107 s | 103.384 s |
| API time | 120.506 s | 100.834 s |
| Engine prefill time | 34.132 s | 25.301 s |
| Engine decode time | 85.046 s | 73.984 s |
| Requests | 29 | 36 |
| Generated tokens | 21545 | 16627 |
| Token-cache hit fraction | 94.03% | 97.29% |
| Final acceptance | 4/4 | 4/4 |
| HTTP errors | 0 | 0 |

The cache-enabled median fresh input and output are 174.5 and 297 tokens.
The disabled arm rereads a 76140-token prefix after its initial 74160-token
request. With the cache enabled, the largest subsequent fresh input is 1757
tokens. This is evidence that retained host states avoid a full prefix read.
Generated output and task trajectories also change, so the entire time saving
cannot be attributed to prefill alone and is not a raw decode speed gain.

| Task | Cache disabled | Cache 512 MiB |
| --- | ---: | ---: |
| Python utilities, including retries | 101.12 s | 58.61 s |
| Interval merging | 4.82 s | 9.28 s |
| Order-summary refactor | 7.43 s | 18.55 s |
| Reservation workflow | 9.73 s | 16.94 s |

The disabled Python task needs two attempts; every other task takes one.
All attempts, tools and independent acceptance checks count toward the total.
As in the earlier comparison, the three shorter tasks become slower. This
four-task result supports the aggregate 5% target for the representative
panel, not a per-task guarantee or a measurement of the original 116 sessions.
The attachment contains no original task text for that broader claim.

The runner restores the current daily configuration with the 512 MiB cache,
both GPUs, 256K context, QSA, HC and K=320 options. Ordinary and thinking
requests return 323 and 1683 without restarting the engine. The record is
`ud31-residual-opt/pi-hc-selected-pipeline.json`, with full API and acceptance
captures in `pi-hc-selected-cache{0,512}-r0/` and service checks in
`production-after-pi-hc-selected/summary.json`.

## November 9 model-card sampling task comparison

The new goal starts from executable
`832ed3729d89738bd21293cd47219d90ce21b0e1b14c168ed64640a198ed5a37`,
with the 512 MiB cache and earlier QSA, HC and K=320 settings already enabled.
It does not compare against the cache-disabled or greedy November 7 runs.
The selected short GSU3 increment alone previously reduced complete-task time
by only 2.32%. The new candidate retains it and adds `FNX_LOOKUP_SEARCH=32`,
`FNX_TOPK_PARTS=16` and `FNX_QSA_SCORE_GRID=128`.

Each arm runs the same four actual Pi coding tasks and tools described above.
Both arms keep model-card sampling, medium thinking, six MTP drafts, lookup
limit 15, a 32768-token output ceiling, context 262144 and the 512 MiB cache.
Initial request bytes match within each seed pair. Later text and tool results
can differ. All failed attempts, retries, tool execution and independent final
acceptance count toward total time. Server startup is outside the task timer;
there are no excluded warmup requests. No build or GPU test runs alongside
these measurements.

The first four pairs use the combined prototype. Their seeds and alternating
order are declared before timing. The acceptance rule requires a pooled
reduction of at least 5%, three of four pairs improving, all 32 final task
acceptances and no HTTP errors. That panel passes. For the formal build,
seeds 505 and 606 and their acceptance rule are declared before observing the
results. After capacity checks, these two independent pairs must also improve
pooled task time by at least 5%, pass all 16 final task checks, have no HTTP
errors, and retain a reduction of at least 5% across all six pairs. They pass
those requirements.

| Seed, build and order | Goal-start reference | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| 101, prototype, reference first | 219.070 s | 122.270 s | 44.19% |
| 202, prototype, candidate first | 110.021 s | 94.052 s | 14.51% |
| 303, prototype, reference first | 141.904 s | 176.590 s | -24.44% |
| 404, prototype, candidate first | 146.082 s | 122.624 s | 16.06% |
| First four pairs | 617.077 s | 515.536 s | 16.46% |
| 505, formal build, candidate first | 145.494 s | 124.949 s | 14.12% |
| 606, formal build, reference first | 155.938 s | 149.618 s | 4.05% |
| Independent formal-build pairs | 301.432 s | 274.567 s | 8.91% |
| All six pairs | 918.509 s | 790.102 s | 13.98% |

All 48 final task acceptances pass across 52 attempts, with 26 attempts per
arm and zero HTTP errors. The independent candidate needs one retry; its time
is included. Five pairs improve, but seed 303 becomes 24.44% slower. These
results do not establish a speedup for every task or session.

Generated length and trajectories account for much of the total difference.
Across all six pairs, output totals change from 161873 to 139711 tokens and
request counts from 246 to 216. Aggregate measured Pi decode throughput is
223.19 versus 223.86 tok/s, almost unchanged. The separate fixed-input
diagnostic increases decode throughput by 10.44%; it is not a 10.44% decode
gain over this actual-task panel. The task result measures time to accepted
completion and does not isolate a GPU-kernel speedup. The supplied export has
metadata without original task text, so this remains a representative panel,
not an exact replay of the user's historical sessions.

The formal binary has SHA-256
`a1e3e8417002394ca6a6c27b2c0cde142cd77c98840d8b3704de11f7e7d08a7b`.
Its default path matches the earlier selected binary on the ordered seeded
request sequence. With the new controls enabled, its outputs and native
counters match the prototype. Exact GPU tests, near-limit sampled continuation,
cancellation recovery and seven installed-Pi provider cases pass; details are
in the [kernel measurements](UD31_TP_OPT.md#pi-lookup-and-exact-sampling-kernels).
The lookup proposal policy can change fixed-seed sampled text. Target sampling
parameters, output budgets and the existing rejection calculation are retained.

The local `strata-unsloth-ud-iq2_xs.json` now points to the immutable qualified
formal binary and enables the three controls above. Short GSU3 and all earlier
accepted controls remain enabled. Both GPUs, 256K context, model-card sampling,
cache capacity and the Pi response-recovery fix are retained. The daily 8081
service remains stopped as it was before tuning; the next normal start uses
this configuration. To undo this selection, restore the private
`before-pi-goal.private.json` backup. That backup includes credentials and must
not be published.

Evidence is in `local-tuning/2026-11-09/ud31-pi-task-opt/`:
`lookup-search-task-pipeline.json` contains all four prototype pairs;
`formal-pi-heldout-plan.json` and `formal-pi-heldout.json` contain the independent
protocol and results. `formal-pi-model-checks.json`, `formal-pi-qualification.json`
and `formal-pi-deployment.json` record the build comparison, recovery checks
and applied selection. Full requests, responses, tool traces and acceptance
logs are retained locally, including unsuccessful earlier experiments.
