# November 4 decode follow-up

The latest validated candidate is V11: **214.73470 tokens/s** on the unchanged
single-request Swift IQ3_S panel (code/prose/edit, 2048 outputs each, twice;
context allocation 32768). The aggregate uses total tokens divided by engine
decode time, excluding warmup. **220 remains open.**

V11 minimizes per-pool padding while retaining the cache byte budget and V10
packed Q2 arithmetic. It uses only tensor geometry, not benchmark routes. The
same binary with only this option disabled measured 213.84493 tokens/s
(+0.42%). All six input/output streams and draft acceptance match, both
processes exited normally, and all decode major-fault counts are zero.

```sh
bash bench/results/2026-11-04-rx7900xtx-decode220/reproduce-cache-padding-v11.sh NEW_OUTPUT_DIRECTORY
```

The tested binary is `build-tensor-parralize-baseline/strata-split-cache-padding-v11`,
SHA256 `006e535eb09e75b7a7fc61c781afd7f0c1d085e7aa1beacc6801f0aace1c4b2b`.
The option `FNX_STATIC_CACHE_MIN_PADDING=1` requires two cache groups and is
disabled by default. Cache slots increase from 1292 to 1402 with unchanged
expert RAM (11.625 GiB), including space temporarily borrowed from unused K/V.

The layout test passes 768 generated cases, default-compatibility and real-model
fixtures; 21945 K/V layout checks pass. The full quality suite passes the
1326 reviewed code checks and exact reference comparisons for near-context,
Chinese generation, fixed seeded sampling with penalties, forced cache churn,
and prefetch probing. See `cache-padding-v11-same-binary.json` and
`cache-padding-v11-validation.json`; raw manifests and engine logs are retained.
A RAM11.5 experiment measured 214.91535 tokens/s, only +0.08%, and is retained
as evidence rather than selected over the configuration with more VRAM margin.

## V10 checkpoint

The latest validated candidate is V10: **214.14082 tokens/s** on Swift IQ3_S,
single request, context allocation 32768, code/prose/edit 2048 outputs each
with two repeats. The metric is total output tokens divided by engine decode
time, excluding warmup. **220 remains open.**

V10 adds lossless packed Q2_0 down weights to V7. The same binary with this
feature disabled and the original RAM budget measured 210.89848 tokens/s
(+1.54% for V10). All six input/output and draft-acceptance streams match;
both processes exited normally with zero decode major faults. The production
service and configuration are unchanged.

```sh
bash bench/results/2026-11-04-rx7900xtx-decode220/reproduce-214.sh NEW_OUTPUT_DIRECTORY
```

The tested binary is `build-tensor-parralize-baseline/strata-split-packed-q2-v10`,
SHA256 `bf1e35c8d08850128c0a8eae2b1306cbea1c19f3e318614629831ae8ab30ef01`.
A rebuilt binary can be selected with `TP_EXE`. Full model paths and settings
are in `run-heads-experiment.sh` and each raw `results.json` manifest.
`packed-q2-v10-same-binary.json` records performance;
`packed-q2-v10-validation.json` records the completed quality checks. Raw
outputs and engine logs are retained in the corresponding directories.

Focused CMake test targets are `split_moe_q2_down_test`, `split_moe_q2_mma_test`
and `split_gguf_release_test`. The reproduced test logs record 1048576 exact
dots, 3072 decode/guard cases, 648 Q8 prefill cases, 648 WMMA prefill cases, and
host conversion/lifetime checks. Quality validation includes 1326 reviewed
code checks, near-context generation, Chinese prompts, fixed seeded sampling,
forced cache churn, and the prefetch probe. All match their prior references.

The `run-*.sh` and `validate-*.sh` files record fixed original artifact names;
copy them with fresh destinations before repeating. `reproduce-214.sh` accepts
a new destination directly. The measurements below retain the earlier V7
checkpoint for comparison.

## V7 checkpoint

The 220 tokens/s target remains open. The validated V7 candidate measured
210.93794 tokens/s on the same Swift IQ3_S, single-request, 32768-context panel
as the accepted 200 tokens/s recipe. Code, prose and edit each generate 2048
tokens, twice; the aggregate is total output tokens divided by engine decode
time, excluding warmup.

V7 combines attention head partitioning, lending unused KV capacity to the
static expert cache, and gate/up cache fill inside the IQ compute launch.
The same V7 binary with only the last feature disabled measured 209.34024.
All six input/output streams and draft acceptance counts match, with zero
decode major faults in both panels. The production service is unchanged.

Run on the recorded host from `/home/john/Strata`, using a fresh destination:

```sh
bash bench/results/2026-11-04-rx7900xtx-decode220/reproduce-211.sh NEW_OUTPUT_DIRECTORY
```

The immutable tested binary is
`build-tensor-parralize-baseline/strata-split-gu-cache-v7`, SHA256
`3cad60618315c9fbe249f38731b655c821997246fc27b7244b40bdbaebd0cffe`.
Full model/tokenizer/pack paths and runtime settings are in
`run-heads-experiment.sh` and each `results.json` manifest. A rebuilt binary
can be selected with `TP_EXE` and must be validated independently.

`gu-cache-v7-same-binary.json` compares the full panels.
`gu-cache-v7-validation.json` records completed-code checks (1326 passed),
near-context generation, held-out Chinese prompts, seeded sampling, forced
cache churn and the prefetch probe. Their raw manifests, outputs and engine
logs are in the corresponding directories. GPU unit-test logs record 12288
new GU/cache checks and 6144 existing cache checks.

The `run-*.sh` and `validate-*.sh` files record the original experiment
sequence and use fixed artifact names; they should be copied and given fresh
destinations before repeating. `reproduce-211.sh` accepts a new destination
directly. Older controls and rejected prototypes remain on the host; they
are not part of the selected recipe. Kernel profiling is diagnostic only:
its shutdown was killed by the harness timeout, so it is excluded from
performance acceptance.
