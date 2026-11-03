# November 4 decode follow-up

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
