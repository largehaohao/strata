# Final-row draft-head validation

These are the exact experiment scripts used on `john@192.168.1.102` in
`/home/john/Strata`. The retained binary is
`build-tensor-parralize-baseline/strata-split-mtp-last-head`; its SHA-256 is
`b051993f8a7793706ee848959226d99876263aa0c9d98630e981d88b2db5da3e`.
The source delta from `a82d4e24` is `../mtp-last-head-source.patch`.

Scripts assume they are copied into `build-tensor-parralize-baseline/` and
launched from the repository. Output directories must be new; choose new
names in the scripts before rerunning. Do not overlap GPU measurements.
`tp_sample_check.py` is a fixed-parameter copy of `tools/tp_decode_bench.py`
with temperature 0.7, top-k 20, top-p 0.9, repeat penalty 1.05, frequency
penalty 0.05, presence penalty 0.03, last-n 128 and seed 20261104. It must
stay directly under the build directory so its repository-root lookup works.

- `bench-mtp-last-head.sh`: last-row option on/off, 32 copy blocks.
- `repeat-mtp-last-head.sh`: 16 copy blocks, then independent 32-block repeat.
- `validate-mtp-last-head.sh`: independent 16-block repeat, reviewed code,
  near-context, Chinese held-out, and sampled/penalized on/off checks.
- `check-mtp-last-head.py`: exact input/output/finish and draft-count checks,
  zero decode major faults, and reviewed-code checks. Warmup is included in
  equality checks, excluded from rate claims.
- `final-tp-metrics.py`: require both final independent panels to exceed
  200 tokens/s with the expected binary and recipe, then assemble acceptance.

The validation script executes only the fixed reviewed code tasks, after
matching all their input/output token IDs against `quality-split`. It does
not execute the performance or Chinese generated code. GPU tests are in
`src/split/tools/mtp_head_gpu_test.cu`; build target
`split_mtp_head_gpu_test`, run on both physical cards. The final GPU log is
`../mtp-last-head-gpu-checks.txt` (768 checks).

For a new normal acceptance run, use the simpler `../reproduce-200.sh`.
