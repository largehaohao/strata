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
- `FNX_MOE_STAGE=1` copies each selected expert matrix into reusable VRAM
  scratch before its quantized dot products. Gate/up and down reuse the same
  scratch in stream order. Copies preserve every weight byte. The normal
  `DW::ep` interface is used by the compute kernels.
- `FNX_DECODE_TIMING=1` reports the existing verify, draft and launch timers.
  `FNX_PROFILE_EXIT=1` lets profiler exit callbacks write their traces. It is a
  diagnostic option; normal operation keeps the existing fast exit.

`FNX_PF_ROWS=256` reduces prefill scratch compared with the existing default
4096. This allowed the cold allocation to fall from 16 to 14 GiB. With staging,
a 32768 context and MTP, only about 288 MiB remained free per card at startup.
This setting has not been validated for the configured 204800 context.

On this host, `HSA_FORCE_FINE_GRAIN_PCIE=1` allowed both directions of the
existing peer probe to pass. The runtime still checks the actual links before
using peer push. The startup log now reports host exchange only when it is
actually selected.

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
| Split, static 16 GiB cold, peer push | 71.9–77.0 | 69.2–75.1 | 70.1–72.4 |
| Same, with matrix staging | 99.1 | 92.3 | 100.8 |
| Staging, 14 GiB cold, prefill rows 256 | 109.0 | 99.6 | 110.8 |
| Same, calibrated decode residency, two runs | 129.6–134.3 | 113.7–115.8 | 136.1–137.8 |

Values are tokens/s from the engine's reported decode time. The benchmark
also records client time, time to first token, exact input/output token IDs,
draft acceptance, binary SHA-256, source diff fingerprint and environment.
Warmup is excluded. The calibrated split's weighted rate is 127.13 tokens/s.
This does not establish an overall gain over the native engine.

The exact token records and the calibration profile are in
[`bench/results/2026-11-03-rx7900xtx-tensor-parallel`](../bench/results/2026-11-03-rx7900xtx-tensor-parallel/).

The profile was collected from separate Rust, Go, SQL, Chinese systems and
graph-algorithm prompts, 1536 generated tokens each. The measurement prompts
were not used to train it. Decode counts select residency; prompt counts are
zeroed in the derived profile. Engine warmup counts are present in the
calibration. This is workload calibration, not a claim that the profile suits
every workload.

Staging and residency changes produced the exact same 2048 output token IDs
on all three measurement prompts, including both calibrated repeats. A trial
that skipped staging for hot experts was slower on these tasks and was removed.
Two gate/up kernel trials were also removed: a bounded persistent grid passed
3072 exact direct/graph checks but was slower; cooperative loading of job
descriptors passed 1024 checks but had no consistent benefit. Their experiment
logs are retained; neither changes the shipped compute kernels.

The split loader already converts 675 dense/small tensors to Q8_0/F32 for this
GGUF. Thus, split and the native engine do not have the same arithmetic or
necessarily the same output tokens. The checks above compare split variants.
They do not establish quality equivalence with the native BF16 path. The
2048-token replies hit their length limit; they are not completed-code tests.

A four-second ROCprofiler sample showed matrix staging at 23.15% and the
window-sum peer kernel at 15.99% of summed kernel duration. This includes
overlap and waiting across both cards; it is not a wall-time speedup estimate.
The raw trace is retained with the experiment outputs. Profiler timings are
excluded from the performance table.

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

Set `MODEL`, `MTP` and `TOKENIZER` to the installed GGUF, MTP directory and
tokenizer directory. Set the SDK library paths for the child process:

```sh
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib"
HIP_VISIBLE_DEVICES=0,1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
FNX_STATIC_EXPERTS=1 FNX_EXPERT_RAM_GIB=14 FNX_MOE_STAGE=1 FNX_PF_ROWS=256 \
python3 tools/tp_decode_bench.py \
  --exe build-tensor-parralize/src/split/strata-split-2 \
  --model "$MODEL" --mtp "$MTP" --tokenizer "$TOKENIZER" \
  --context 32768 --tokens 2048 --repeats 2 --out /tmp/strata-tp-new-run
```

The output directory must be new. For calibrated placement, also set
`FNX_EXPERT_PROFILE` to the measured split-format count file. It is different
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
- Static placement: all 256 partitions of eight experts passed, including
  empty hot/cold banks and invalid-input rejection; tested with `NDEBUG` too.
- Staging: 288 byte-for-byte direct/graph checks passed on both cards, with
  changing routes, mixed host/device pointer tables, contiguous weights,
  several strides and guard bytes.
- Existing split Python suite: 52 tests passed.
- CUDA has not been built or run here. No SYCL files are changed.
- Remaining: the 200 tokens/s target, long-context acceptance, completed-code
  quality checks, broader workload coverage and integration into launch policy.

The implementation builds on Strata's existing sFNX split engine. Reference
projects considered include [vLLM's custom all-reduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py),
[Paiton](https://github.com/Eliovp-BV/paiton-vllm-plugin), and
[Strata dual 3090](https://github.com/q8atnight/strata-dual-3090).
