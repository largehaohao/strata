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
- `FNX_IQ_REUSE=1` decodes an IQ2_S, IQ3_XXS or IQ3_S weight group once for
  all token rows in an expert job. Its integer divisions, scaling and reduction
  order match the existing Q8 dot product. Other weight types keep their path.
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
| Same, IQ weight reuse, two runs | 130.0–141.4 | 110.2–120.1 | 139.3–145.2 |
| IQ reuse plus cold-only staging, two runs | 133.7–149.1 | 111.0–124.9 | 142.6–150.5 |
| Same, embedding refresh after load, two runs | 148.1–149.0 | 123.7–125.0 | 149.6–150.4 |
| Cached host weights, direct reads, embedding refresh, two runs | 155.7–156.1 | 131.5–131.6 | 158.3–158.7 |

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
Neither setting is adopted by the reproduction recipe.
`FNX_RING_GRID=80` with cold-only staging measured 138.7–149.1 / 116.8–125.4 /
145.0–150.6. Its second-pass GPU verify times differ from the default grid by
less than 0.5%; it is not adopted as a demonstrated optimization.
The existing prefill FP16 matrix kernel was also slower than IQ reuse for
the representative small expert jobs in a separate synthetic trial. It is
not selected for decode.

The split loader already converts 675 dense/small tensors to Q8_0/F32 for this
GGUF. Thus, split and the native engine do not have the same arithmetic or
necessarily the same output tokens. The checks above compare split variants.
They do not establish quality equivalence with the native BF16 path. The
2048-token performance replies hit their length limit.

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

A four-second ROCprofiler sample showed matrix staging at 23.15% and the
window-sum peer kernel at 15.99% of summed kernel duration. This includes
overlap and waiting across both cards; it is not a wall-time speedup estimate.
The raw trace is retained with the experiment outputs. Profiler timings are
excluded from the performance table.
With IQ reuse, another sample still puts staging at 23.99% and peer window sums
at 16.38% of summed kernel time. The IQ gate/up calls average about 76–82 us for
seven-row windows; reducing this alone cannot close the remaining decode gap.

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
FNX_STATIC_EXPERTS=1 FNX_EXPERT_RAM_GIB=14 FNX_MOE_STAGE=0 FNX_PF_ROWS=256 \
FNX_IQ_REUSE=1 FNX_STATIC_HOST_CACHE=1 FNX_EMBED_WARM_AFTER_LOAD=1 \
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
- Both engines passed the three completed-code tasks described above.
- CUDA has not been built or run here. No SYCL files are changed.
- Remaining: the 200 tokens/s target, long-context acceptance, broader quality
  and workload coverage, and integration into launch policy.

The implementation builds on Strata's existing sFNX split engine. Reference
projects considered include [vLLM's custom all-reduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py),
[Paiton](https://github.com/Eliovp-BV/paiton-vllm-plugin), and
[Strata dual 3090](https://github.com/q8atnight/strata-dual-3090).
