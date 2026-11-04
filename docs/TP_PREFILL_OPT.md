# Dual RX 7900 XTX prefill experiments

This work is on `tensor-parralize`, based at `1b2e778c`, with the existing Swift IQ3_S model and a requested context of 200768. All measurements below are single-request, reset/cold-prefix runs. Experimental controls use the existing server JSON `env` and `exe` fields.

## Reusing the strata branch

Reviewed `strata` at `5da7b379` (including `a7bac5da`, `6205223c`, and `5445ce49`). Its reported agent-prefill gains primarily concern short increments on one GPU. Its historical long-prefill comparison was approximately unchanged: 4K 3192, 16K 3677, 32K 3682 tokens/s. Those are not dual-GPU TP acceptance results.

| Change in strata | Applicability to the TP engine |
| --- | --- |
| Actual-route expert uploads, batched HIP copies, copy/compute events | Adapted as experimental `FNX_PF_STATIC_STAGE=3`. Retains all routed experts and immutable TP placement; overlaps actual cold-expert copies with the shared expert. |
| `hip_batch_copy.hpp` | Imported unchanged from `5da7b379`; `STRATA_PREFILL_HIP_BATCH_COPY=1` attempts the runtime API and falls back to individual copies on the same stream. |
| Compact ring and previous-chunk forecasts | The current TP stage already copies only requested cold experts and reuses them across chunks of a pass. Forecasting is not enabled: it adds transfers and needs separate evidence. |
| Active expert launch grid | Native batches previously used the whole layer's upper bound repeatedly. TP already uses one whole-layer job list; introducing host synchronization just for its exact grid needs a separate performance case. |
| Direct IQ sign generation and parallel F16 GU epilogue | Adapted the direct-sign idea for IQ3_S, IQ3_XXS and IQ2_S in the opt-in wider TP kernel. Keeps TP's original FP16 activation and weight rounding; native kernels use a different activation/scale path and cannot be copied unchanged. |
| Rare-expert CPU sharing | The branch explicitly restricts it to a single GPU/stage and an idle CPU pool. Not enabled on TP. |
| Compact PLE direct reads | TP uses a separate embedding path. Existing profiles do not show host embedding upload as the main long-prefill cost. Not ported. |
| Wider decode QSA/GR windows, fine draft cost policy | Separate decode work, not a demonstrated solution for long TP prefill. Not merged as part of this experiment. |

Mode 3 reads back each GPU's actual router counts, after issuing both GPUs' router work. A per-card readiness event orders the independent copy stream after preceding users of the stage. The shared expert executes while copies are queued. Routed experts wait for the copy completion event. An expert becomes ready in the host plan only after that wait has been enqueued. Empty batches need no wait. The stage is reused across chunks, then reset for the next layer. Every original expert contribution and arithmetic path remains present.

Mode 2 remains the GPU-copy control. Mode 0 remains the original default. The change does not alter the production JSON or launcher.

The first DMA experiment (v13) passed 288 byte/route/reuse/guard checks on both the batch API and its individual-copy fallback. All warmup and formal input/output IDs, accepted/offered counts, and finish reasons matched v12 in both configurations below. It did not produce a clear throughput gain: CPU enqueue time grew from tens of milliseconds to nearly the whole prefill duration because every chunk/layer waits for route readback. Mode 3 is an experimental control, not a selected serving configuration.

Mode 4 experiments with the same copy/shared-expert overlap while keeping route decisions entirely on the GPU. The copy stream waits for the router, fills only newly selected experts, then signals the routed-expert consumer. `FNX_PF_COPY_BLOCKS` controls copy blocks per expert (1 by default for mode 4; the previous path retains 8). Updated byte/event tests passed 432 cases. Model tests with both 1 and 8 blocks preserved all input/output IDs and draft counts. Neither delivered a consistent speedup; mode 2 remains selected. With 8 blocks, measured times were 1327.6 / 3943.5 / 7959.5 ms, versus v12's 1325.2 / 3951.7 / 7950.8 ms.

## Current measured reference

Fixed input lengths are 4015 / 16015 / 31956 tokens, with 24 greedy output tokens. These are the existing native-protocol benchmark cases, **not** the final exact-length API benchmark.

| Candidate | RAM / chunks per pass | 4K | 16K | 32K |
| --- | --- | ---: | ---: | ---: |
| v12, direct-layout attention and register HC up/mix | 19 GiB / 4 | 3030 | 4053 | 4019 |
| v12, same kernels | 20.5 GiB / 8 | 2952 | 4028 | 4166 |
| v13, actual-route batched DMA | 19 GiB / 4 | 2997 | 4044 | 3996 |
| v13, actual-route batched DMA | 20.5 GiB / 8 | 2915 | 4016 | 4173 |
| v15, residual addend buffer reuse | 18.5 GiB / 4 | 3055 | 4092 | 4037 |
| v18, wider MoE tiles and exact packed IQ dequantization | 18.5 GiB / 4 | 3109 | 4145 | 4111 |
| v19, GDN convolution/normalization wave rows | 18.5 GiB / 4 | 3172 | 4317 | 4267 |
| v20, compact GDN preparation scratch | 18.5 GiB / 4 | 3265 | 4415 | 4371 |
| v21, phase scratch reuse and more resident experts | 18 GiB / 4 | 3232 | 4397 | 4336 |
| v23, 96-column FP32 GEMM | 18 GiB / 4 | 3291 | 4482 | 4434 |
| v25, temporary KV/cache scratch for larger passes | 18 GiB / 4 or 8 | 3291 | 4492 | 4585 (repeat) |
| Original native engine reference | Original configuration | 1772 | 4420 | 5440 |

Units are prompt tokens/s. The v12 runs had zero major faults in the formal cases and identical input/output token IDs, accepted/offered draft counts, and finish reasons compared with their corresponding v11 controls. The attention kernel passed 21504 exact checks; the register HC up/mix kernel passed 80 exact comparisons. Earlier native-GEMM adoption has rounding differences relative to the original BLAS path, documented separately; these exact comparisons do not establish bitwise equivalence to BLAS.

`FNX_PF_MOE_WIDE=1` selects a 256-thread, 128-output-row MoE tile for IQ3_S, IQ3_XXS, IQ2_S, IQ4_NL, Q4_0 and packed Q2_0. Its epilogue stages one token tile at a time, keeping shared memory below 32 KiB and preserving the existing split-K sums. The IQ formats form the exactly representable codebook/scale factor in packed FP16, multiply by the original half scale once, then apply signs as bits. They do not round the original scale times subscale early. The independent narrow control matched all 1460 GPU comparisons, including boundary rows, partial jobs, subnormal/negative scales, graph replay and output guards. All v18 model token IDs and draft counts matched v15, with zero major faults in the formal requests. Its measured full-model gain is only 1.3–1.8%; larger microbenchmark gains should not be advertised as serving throughput.

`FNX_PF_GDN_ROWS=1` assigns a head to one wave and groups eight heads per block for convolution and gated normalization. It keeps the reference's independent 32-column reductions, broadcasts lane 0's sum exactly as the original shared-memory reduction did, and adds them in the original order. This avoids assuming that FMA-contracted XOR sums are identical in every lane. The row kernels and carried recurrent state passed 768 exact comparisons, including history parity, short tails, graph replay, tiny values and guards. v19 retained all model output IDs and draft counts from v18, with zero major faults in the formal cases.

The existing `FNX_PF_RCCL_F32=1` switch was tested separately. Times were 1390.2 / 3704.1 / 7527.5 ms, versus v19's 1265.9 / 3709.5 / 7489.8 ms with FP16 sums. It changed 4K output IDs and some draft counts, and did not provide a consistent throughput gain. It is not selected.

`FNX_PF_GDN_PREP_COMPACT=1` reuses the dead K/Q input tiles for the preparation matrices and gates. XOR-swizzled input columns avoid padding, reducing LDS to 32 KiB; barriers separate every read/write lifetime. Dot products, decay prefixes, triangular inversion and FP32 operation order are unchanged. The extended suite passed 1056 exact preparation, carried-state, graph and guard comparisons. v20's full-model IDs and draft counts matched v19, with zero major faults in the formal cases.

`FNX_PF_PHASE_SCRATCH=1` reuses the existing phase arena for HC and draft `up32`/`xf` scratch. It requires native compact heads, FP32 accumulation, scratch aliasing, head-split attention and RCCL. A separate small gather buffer preserves the original padded head projections below 64 rows, avoiding an overlapping copy into the phase arena. Actual prefill allocation fell from 1747 to 1589 MiB per GPU. All model IDs and draft counts matched v20 at 18.5 GiB RAM; that run's first 4K request had 16 major faults, so its timing is not a clean comparison. Reducing RAM backing to 18 GiB retained 368 MiB free VRAM per GPU after loading and again preserved all IDs/counts, with zero major faults in all formal cases. Its prefill speed was similar to v20; its intended benefit is room for additional resident experts during decode. API/decode/200K validation is recorded below.

`STRATA_PF_GEMM96=1` uses a 128×96 tile with 31.5 KiB LDS, hardware buffer loads and FP32 accumulation for eligible wide shapes. Operand byte ranges and grid size must fit 32 bits; other calls retain the old path. The existing fallback also remains for compilers without the buffer-load builtins. Scheduling ideas were checked against [AMD's GEMM walkthrough](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html); its FP16 accumulation performance claims are not used here. The selected burst schedule passed 448 byte-exact comparisons and the existing 146 FP64/range/padding/fallback checks. Full-model outputs and draft counts matched v21, with zero major faults in the formal cases and an approximately 2% improvement.

Two scheduling alternatives were rejected. Compiling the GEMM object in CU mode preserved outputs but made 16K/32K about 3% slower; the CMake option defaults off. Interleaving with explicit compiler barriers initially spilled registers. Static indexing removed the spills, but the selected burst schedule remained faster. Skinny 96×96 and 64×320 probes passed their exact comparisons but did not beat the existing narrow kernel in paired warm measurements. They live only in the benchmark tools, not in serving dispatch. A separate hipBLASLt probe also showed why unpaired microbenchmark times should not be treated as a serving speedup.

## Temporary prefill ownership of unused KV/cache pages

`FNX_PF_BORROW_PASS=8` keeps the normal four-chunk buffers, and temporarily reserves 800 MiB per GPU for four more chunks when a staged read is long enough and its complete KV footprint leaves room. This uses the existing static decode-cache arena; prefill reads separate immutable expert tables. Both GPUs are idled, cache entries are invalidated, and reserved page IDs are removed from the free list before scratch writes begin. Reclaim/churn respect the temporary owner. Normal completion and recoverable interruption synchronize before returning the pages; cache entries refill on demand. Long contexts that cannot fit the reservation retain four chunks and the full 200768-token capacity.

The host ownership planner passed 38922 bounds, conflict, rollback and full-capacity checks, including ASan/UBSan. Full-model 4K/16K/32K, a repeated 32K, and 4K/16K after return all matched v23's input/output IDs, draft counts and finish reasons. A separate 63914-token two-pass check also matched v23. The first v25 32K request took 7173.5 ms because the additional pinned embedding buffers were allocated on demand; the repeat took 6969.2 ms. No formal request had a major fault.

v26 prepares those pinned buffers during the internal warm read and optionally uses the existing card workers for independent host embedding chunks (`FNX_PF_EMBED_PARALLEL=1`). Workers join before lane/history data can change. `FNX_PF_EMBED_CHECK=1` compared every generated embedding and PLE byte with serial output, including the two-pass 63914-token case; all matched. The check is disabled for timing. The first and repeated native 32K cases then took 6930.3 and 6939.3 ms; 4K/16K were 1223.1 and 3547.2 ms, with identical model outputs and zero major faults in the formal cases.

Lower-cache alternatives (768 slots / 17.5 GiB RAM and 512 slots / 17 GiB RAM) retained all 2048-token output IDs and draft counts, but their weighted decode rates were 203.68 and 202.81 versus 205.47 for 1024 slots / 18 GiB. They showed no decode advantage, so the latter remains selected.

## v15 API and capacity validation

`FNX_PF_RESIDUAL_ALIAS=1` reuses each chunk's attention and FFN addend buffer: their consumers are ordered by `ev_ra`/`ev_rf`, and their lifetimes do not overlap. This saves 160 MiB per GPU with four chunks (320 MiB with eight). It requires HIP, head-split attention, and RCCL prompt sums. The four-chunk configuration reduces host expert backing from 19 to 18.5 GiB, retains 456 MiB free VRAM per GPU after loading, and preserves all reference output IDs and accepted/offered counts.

Using `tools/strata_bench.py`'s API client and `Suite.run_prompt`, with locally token-counted versions of its existing `long_prompt` text (at most one padding token), three cold-prefix requests per length returned:

| Actual input tokens | Median prompt tokens/s | Range | Reused tokens |
| ---: | ---: | ---: | ---: |
| 4096 | 3108.7 | 3093.4–3115.8 | 0 |
| 16384 | 4090.6 | 4075.2–4102.7 | 0 |
| 32768 | 4036.8 | 4035.3–4037.2 | 0 |

The existing three 2048-token decode cases produced 207.0 / 182.5 / 216.3 tokens/s, with a token/time-weighted mean of 200.9. This is close to the 200 target, with less margin than the production configuration's prior 215.7 result. It does not mean every prompt decodes above 200.

A separate API request with 200579 input tokens and 64 output tokens succeeded with zero prefix reuse and no OOM. Prefill took 60217.4 ms (3330.9 tokens/s); this long-context request decoded at 83.2 tokens/s. The short-prompt decode figure must not be advertised as the speed at 200K.

## v21 API, decode and compatibility validation

The 18 GiB RAM configuration passed three cold requests at each exact input length:

| Actual input tokens | Median prompt tokens/s | Range | Reused tokens |
| ---: | ---: | ---: | ---: |
| 4096 | 3307.8 | 3301.9–3324.4 | 0 |
| 16384 | 4431.7 | 4388.6–4437.6 | 0 |
| 32768 | 4349.4 | 4348.1–4359.7 | 0 |

The same three long-output cases through the API generated 2048 tokens each at 207.9 / 183.9 / 225.0 tokens/s. The token/time-weighted mean is 204.21. The prose case remains below 200. A cold request with 200579 input tokens and 64 output tokens completed without OOM: prompt 56654.3 ms (3540.4 tokens/s), TTFT 56.768 seconds, decode 82.3 tokens/s.

With all new options off and the original production JSON (PF_ROWS=256), v21 matched the immutable production v26 executable on warmup, the 4015-token prompt and the prose prompt, generating 128 tokens in each formal case. Input IDs, output IDs, accepted/offered draft counts and finish reasons were all identical. Both configurations had zero major faults in the formal requests.

Targets remain 3000 / 5000 / 6000 tokens/s through the API. **4K is met; 16K and 32K are not.** The production executable/configuration has not been replaced.

## v25 API and saved profile

Three exact-length cold API requests per length gave medians 3339.9 / 4522.0 / 4586.0 tokens/s at 4096 / 16384 / 32768 input tokens, all with zero reuse. Ranges were 3333.9–3358.2, 4503.7–4530.1 and 4466.4–4590.0. The first 32K request includes its extra pinned-buffer allocation; v26 moves that work into warmup.

The three API long-output cases generated 2048 tokens each at 208.3 / 184.0 / 223.7 tokens/s (weighted 204.01). Their complete texts matched v21. A 200579-input / 64-output request completed with no reuse and no OOM: 55901.4 ms prefill, 3588.1 prompt tokens/s and 82.1 decode tokens/s. Its output text also matched v21. A fresh 4096-token request afterward succeeded with zero reuse.

The validated native-format profile is saved on the host as `strata-swift-iq3_s-tp-prefill.json`, with `start-swift-tp-prefill.sh` copied from the existing launcher format. The JSON is private (0600), retains the existing authentication settings, and references an immutable executable. The original `tp220` JSON and launcher are preserved. No new Python launcher or configuration schema is introduced.

## v26 serving-profile validation

The saved profile now uses `strata-split-pf-embed-v26` (SHA-256 `6e3630aab75122b0c35ce78c865fc4779299538c5e72390e764cb8d168c52633`), with 1024 cache slots, two cache groups, 18 GiB expert RAM backing, PF_ROWS=4096, base pass=4, borrowed pass=8, and parallel host embedding enabled. The diagnostic embedding check is disabled.

| Exact cold API input | Median prompt tokens/s | Range, three runs |
| ---: | ---: | ---: |
| 4096 | 3326.0 | 3311.5–3336.3 |
| 16384 | 4533.1 | 4519.0–4534.6 |
| 32768 | 4601.5 | 4598.2–4608.2 |

All nine requests had zero reused tokens. API decode for the three 2048-token cases was 208.2 / 183.8 / 223.5 tokens/s, weighted 203.86. A 200579-input / 64-output cold request succeeded: prefill 56145.4 ms (3572.5 tokens/s), TTFT 56.260 seconds, decode 82.1 tokens/s. A fresh 4096-token request afterward also succeeded. The long-output texts and the 200K output text matched v21 exactly.

Default-off v26 again matched the original production v26 binary on warmup, 4015-token input with 128 output tokens, and prose with 128 output tokens: all input/output IDs, draft counts and finish reasons matched. This is a default-off compatibility result, not a claim that the earlier native-GEMM adoption is bit-identical to BLAS.

Cancellation was exercised while the 800 MiB reservation was active. Sending the supported `STOP` command during a 63914-token prompt returned `DONE ... cancel` with zero generated tokens. Retrying the same prompt, sending a new 4015-token prompt, and borrowing again for 31956 tokens all succeeded on the same process and matched control output IDs and draft counts. The test engine then exited normally.

The 16K and 32K targets remain unmet. The measurements do not establish 200 tokens/s decode at 200K context, nor 200 tokens/s on every prompt. Only HIP on these two gfx1100 cards has been validated; other backends were not exercised.

Private raw logs and native-format experiment configurations are under `build-tensor-parralize-baseline/prefill-20261104/`. They may contain local paths or authentication settings and must not be published wholesale.

## Fused routed gate/up and SwiGLU

The opt-in `FNX_PF_MOE_FUSED=1` path pairs 64 gate rows with their corresponding 64 up rows in a 128-row tile. Each row keeps the existing FP32 K-step and split-K reduction order. The epilogue applies the same SwiGLU operation and FP16 rounding, and writes the down projection's input directly. It avoids storing and rereading the full FP32 gate/up intermediate.

The supported gate formats are IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, IQ4_NL, Q4_0 and packed Q2_0. Initialization validates every layer's format, row stride and geometry before omitting the intermediate allocation. The option requires HIP and FP16 expert activations. It defaults off. Unsupported public-kernel calls return false without writing output; enabling the engine option with incompatible model weights gives an initialization error.

On the two gfx1100 cards, 5056 exact comparisons passed, covering both storage layouts, zero/partial/full jobs, split-K boundaries, extreme half scales, graph replay, output guards and shape/type fallback. The reference uses the existing narrow IQ4_XS kernel and the selected wide kernels for the other formats. Paired micros improved by about 5–9%; these are not whole-model rates.

At PF_ROWS=4096 the FFN region shrinks from 577.5 to 477.5 MiB. Attention then determines the shared region's size at 483.875 MiB, so the actual allocation saving is 93.625 MiB per GPU. Native 4K/16K/32K, repeated requests and a 63914-token two-pass request matched all control input/output IDs, accepted/offered counts and finish reasons. First-run 4K/16K/32K times were 1217.9/3525.9/6882.4 ms, but had 21/18/2 major faults and are not clean timing comparisons. Subsequent 4K/16K/32K reads took 1207.2/3566.1/6895.4 ms with zero major faults. The whole-model gain is approximately 1%, much smaller than the kernel microbenchmark gain.

Two alternatives were not selected. PF_ROWS=8192/pass=2/borrow=4 needed approximately 914 MiB more permanent allocation per GPU; with 20 GiB expert RAM backing, its warm 32K/16K reads took 6979.9/3637.4 ms, with no throughput advantage. Output IDs matched, but 32K draft counts differed. A benchmark-only HC kernel that shared weight reads between tokens failed exact comparisons (signed zero and a one-ULP half activation difference); it was removed from serving sources and retained only among private experiment artifacts.

The validated v27b binary is `strata-split-pf-moe-fused-v27b`, SHA-256 `ebcfe9ec9a58473993a263e4df7efde0ac641e7bf0ad1f6e5057d669ad08b4d9`. Three exact cold API requests per length, all with zero reuse, produced:

| Input tokens | Median prompt tokens/s | Range |
| ---: | ---: | ---: |
| 4096 | 3346.7 | 3337.1–3362.9 |
| 16384 | 4566.7 | 4524.0–4573.9 |
| 32768 | 4643.9 | 4641.2–4650.8 |

These medians improve on v26 by 0.6–0.9%. The same three 2048-token API decode cases gave 207.6/183.5/223.3 tokens/s, weighted 203.44; complete texts matched v26. The prose case remains below 200. A 200579-input / 64-output request succeeded without OOM: prompt 55588.3 ms (3608.3 tokens/s), TTFT 55.707 seconds, decode 82.1 tokens/s. Its text and draft counts matched v26. A fresh 4096-token request afterward succeeded with zero reuse. The 16K and 32K acceptance targets remain unmet.

With the original production configuration and all new options off, v27b matched the immutable production binary on warmup, 4015-token input with 128 generated tokens, and the prose case with 128 generated tokens. All input/output IDs, accepted/offered counts and finish reasons matched, with zero major faults. The saved `tp-prefill` JSON now selects v27b and enables the fused option. Its engine arguments and performance environment match the validated API configuration; the original authentication settings are retained. The original `tp220` configuration and executable remain available.
