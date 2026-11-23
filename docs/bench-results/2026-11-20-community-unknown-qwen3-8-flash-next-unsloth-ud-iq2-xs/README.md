# Community benchmark on AMD Radeon + AMD Radeon

Measured on 2026-11-20 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon + AMD Radeon, 48 GiB VRAM
- CPU: AMD Ryzen 7 7700 8-Core Processor (8 cores, 16 threads); installed RAM: 62.0 GiB; storage: nvme0n1 Predator SSD GM7000 1TB nvme
- OS: Linux 6.14.0-28-generic; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.62, commit 2003c31b; engine sfnx
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-split --native Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf --mtp mtp-rt-q4 --draft-vocab draft_vocab.bin --spec 6 --lookup 15 --conversation-cache-mib 8192 --max-context 262144
```

Environment of the config: `HIP_VISIBLE_DEVICES=0,1`, `STRATA_PF_NARROW_BK64=1`, `STRATA_PF_HC_UP_REGISTER=1`, `STRATA_PF_GEMM96=1`, `STRATA_PF_HC_DOWN_INCREMENT=1`, `STRATA_PF_Q8_PAIR=2`, `STRATA_PF_SHORT_GSU3=1`

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 175.8 (145.9-260.4) | 0.20 (0.19-0.24) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3744.2 (3734.4-3888.4) | not measured | 1.09 (1.05-1.09) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4812.1 (4801.3-4860.2) | not measured | 3.36 (3.32-3.36) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 4852.2 (3779.3-4920.5) | not measured | 6.64 (6.54-8.51) |
| agent session, follow-up turns | 13932 (13889-13977) at the last turn | see below | up to 64 | 3 | see below | 162.1 (135.9-211.0) | 0.91 (0.59-1.07) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.1 % (70.0-70.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13115 (13114-13116) new tokens of the files are read once); first turn (nothing cached) first token: 0.52 s (0.51-0.52).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1188.1 | 0.524 | 136.7 |
| 1 | 4203 | 621 | 3582 | 3392.7 | 1.074 | 184.2 |
| 2 | 5739 | 4266 | 1473 | 2534.8 | 0.596 | 178.7 |
| 3 | 8976 | 5802 | 3174 | 3325.0 | 0.971 | 146.3 |
| 4 | 11055 | 9022 | 2033 | 2955.4 | 0.704 | 177.9 |
| 5 | 13977 | 11118 | 2859 | 3194.1 | 0.912 | 190.1 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 63 s.
