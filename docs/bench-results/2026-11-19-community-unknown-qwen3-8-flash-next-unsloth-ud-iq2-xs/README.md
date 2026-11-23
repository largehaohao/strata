# Community benchmark on AMD Radeon + AMD Radeon

Measured on 2026-11-19 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon + AMD Radeon, 48 GiB VRAM
- CPU: AMD Ryzen 7 7700 8-Core Processor (8 cores, 16 threads); installed RAM: 62.0 GiB; storage: nvme0n1 Predator SSD GM7000 1TB nvme
- OS: Linux 6.14.0-28-generic; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.62, commit not a git checkout; engine sfnx
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq2_xs
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-split-verified --native Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf --mtp mtp-rt-q4 --draft-vocab draft_vocab.bin --spec 6 --lookup 15 --conversation-cache-mib 8192 --max-context 262144
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 174.9 (146.9-262.7) | 0.19 (0.18-0.24) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 3778.0 (3772.3-3877.5) | not measured | 1.08 (1.05-1.08) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 4788.3 (4653.1-4852.2) | not measured | 3.37 (3.33-3.47) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 4860.7 (4845.1-4900.2) | not measured | 6.62 (6.57-6.65) |
| agent session, follow-up turns | 8965 (8910-8970) at the last turn | see below | up to 64 | 3 | see below | 77.7 (51.8-219.0) | 0.59 (0.53-1.11) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 70.9 % (70.7-70.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 8207 (8206-8207) new tokens of the files are read once); first turn (nothing cached) first token: 0.49 s (0.46-0.53).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 1168.9 | 0.533 | 141.7 |
| 1 | 2264 | 622 | 1642 | 2253.9 | 0.74 | 144.3 |
| 2 | 3970 | 2327 | 1643 | 3013.0 | 0.558 | 141.8 |
| 3 | 5623 | 3981 | 1642 | 2839.4 | 0.591 | 77.7 |
| 4 | 7267 | 5625 | 1642 | 2829.1 | 0.593 | 51.9 |
| 5 | 8910 | 7268 | 1642 | 2801.1 | 0.6 | 51.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 56 s.
