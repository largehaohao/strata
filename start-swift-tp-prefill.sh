#!/bin/sh
cd "/home/john/Strata" || exit 1
unset HSA_ENABLE_INTERRUPT HSA_ENABLE_MWAITX FNX_GPU_TIMING
export STRATA_NO_PRESET=1 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
exec taskset -c 0-3 "/home/john/Strata/.venv/bin/python" "/home/john/Strata/local-tuning/2026-11-03/guarded_server.py" --config "/home/john/Strata/strata-swift-iq3_s-tp-prefill.json" --thermal-log "/home/john/Strata/strata-swift-iq3_s-tp-prefill-thermal.jsonl" --open "$@"
