#!/bin/sh
cd "/home/john/Strata" || exit 1
unset LD_LIBRARY_PATH LD_PRELOAD HSA_ENABLE_INTERRUPT HSA_ENABLE_MWAITX FNX_GPU_TIMING
export STRATA_NO_PRESET=1 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
exec taskset -c 0-3 "/home/john/Strata/.venv/bin/python" "/home/john/Strata/local-tuning/2026-11-03/guarded_server.py" --config "/home/john/Strata/terminal/strata-swift-iq3_s-tp-pi-increment.json" --thermal-log "/home/john/Strata/terminal/strata-swift-iq3_s-tp-pi-increment-thermal.jsonl" --open "$@"
