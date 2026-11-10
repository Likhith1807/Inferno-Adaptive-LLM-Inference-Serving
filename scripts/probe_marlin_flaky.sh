#!/usr/bin/env bash
# Marlin flakiness probe on the Qwen down shape, plus the other Qwen shapes as controls.
set -uo pipefail
python scripts/probe_marlin_flaky.py 1536 8960 2>&1 | grep -v INFO
python scripts/probe_marlin_flaky.py 17920 1536 2>&1 | grep -v INFO
python scripts/probe_marlin_flaky.py 2048 1536 2>&1 | grep -v INFO
