#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/toolchain_env.sh"
cd "$(dirname "$0")/../src/inferno/kernels"
make libinferno_int4.so
