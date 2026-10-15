#!/usr/bin/env bash
# Run the whole local pipeline in dependency order and produce the report + README block.
#
#   source scripts/toolchain_env.sh
#   PROFILE=quick bash scripts/run_all_local.sh     # ~1-2 h: 3 reps, 60 s runs (sanity-level statistics)
#   PROFILE=full  bash scripts/run_all_local.sh     # many hours: config defaults (8 reps, 120 s runs)
#
# Each stage writes its own results directory; a failure stops the pipeline so you never build a report
# on partial data. Stages already finished can be skipped with SKIP="kernel mpi" (space separated).
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-quick}"
SKIP=" ${SKIP:-} "
CFG=configs/experiment_config.yaml
if [ "$PROFILE" = "quick" ]; then EXTRA="--reps 3 --duration 60"; else EXTRA=""; fi

stage() { case "$SKIP" in *" $1 "*) echo "== skip $1 =="; return 1;; esac; echo; echo "== $1 =="; }

stage tests && pytest tests -q
stage smoke && python -m inferno.experiments.run_baseline --config $CFG --reps 1 --duration 45 --patterns steady
stage kernel && make -C src/inferno/kernels run
stage mpi && bash scripts/run_mpi_microbench.sh
stage sweep && python -m inferno.experiments.run_sweep --config $CFG $EXTRA
stage scaling && python -m inferno.experiments.run_scaling --config $CFG $EXTRA --replicas 1 2
stage campaign && python -m inferno.experiments.run_full_campaign --config $CFG $EXTRA

CAMPAIGN=$(ls -dt results/campaign_* 2>/dev/null | head -1 || true)
if [ -n "$CAMPAIGN" ]; then
    python -m inferno.experiments.report "$CAMPAIGN"
    KERNEL_ARG=""
    [ -f results/kernel_bench.json ] && KERNEL_ARG="--kernel results/kernel_bench.json"
    python scripts/update_readme_results.py --run "$CAMPAIGN" $KERNEL_ARG
    echo "Report: $CAMPAIGN/report/report.md   README results block updated."
    echo "Dashboard: python demo/serve.py --run $CAMPAIGN --pattern bursty --rep 0"
fi
