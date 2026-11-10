.PHONY: figure quick-bench layer-matrix marlin-probe findings pdf kernel-all setup toolchain test plan calibrate contention site smoke trace kernel kernel-profile kernel-compare mpi-bench baseline sweep scaling adaptive campaign report readme freeze clean
CONFIG ?= configs/experiment_config.yaml

setup:
	bash scripts/setup_env.sh

toolchain:
	bash scripts/setup_toolchain.sh

test:
	pytest tests/ -v

plan:
	python -m inferno.experiments.run_baseline --config $(CONFIG) --dry-run
	python -m inferno.experiments.run_sweep --config $(CONFIG) --dry-run
	python -m inferno.experiments.run_scaling --config $(CONFIG) --dry-run
	python -m inferno.experiments.run_adaptive --config $(CONFIG) --dry-run

# Measure this machine's capacity and derive load rates from it (writes configs/workload_config.local.yaml)
calibrate:
	python -m inferno.experiments.calibrate --config $(CONFIG)

# ~10 min end-to-end check on one config: proves server + client + metrics + power work on this machine
smoke:
	python -m inferno.experiments.run_baseline --config $(CONFIG) --reps 1 --duration 45 --patterns steady --only int4-bs32

trace:
	python scripts/prepare_trace.py

# Phase 3: CUDA kernel. Builds with nvcc, checks correctness vs a CPU reference, times against cuBLAS.
kernel:
	$(MAKE) -C src/inferno/kernels run

kernel-profile:
	$(MAKE) -C src/inferno/kernels profile
	python scripts/summarize_ncu.py

# In-situ vLLM Marlin decode timing, then the merged custom-vs-library table
kernel-compare:
	python -m inferno.kernels.bench_vllm_marlin --quantization awq_marlin
	python -m inferno.kernels.bench_vllm_marlin --quantization awq
	python scripts/compare_kernels.py

# Phase 4: GPU-free MPI coordination cost vs worker count
mpi-bench:
	bash scripts/run_mpi_microbench.sh

baseline:
	python -m inferno.experiments.run_baseline --config $(CONFIG)

sweep:
	python -m inferno.experiments.run_sweep --config $(CONFIG)

scaling:
	python -m inferno.experiments.run_scaling --config $(CONFIG)

adaptive:
	python -m inferno.experiments.run_adaptive --config $(CONFIG) --reps 3

campaign:
	python -m inferno.experiments.run_full_campaign --config $(CONFIG)

# Mechanism study: one replica vs two co-resident replicas (idle / active) at equal total load
contention:
	python -m inferno.experiments.run_contention --config $(CONFIG)

# Technical report PDF from docs/results (needs reportlab, matplotlib, pandas, scipy)
pdf:
	python scripts/build_report_pdf.py

# Kernel study end to end: library, ladder + small-batch runs, rotation check, in-situ and vLLM decode benchmarks
kernel-all:
	bash scripts/rerun_kernel_all.sh
	bash scripts/flush_sweep.sh
	bash scripts/insitu_graphs.sh
	bash scripts/run_e2e_matrix.sh 3

# Rebuild the showcase site (docs/demo/index.html) from docs/results; open it in a browser or host it as-is
site:
	python demo/build_site.py

# usage: make report RUN=results/campaign_20260101_120000
report:
	python -m inferno.experiments.report $(RUN)

# usage: make readme RUN=results/campaign_... KERNEL=results/kernel_bench.json
readme:
	python scripts/update_readme_results.py --run $(RUN) $(if $(KERNEL),--kernel $(KERNEL),)

freeze:
	pip freeze > requirements.lock

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
	$(MAKE) -C src/inferno/kernels clean

# Milestone 5: headline figure from saved data only (no GPU needed)
figure:
	python scripts/make_headline_figure.py

# Short GPU benchmark for independent reproduction (~3-5 min); writes results/quick/<timestamp>/
quick-bench:
	bash scripts/quick_bench.sh

# Every Qwen2.5-1.5B decode layer x batch 1..64: custom vs Marlin vs AWQ vs fp16 (docs/results/layer_matrix/report.md)
layer-matrix:
	bash scripts/run_layer_matrix.sh

# Marlin batch-size correctness probe (kernel level + real checkpoint) -> docs/results/marlin_fault/report.md
marlin-probe:
	mkdir -p results/marlin_fault
	bash scripts/probe_marlin_flaky.sh > results/marlin_fault/kernel_probe.txt
	bash scripts/probe_marlin_e2e.sh
	python scripts/summarize_marlin_fault.py

# Regenerate FINDINGS.md, POST_DRAFT.md and the README findings block from docs/results
findings:
	python scripts/build_findings.py
