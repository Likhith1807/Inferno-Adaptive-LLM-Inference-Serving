.PHONY: pdf kernel-all setup toolchain test plan calibrate contention site smoke trace kernel kernel-profile kernel-compare mpi-bench baseline sweep scaling adaptive campaign report readme freeze clean
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
