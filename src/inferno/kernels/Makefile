# Standalone kernel benchmark (no PyTorch needed).
#   make            build bench_int4_gemv
#   make run        run; writes ../../../results/kernel_bench.json (+ .txt with the human-readable table)
#   make profile    Nsight Compute report for every GEMV version (needs ncu and GPU counter access)
#
# ARCH: sm_89 = RTX 40-series (Ada). Use sm_80 for A100, sm_86 for A10G/RTX 30-series, sm_90 for H100.
# EXTRA: extra nvcc flags for A/B experiments, e.g.  make EXTRA=-DINFERNO_NO_STREAM_LOADS
ARCH ?= sm_89
NVCC ?= nvcc
EXTRA ?=
# --cudart shared: keeps the link independent of the static runtime (conda ships both; shared is simpler).
NVFLAGS = $(EXTRA) -O3 -std=c++17 -arch=$(ARCH) -lineinfo --use_fast_math --cudart shared -Xptxas -v

bench_int4_gemv: bench_int4_gemv.cu dequant_kernel.cu int4_gemv.cuh
	@$(NVCC) $(NVFLAGS) -o $@ bench_int4_gemv.cu dequant_kernel.cu -lcublas 2> ptxas_report.txt \
	    || { cat ptxas_report.txt; exit 1; }
	@grep -E "registers|spill" ptxas_report.txt | sed -n 1,12p || true
	@echo "built $@ (full compiler report: ptxas_report.txt)"

bench_int4_batch: bench_int4_batch.cu int4_gemv_batch.cuh int4_gemv.cuh
	@$(NVCC) $(NVFLAGS) -o $@ bench_int4_batch.cu -lcublas 2> ptxas_batch_report.txt \
	    || { cat ptxas_batch_report.txt; exit 1; }
	@grep -E "spill" ptxas_batch_report.txt | sort | uniq -c | head -5 || true
	@echo "built $@"

run-batch: bench_int4_batch
	mkdir -p ../../../results
	./bench_int4_batch > ../../../results/kernel_batch.json 2> ../../../results/kernel_batch.txt
	@cat ../../../results/kernel_batch.txt

libinferno_int4.so: int4_capi.cu int4_gemv_batch.cuh int4_gemv.cuh
	@$(NVCC) $(NVFLAGS) -shared -Xcompiler -fPIC -o $@ int4_capi.cu 2> ptxas_capi_report.txt \
	    || { cat ptxas_capi_report.txt; exit 1; }
	@echo "built $@"

run: bench_int4_gemv
	mkdir -p ../../../results
	./bench_int4_gemv > ../../../results/kernel_bench.json 2> ../../../results/kernel_bench.txt
	@cat ../../../results/kernel_bench.txt
	@echo "wrote results/kernel_bench.json and results/kernel_bench.txt"

# Per-kernel Nsight Compute: one full-set report per GEMV version on the Qwen gate_up shape (index 2) and the
# Llama mlp_up shape (index 5), plus a CSV of the metrics that explain each optimization stage.
profile: bench_int4_gemv
	bash ../../../scripts/profile_kernels.sh ./bench_int4_gemv

clean:
	rm -f libinferno_int4.so bench_int4_gemv bench_int4_batch ptxas_report.txt ptxas_batch_report.txt
