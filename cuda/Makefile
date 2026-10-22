# Linux build. Needs nvcc and, for the demo, libglfw3-dev.
#   make demo       real-time demo (build/lbm_cuda)
#   make bench      headless benchmark, FP32 and FP64 (build/lbm_bench, build/lbm_bench_fp64)
#   make headless   demo without a window (build/lbm_cuda_headless)
#   make tools      bandwidth probe (build/stream_copy)
#   make baseline   first CUDA version, headless, for timing (build/karman_showcase_headless)
#   make check      compile everything that needs no GPU at run time (used by the CI)
#   make test       run two short cases and compare them with stored values (needs a GPU)
# Executables go to build/, the folder every logged command in results/ uses.
# Developed on Windows (cuda\run.bat); the CI compiles these targets on Linux.
NVCC  ?= nvcc
ARCH  ?= -arch=native
B      = build
FLAGS  = -O3 -std=c++17 $(ARCH)
DEPS   = lbm_core.cuh cases.h

.PHONY: all demo bench headless tools baseline check test clean
all: demo bench
demo: $(B)/lbm_cuda
bench: $(B)/lbm_bench $(B)/lbm_bench_fp64
headless: $(B)/lbm_cuda_headless
tools: $(B)/stream_copy
baseline: $(B)/karman_showcase_headless
check: bench headless tools baseline demo
test: $(B)/lbm_bench
	python3 ../tests/run_tests.py $(B)/lbm_bench

$(B):
	mkdir -p $(B)
$(B)/lbm_cuda: lbm_cuda.cu $(DEPS) | $(B)
	$(NVCC) $(FLAGS) $< -o $@ -lglfw -lGL
$(B)/lbm_cuda_headless: lbm_cuda.cu $(DEPS) | $(B)
	$(NVCC) $(FLAGS) -DHEADLESS $< -o $@
$(B)/lbm_bench: lbm_bench.cu $(DEPS) | $(B)
	$(NVCC) $(FLAGS) $< -o $@
$(B)/lbm_bench_fp64: lbm_bench.cu $(DEPS) | $(B)
	$(NVCC) $(FLAGS) -DUSE_DOUBLE $< -o $@
$(B)/stream_copy: tools/stream_copy.cu | $(B)
	$(NVCC) $(FLAGS) $< -o $@
$(B)/karman_showcase_headless: showcase/karman_showcase.cu | $(B)
	$(NVCC) $(FLAGS) -DHEADLESS $< -o $@
clean:
	rm -rf $(B)
