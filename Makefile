# Compilers
CXX = g++
NVCC = nvcc

MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

# Sources
CPU_SRC = tests/cpu_test.cpp
GPU_SRC = tests/cuda_test.cu src/gpu/run_kernels.cu

# Benchmarks directory
BENCH_DIR = benchmarks

INCLUDES = -Iinclude -Iutils -Isrc/cpu -Isrc/gpu

# Targets
CPU_TARGET = $(BENCH_DIR)/cpu_test.exe
GPU_TARGET = $(BENCH_DIR)/gpu_test.exe

# OpenBLAS
OPENBLAS_INC   = C:/OpenBLAS/include
OPENBLAS_LIB   = C:/OpenBLAS/lib
OPENBLAS_FLAGS = -lopenblas -static

# Parameters 
TYPE  ?= float
ALGO  ?= classic
BLOCK ?= 64
M     ?= 256
N     ?= 256
K     ?= 256
ARCH ?= sm_86

# Flags
CXXFLAGS = -std=c++17 -DSCALAR_TYPE=$(TYPE) -DBLOCK_SIZE=$(BLOCK)
NVCCFLAGS = -std=c++17 -arch=$(ARCH)

ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
NVCCFLAGS += -g -G

else
CXXFLAGS += -O3 -march=native -Wall -fopenmp
NVCCFLAGS += -O3
endif

.PHONY: all cpu gpu run bench-classic bench-block bench-all clean setup
default: help

docker-run:
	docker run -it --rm \
		--gpus all \
		-v $(CURDIR):/usr/local/workspace \
		-w /usr/local/workspace \
		matrix-gemm

docker-build:
	docker build -t matrix-gemm .

setup:
	mkdir -p $(BENCH_DIR)

all: cpu gpu

cpu: setup
	@echo "Compiling CPU benchmark..."
	$(CXX) $(CXXFLAGS) $(CPU_SRC) $(INCLUDES) \
	-I$(OPENBLAS_INC) \
	-L$(OPENBLAS_LIB) $(OPENBLAS_FLAGS) \
	-o $(CPU_TARGET)

gpu: setup
	@echo "Compiling GPU benchmark..."
	$(NVCC) $(NVCCFLAGS) $(GPU_SRC) $(INCLUDES) \
	-o $(GPU_TARGET)

run: cpu
	set GEMM_ALGO=$(ALGO)&& \
	set GEMM_BLOCK=$(BLOCK)&& \
	set GEMM_M=$(M)&& \
	set GEMM_N=$(N)&& \
	set GEMM_K=$(K)&& \
	$(CPU_TARGET)

# =========================================================
# Benchmarks
# =========================================================

bench-classic: cpu
	set GEMM_ALGO=classic&& \
	set GEMM_M=512&& \
	set GEMM_N=512&& \
	set GEMM_K=512&& \
	$(CPU_TARGET)

bench-block: cpu
	set GEMM_ALGO=block&& \
	set GEMM_BLOCK=64&& \
	set GEMM_M=512&& \
	set GEMM_N=512&& \
	set GEMM_K=512&& \
	$(CPU_TARGET)

bench-all: cpu
	@echo "--- Running Benchmarks ---"

	set GEMM_ALGO=classic&& \
	set GEMM_M=512&& \
	set GEMM_N=512&& \
	set GEMM_K=512&& \
	$(CPU_TARGET)

	set GEMM_ALGO=block&& \
	set GEMM_BLOCK=32&& \
	set GEMM_M=512&& \
	set GEMM_N=512&& \
	set GEMM_K=512&& \
	$(CPU_TARGET)

	set GEMM_ALGO=block&& \
	set GEMM_BLOCK=64&& \
	set GEMM_M=512&& \
	set GEMM_N=512&& \
	set GEMM_K=512&& \
	$(CPU_TARGET)

	set GEMM_ALGO=block&& \
	set GEMM_BLOCK=128&& \
	set GEMM_M=512&& \
	set GEMM_N=512&& \
	set GEMM_K=512&& \
	$(CPU_TARGET)

clean:
	del /Q *.exe 2>nul
	del /Q benchmarks\*.exe 2>nul
	del /Q benchmarks\*.csv 2>nul

help:
	@echo "=================================================="
	@echo "                 GEMM Makefile"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make cpu              Compile CPU benchmark"
	@echo "  make gpu              Compile GPU benchmark"
	@echo "  make all              Compile CPU + GPU benchmarks"
	@echo "  make run              Run CPU benchmark"
	@echo "  make bench-classic    Run classic GEMM benchmark"
	@echo "  make bench-block      Run blocked GEMM benchmark"
	@echo "  make bench-all        Run all CPU benchmarks"
	@echo "  make clean            Remove executables and CSV files"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=yes             Enable debug mode"
	@echo "  TYPE=float|double     Scalar type (default: float)"
	@echo "  BLOCK=<size>          Block size for blocked GEMM (default: 64)"
	@echo "  ALGO=classic|block    GEMM algorithm (default: classic)"
	@echo "  M=<size>              Number of rows of A/C (default: 256)"
	@echo "  N=<size>              Number of cols of B/C (default: 256)"
	@echo "  K=<size>              Shared dimension (default: 256)"
	@echo "  ARCH=sm_XX            CUDA architecture (default: sm_86)"
	@echo ""
	@echo "Configuration:"
	@echo "  CXX              : $(CXX)"
	@echo "  NVCC             : $(NVCC)"
	@echo "  CXXFLAGS         : $(CXXFLAGS)"
	@echo "  NVCCFLAGS        : $(NVCCFLAGS)"
	@echo "  OPENBLAS_INC     : $(OPENBLAS_INC)"
	@echo "  OPENBLAS_LIB     : $(OPENBLAS_LIB)"
	@echo "  OPENBLAS_FLAGS   : $(OPENBLAS_FLAGS)"
	@echo ""
	@echo "Examples:"
	@echo "  make cpu"
	@echo "  make gpu ARCH=sm_86"
	@echo "  make all TYPE=double"
	@echo "  make run ALGO=block BLOCK=128"
	@echo "  make run TYPE=double M=1024 N=1024 K=1024"
	@echo "  make bench-all"
	@echo ""
	@echo "Executables:"
	@echo "  $(CPU_TARGET)"
	@echo "  $(GPU_TARGET)"
	@echo "=================================================="
	@echo ""
	@echo "=================================================="
	@echo "                    DOCKER"
	@echo "=================================================="
	@echo ""
	@echo "Docker usage:"
	@echo "  make docker-run       Run project inside Docker container"
	@echo ""
	@echo "Docker features:"
	@echo "  - Prebuilt environment (g++, nvcc, OpenBLAS, Python)"
	@echo "  - Reproducible builds"
	@echo "  - GPU support (if --gpus available)"
	@echo ""
	@echo "Example workflow:"
	@echo "  docker build -t matrix-gemm ."
	@echo "  make docker-run"
	@echo "  make cpu"
	@echo "  make gpu"
	@echo ""