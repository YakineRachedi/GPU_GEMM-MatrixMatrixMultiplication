# High-Performance Matrix Multiplication (GEMM)

## Overview

This project implements high-performance dense matrix multiplication (GEMM) in modern C++ and CUDA.

The goal is to study and compare several CPU and GPU matrix multiplication strategies while keeping a clean and modular codebase inspired by BLAS-style APIs.

Features include:

- Column-major matrix storage (BLAS/LAPACK convention)
- CPU and CUDA implementations
- OpenMP parallelization
- Cache-aware blocked GEMM
- CUDA tiled shared-memory kernels
- Benchmarking and performance comparison against OpenBLAS

---

# Features

## Matrix Library

- Generic `Matrix<T>` class
- Column-major storage layout
- Operator overloading:
  - `+`
  - `-`
  - `*`
- Random matrix generation utilities

---

## CPU Implementations

Implemented in `src/cpu/GEMM.hpp`

### Classic GEMM

Reference triple-loop implementation:

```cpp
C = alpha * A * B + beta * C
```

### Blocked GEMM

Cache-friendly tiled implementation with configurable block size.

### Parallelization

- OpenMP support
- Dynamic scheduling for blocked GEMM

### BLAS Comparison

OpenBLAS is used as a reference backend for:
- correctness validation
- performance comparison

---

## GPU Implementations (CUDA)

Implemented in `src/gpu/cuda_GEMM.cuh`

### CUDA Naive GEMM

Basic kernel:
- one thread computes one output element
- global memory only

### Coalesced GEMM

1D-thread mapping improving memory coalescing.

### Tiled GEMM

Shared-memory tiled GEMM:
- reduced global memory traffic
- better cache reuse
- configurable tile size

---

## GPU Development Notes

The CUDA implementation was primarily developed and validated without access to a local NVIDIA GPU.

To verify correctness and execution of the GPU kernels, a dedicated series of experiments and validation tests was performed on Google Colab using NVIDIA T4 GPUs.

This allowed:
- validation of CUDA kernel execution
- correctness checks against the CPU reference GEMM
- benchmarking of naive, coalesced, and tiled CUDA kernels
- testing with both `float` and `double` precision


# Benchmarking

Benchmarks automatically measure:

- execution time
- speedup vs OpenBLAS
- numerical error

Results are stored in:

```text
benchmarks/results.csv
```

Python scripts are provided for:
- plotting performance
- comparing algorithms
- tuning block sizes

---

# Project Structure

```text
.
├── benchmarks/
│   ├── results.csv
│   └── plot.py
│
├── include/
│   └── Matrix.hpp
│
├── src/
│   ├── cpu/
│   │   └── GEMM.hpp
│   │   └── BLAS.hpp
│   │
│   └── gpu/
│       ├── cuda_GEMM.cuh
│       └── run_kernels.cu
│
├── tests/
│   ├── cpu_test.cpp
│   └── cuda_test.cu
│
├── utils/
│   ├── cuda_check.cuh
│   ├── generate_matrix.hpp
│   └── env_config.hpp
│
├── Makefile
└── README.md
```

---

# Build System

The project uses a unified Makefile for both CPU and GPU builds.

---

# Requirements

## CPU

- C++17 compiler
- OpenMP
- OpenBLAS

## GPU

- CUDA Toolkit
- NVIDIA GPU supporting CUDA

---

# Build

## CPU

```bash
make cpu
```

## GPU

```bash
make gpu
```

## Build Everything

```bash
make all
```

---

# Running CPU Benchmarks

## Default Run

```bash
make run
```

---

## Run Blocked GEMM

```bash
make run ALGO=block BLOCK=64
```

---

## Double Precision

```bash
make run TYPE=double
```

---

## Custom Matrix Sizes

```bash
make run M=1024 N=1024 K=1024
```

---

# Benchmark Automation

## Classic GEMM

```bash
make bench-classic
```

## Blocked GEMM

```bash
make bench-block
```

## Full Benchmark Suite

```bash
make bench-all
```

---

# Plotting Results

After benchmarks:

```bash
cd benchmarks
python plot.py
```

Generated plots include:

- execution time
- speedup
- numerical error

---

# Configuration Options

| Variable | Description | Default |
|---|---|---|
| `TYPE` | Scalar type (`float`, `double`) | `float` |
| `ALGO` | GEMM algorithm (`classic`, `block`) | `classic` |
| `BLOCK` | Block size for blocked GEMM | `64` |
| `M` | Rows of A/C | `256` |
| `N` | Columns of B/C | `256` |
| `K` | Inner dimension | `256` |
| `ARCH` | CUDA architecture | `sm_86` |

---

# Performance Notes

## CPU

### Classic GEMM
- simple implementation
- poor cache locality

### Blocked GEMM
- improved cache reuse
- significantly faster

### OpenMP
- parallel execution on CPU cores

---

## GPU

### Naive Kernel
- limited by global memory bandwidth

### Coalesced Kernel
- improved memory access patterns

### Tiled Kernel
- shared-memory optimization
- much better arithmetic intensity

---

# Future Work

- CUDA WMMA / Tensor Cores
- Asynchronous CUDA pipelines
- Auto-tuning system
- SIMD vectorization
- Multi-GPU support
- Batched GEMM
- Integration with cuBLAS / CUTLASS

---

# Author

Personal HPC / CUDA project focused on:
- performance engineering
- numerical computing
- low-level optimization
- GPU programming