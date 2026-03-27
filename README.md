# High-Performance Matrix Multiplication (GEMM)

## Overview

This project implements high-performance dense matrix multiplication (GEMM) in C++ and CUDA.

- Column-major storage (BLAS/LAPACK convention)
- CPU and GPU implementations
- Cache-aware and parallel algorithms
- Benchmarking and autotuning

---

## Features

### Core
- `Matrix<T>` class (column-major)
- Operator overloading (`+`, `-`, `*`)

### CPU
- Classic GEMM (triple loop)
- Blocked GEMM (cache-friendly)
- OpenMP parallelization
- Runtime configuration

### GPU (CUDA)
- Classic GEMM kernel (grid-stride and global memory)
- Tiled GEMM kernel (shared memory)
- Template support (`float`, `double`)
- Tile mapping validation kernel

### Benchmarking
- Python script for:
  - Matrix size sweep
  - Block size tuning
  - Performance comparison

---

## Project Structure
├── Matrix.hpp                       # Matrix class (column-major)

├── GEMM.hpp                         # CPU GEMM implementations

├── CUDA_GEMM.hpp                    # CUDA kernels (naive + tiled)

├── CUDA_GEMM.cu                     # Kernel launchers / wrappers

├── cuda_test_mapping.cu             # Tile mapping validation

├── cuda_test.cu                     # GPU GEMM tests

├── test.cpp                         # CPU tests

├── Makefile                         # Build system

├── LaunchTest.py                    # Python autotuning (CPU)


---

## Configuration

### CPU (via environment variables)

| Variable     | Description                          | Example |
|--------------|--------------------------------------|--------|
| `GEMM_ALGO`  | Algorithm (`classic`, `block`)       | `block` |
| `GEMM_BLOCK` | Block size                           | `64`    |
| `GEMM_M`     | Rows of A                            | `512`   |
| `GEMM_N`     | Columns of B                         | `512`   |
| `GEMM_K`     | Inner dimension                      | `512`   |

---

### GPU

GPU kernels are compiled and executed with `nvcc`.

> Note: GPU experiments were performed on Google Colab (NVIDIA T4 GPU).

---

## Usage

### Build
- CPU 
```bash
make test
```
- GPU
```bash
make cuda_test
```

### Run with configuration (CPU)
```bash
GEMM_ALGO=block GEMM_BLOCK=64 GEMM_M=512 GEMM_N=512 GEMM_K=512 make test
```
## Python Benchmarking 
A Python script is provided to automate benchmarking and parameter tuning:
```bash
python LaunchTest.py
```


---

## Performance Notes

- Classic GEMM: simple, cache-inefficient
- Blocked GEMM: better cache locality
- CUDA naive: limited by global memory
- CUDA tiled: optimized with shared memory and coalescing

---

