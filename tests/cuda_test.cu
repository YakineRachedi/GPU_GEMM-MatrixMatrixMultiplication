#include "cuda_GEMM.cuh"
#include "GEMM.hpp"
#include "utils_matrix.hpp"
#include "cuda_check.cuh"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>


int main() {

    using T = float;

    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;

    constexpr T alpha = static_cast<T>(1);
    constexpr T beta  = static_cast<T>(0);

    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    //
    // ============================================================
    // Host matrices
    // ============================================================
    //

    std::vector<T> h_A(sizeA);
    std::vector<T> h_B(sizeB);
    std::vector<T> h_C_gpu(sizeC, 0);

    fill_random(h_A);
    fill_random(h_B);

    //
    // ============================================================
    // Create Matrix objects for CPU reference GEMM
    // ============================================================
    //

    Matrix<T> A(h_A, M, K);
    Matrix<T> B(h_B, K, N);
    Matrix<T> C(M, N, static_cast<T>(0));

    //
    // ============================================================
    // CPU reference GEMM
    // ============================================================
    //

    std::cout << "Running CPU reference GEMM...\n";

    auto cpu_start = std::chrono::high_resolution_clock::now();

    Matrix<T> C_ref =
        classic_gemm<T>(A, B, C, alpha, beta);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();

    std::cout << "CPU time: "
              << cpu_time_ms
              << " ms\n";

    //
    // ============================================================
    // CUDA events
    // ============================================================
    //

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    //
    // ============================================================
    // Device memory allocation
    // ============================================================
    //

    T *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A.data(),
        sizeA * sizeof(T),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B.data(),
        sizeB * sizeof(T),
        cudaMemcpyHostToDevice
    ));

    //
    // ============================================================
    // Naive GEMM
    // ============================================================
    //

    {
        CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(T)));

        dim3 threads(32, 32);

        dim3 blocks(
            (N + threads.x - 1) / threads.x,
            (M + threads.y - 1) / threads.y
        );

        cudaEventRecord(start);

        cuda_naive_gemm<T>
        <<<blocks, threads>>>(
            d_A, d_B, d_C,
            M, N, K,
            alpha, beta
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_ms = 0.0f;

        cudaEventElapsedTime(
            &gpu_time_ms,
            start,
            stop
        );

        CUDA_CHECK(cudaMemcpy(
            h_C_gpu.data(),
            d_C,
            sizeC * sizeof(T),
            cudaMemcpyDeviceToHost
        ));

        std::cout << "\n=== NAIVE GEMM ===\n";
        std::cout << "GPU time: "
                  << gpu_time_ms
                  << " ms\n";

        if (check_error(C_ref.data_vector(), h_C_gpu))
            std::cout << "Correct\n";
        else
            std::cout << "Wrong result\n";
    }

    //
    // ============================================================
    // Global Memory Coalesced GEMM
    // ============================================================
    //

    {
        CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(T)));

        constexpr int BLOCKSIZE = 16;

        dim3 threads(BLOCKSIZE * BLOCKSIZE);

        dim3 blocks(
            (N + BLOCKSIZE - 1) / BLOCKSIZE,
            (M + BLOCKSIZE - 1) / BLOCKSIZE
        );

        cudaEventRecord(start);

        classic_gemm_global_mem_coalesce<T, BLOCKSIZE>
        <<<blocks, threads>>>(
            d_A, d_B, d_C,
            M, N, K,
            alpha, beta
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_ms = 0.0f;

        cudaEventElapsedTime(
            &gpu_time_ms,
            start,
            stop
        );

        CUDA_CHECK(cudaMemcpy(
            h_C_gpu.data(),
            d_C,
            sizeC * sizeof(T),
            cudaMemcpyDeviceToHost
        ));

        std::cout << "\n=== COALESCED GEMM ===\n";
        std::cout << "GPU time: "
                  << gpu_time_ms
                  << " ms\n";

        if (check_error(C_ref.data_vector(), h_C_gpu))
            std::cout << "Correct\n";
        else
            std::cout << "Wrong result\n";
    }

    //
    // ============================================================
    // Shared Memory Tiled GEMM
    // ============================================================
    //

    {
        CUDA_CHECK(cudaMemset(d_C, 0, sizeC * sizeof(T)));

        constexpr int TILE = 16;

        dim3 threads(TILE, TILE);

        dim3 blocks(
            (N + TILE - 1) / TILE,
            (M + TILE - 1) / TILE
        );

        cudaEventRecord(start);

        tile2D_gemm<T, TILE>
        <<<blocks, threads>>>(
            d_A, d_B, d_C,
            M, N, K,
            alpha, beta
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_ms = 0.0f;

        cudaEventElapsedTime(
            &gpu_time_ms,
            start,
            stop
        );

        CUDA_CHECK(cudaMemcpy(
            h_C_gpu.data(),
            d_C,
            sizeC * sizeof(T),
            cudaMemcpyDeviceToHost
        ));

        std::cout << "\n=== TILED GEMM ===\n";
        std::cout << "GPU time: "
                  << gpu_time_ms
                  << " ms\n";

        if (check_error(C_ref.data_vector(), h_C_gpu))
            std::cout << "Correct\n";
        else
            std::cout << "Wrong result\n";
    }

    //
    // ============================================================
    // Cleanup
    // ============================================================
    //

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}