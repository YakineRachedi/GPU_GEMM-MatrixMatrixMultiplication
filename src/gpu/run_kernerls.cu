#include "cuda_GEMM.hpp"
#include "cuda_check.cuh"

#include <vector>

//
// ============================================================
// Naive GEMM Wrapper
// ============================================================
//

template<typename T>
std::vector<T> run_cuda_naive_gemm(const std::vector<T>& A, const std::vector<T>& B, int M, int N, int K, 
                            int blockSize, T alpha, T beta) {

    T *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));

    dim3 block(blockSize, blockSize);

    dim3 grid((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);

    cuda_naive_gemm<T><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> C(M * N);

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

//
// ============================================================
// Global Memory Coalesced GEMM Wrapper
// ============================================================
//

template<typename T, const uint BLOCKSIZE> std::vector<T> run_cuda_coalesced_gemm(const std::vector<T>& A, const std::vector<T>& B, 
                                int M, int N, int K, T alpha, T beta) {

    T *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));

    // 1D thread block
    dim3 block(BLOCKSIZE * BLOCKSIZE);

    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);

    classic_gemm_global_mem_coalesce<T, BLOCKSIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> C(M * N);

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

//
// ============================================================
// Shared Memory Tiled GEMM Wrapper
// ============================================================
//

template<typename T, int TILE> std::vector<T> run_cuda_tile_gemm(const std::vector<T>& A, const std::vector<T>& B, 
                                    int M, int N, int K, T alpha, T beta) {

    T *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));

    dim3 block(TILE, TILE);

    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    tile2D_gemm<T, TILE><<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> C(M * N);

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

template std::vector<float> run_cuda_naive_gemm<float>(const std::vector<float>&, const std::vector<float>&, int, int, int, int, float, float);
template std::vector<double> run_cuda_naive_gemm<double>(const std::vector<double>&, const std::vector<double>&, int, int, int, int, double, double);
template std::vector<float> run_cuda_coalesced_gemm<float, 16>(const std::vector<float>&, const std::vector<float>&, int, int, int, float, float);
template std::vector<float> run_cuda_coalesced_gemm<float, 32>(const std::vector<float>&, const std::vector<float>&, int, int, int, float, float);
template std::vector<float> run_cuda_tile_gemm<float, 16>(const std::vector<float>&, const std::vector<float>&, int, int, int, float, float);
template std::vector<float> run_cuda_tile_gemm<float, 32>(const std::vector<float>&, const std::vector<float>&, int, int, int, float, float);