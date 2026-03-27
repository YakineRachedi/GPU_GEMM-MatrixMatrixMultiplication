#include "cuda_GEMM.hpp"
#include <stdexcept>

#define CUDA_CHECK(err)                                                        \
    if ((err) != cudaSuccess) {                                                \
        throw std::runtime_error(std::string("CUDA error: ")                  \
                                 + cudaGetErrorString(err));                   \
    }

template<typename T>
    std::vector<T> cuda_classic_gemm(const std::vector<T>& A,
                                  const std::vector<T>& B,
                                  int M, int N, int K,
                                  int blockSize) {
        T *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));

        dim3 block(blockSize, blockSize);
        dim3 grid((M + blockSize - 1) / blockSize,
                (N + blockSize - 1) / blockSize);

        classic_gemm<T><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> C(M * N);
        CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return C;
    }

template<typename T, int TILE>
    std::vector<T> cuda_tile_gemm(const std::vector<T>& A,
                               const std::vector<T>& B,
                               int M, int N, int K) {
        T *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));

        dim3 block(TILE, TILE);
        dim3 grid((M + TILE - 1) / TILE,
                (N + TILE - 1) / TILE);

        tile_gemm<T, TILE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> C(M * N);
        CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return C;
    }

template std::vector<float>  cuda_classic_gemm(const std::vector<float>&,  const std::vector<float>&,  int, int, int, int);
template std::vector<double> cuda_classic_gemm(const std::vector<double>&, const std::vector<double>&, int, int, int, int);

template std::vector<float>  cuda_tile_gemm<float,  16>(const std::vector<float>&,  const std::vector<float>&,  int, int, int);
template std::vector<double> cuda_tile_gemm<double, 16>(const std::vector<double>&, const std::vector<double>&, int, int, int);
template std::vector<float>  cuda_tile_gemm<float,  32>(const std::vector<float>&,  const std::vector<float>&,  int, int, int);
template std::vector<double> cuda_tile_gemm<double, 32>(const std::vector<double>&, const std::vector<double>&, int, int, int);