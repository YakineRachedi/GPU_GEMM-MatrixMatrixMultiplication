#include "CUDA_GEMM_HPP"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

template<typename T>
    void cpu_gemm(const std::vector<T>& A,
                const std::vector<T>& B,
                std::vector<T>& C,
                int M, int N, int K) {

        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                for (int i = 0; i < M; ++i)
                    C[i + j * M] += A[i + k * M] * B[k + j * K];
    }

template<typename T>
    bool check_error(const std::vector<T>& ref,
                    const std::vector<T>& gpu) {

        double max_err = 0.0;

        for (size_t i = 0; i < ref.size(); ++i) {
            double err = std::abs(ref[i] - gpu[i]);
            max_err = std::max(max_err, err);
        }

        std::cout << "Max error: " << max_err << "\n";
        return max_err < 1e-4;
    }

template<typename T>
    void fill_random(std::vector<T>& A) {
        std::mt19937 gen(0);
        std::uniform_real_distribution<T> dist(0, 1);

        for (auto& x : A) x = dist(gen);
    }

int main() {

    using T = float;

    int M = 512;
    int N = 512;
    int K = 512;
    T alpha = 1;
    T beta = 0;

    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    std::vector<T> h_A(sizeA);
    std::vector<T> h_B(sizeB);
    std::vector<T> h_C_cpu(sizeC, 0);
    std::vector<T> h_C_gpu(sizeC, 0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fill_random(h_A);
    fill_random(h_B);

    auto t1 = std::chrono::high_resolution_clock::now();
    cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
    auto t2 = std::chrono::high_resolution_clock::now();

    double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "CPU GEMM time: " << cpu_time << " ms\n";

    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(T));
    cudaMalloc(&d_B, sizeB * sizeof(T));
    cudaMalloc(&d_C, sizeC * sizeof(T));

    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(T), cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks((M + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    cudaEventRecord(start);

    classic_gemm<T><<<blocks, threads>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(h_C_gpu.data(), d_C, sizeC * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "\n=== CLASSIC GEMM ===\n";
    std::cout << "GPU time: " << gpu_time_ms << " ms\n";

    if (check_error(h_C_cpu, h_C_gpu))
        std::cout << "Correct\n";
    else
        std::cout << "Wrong result\n";

    std::fill(h_C_gpu.begin(), h_C_gpu.end(), 0);
    cudaMemset(d_C, 0, sizeC * sizeof(T));

    constexpr int TILE = 16;

    dim3 threads2(TILE, TILE);
    dim3 blocks2((M + TILE - 1) / TILE,
                (N + TILE - 1) / TILE);

    cudaEventRecord(start);

    tile2D_gemm<T, TILE><<<blocks2, threads2>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_tile_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_tile_ms, start, stop);

    cudaMemcpy(h_C_gpu.data(), d_C, sizeC * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "\n=== TILED GEMM ===\n";
    std::cout << "GPU time: " << gpu_time_tile_ms << " ms\n";

    if (check_error(h_C_cpu, h_C_gpu))
        std::cout << "Correct\n";
    else
        std::cout << "Wrong result\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}