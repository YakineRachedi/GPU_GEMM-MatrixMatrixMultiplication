#include "cuda_GEMM.hpp"
#define TILE_SIZE 8

int main() {

    int M = 25;
    int N = 25;

    size_t size = M * N;

    std::vector<float> h_A(size);
    std::vector<float> h_C(size, 0);

    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i)
            h_A[i + j * M] = i + j * 100;

    float *d_A, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    test_tile_mapping<float, TILE_SIZE><<<blocks, threads>>>(d_A, d_C, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    if (compare(h_A, h_C)) {
        std::cout << "SUCCESS: Mapping is CORRECT\n";
    } else {
        std::cout << "ERROR: Mapping is WRONG\n";
    }


    std::cout << "\nInput matrix:\n";
    for (int i = 0; i < std::min(M, 10); ++i) {
        for (int j = 0; j < std::min(N, 10); ++j) {
            std::cout << h_A[i + j * M] << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\nOutput matrix:\n";
    for (int i = 0; i < std::min(M, 10); ++i) {
        for (int j = 0; j < std::min(N, 10); ++j) {
            std::cout << h_C[i + j * M] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}