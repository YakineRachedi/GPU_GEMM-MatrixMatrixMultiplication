#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP


#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>


/**
 * @brief CUDA Kernel for Classical General Matrix Multiplication (GEMM).
 * * This kernel computes the matrix product C = A * B. It is designed for 
 * matrices stored in Column-Major format (BLAS standard). 
 * * The implementation utilizes "Grid-stride loops," allowing the kernel to 
 * process matrices of any size regardless of the grid dimensions.
 *
 * @tparam T           Data type (typically float or double).
 * @param[in]  A       Pointer to the input matrix A (M x K).
 * @param[in]  B       Pointer to the input matrix B (K x N).
 * @param[out] C       Pointer to the output matrix C (M x N).
 * @param[in]  M       Number of rows of A and C.
 * @param[in]  N       Number of columns of B and C.
 * @param[in]  K       Number of common dimensions (columns of A, rows of B).
 */
template<typename T>
    __global__ void classic_gemm(const T* __restrict__ A,
                const T* __restrict__ B,
                T* __restrict__ C,
                int M, int N, int K) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        // Grid-stride loops
        for (int row = i; row < M; row += blockDim.x * gridDim.x) {
            for (int col = j; col < N; col += blockDim.y * gridDim.y) {

                T sum = static_cast<T>(0);

                for (int k = 0; k < K; ++k) {
                    sum += A[row + k * M] * B[k + col * K];
                }

                C[row + col * M] = sum;
            }
        }
    }


/**
 * @brief Tiled CUDA kernel for matrix-matrix multiplication (GEMM) in column-major format.
 *
 * This kernel computes C = A * B using shared memory tiles for better memory coalescing
 * and reduced global memory accesses. Each thread block computes a TILE x TILE submatrix of C.
 *
 * @tparam T    Data type of matrix elements (e.g., float, double).
 * @tparam TILE Tile size in both row and column directions.
 * @param[in]  A Pointer to input matrix A (size M x K) in column-major order.
 * @param[in]  B Pointer to input matrix B (size K x N) in column-major order.
 * @param[out] C Pointer to output matrix C (size M x N) in column-major order.
 * @param[in]  M Number of rows in A and C.
 * @param[in]  N Number of columns in B and C.
 * @param[in]  K Number of columns in A / rows in B.
 */
template<typename T, int TILE>
    __global__ void tile_gemm(const T* __restrict__ A,
                          const T* __restrict__ B,
                          T* __restrict__ C,
                          int M, int N, int K) {

    // Shared memory tiles for A and B
    __shared__ T As[TILE * TILE];
    __shared__ T Bs[TILE * TILE];

    // Global row and column indices for this thread
    int row = threadIdx.x + blockIdx.x * TILE;
    int col = threadIdx.y + blockIdx.y * TILE;

    // Local thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    T sum = static_cast<T>(0);

    // Loop over tiles of K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {

        // Compute global indices of elements to load into shared memory
        int kA = t * TILE + ty;  // Column index for A
        int kB = t * TILE + tx;  // Row index for B

        // Load elements of A into shared memory tile
        if (row < M && kA < K)
            As[tx + ty * TILE] = A[row + kA * M];
        else
            As[tx + ty * TILE] = static_cast<T>(0);

        // Load elements of B into shared memory tile
        if (col < N && kB < K)
            Bs[tx + ty * TILE] = B[kB + col * K];
        else
            Bs[tx + ty * TILE] = static_cast<T>(0);

        // Synchronize to make sure all threads have loaded their tile elements
        __syncthreads();

        // Compute partial sum for the tile
        for (int k = 0; k < TILE; ++k) {
            T a = As[tx + k * TILE];
            T b = Bs[k + ty * TILE];
            sum += a * b;
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the computed element to global memory
    if (row < M && col < N)
        C[row + col * M] = sum;
}


/**
 * @brief CUDA kernel to validate tile mapping using shared memory.
 *
 * Each thread loads one element from the input matrix into a shared memory tile,
 * then writes it back to the output matrix. If the mapping is correct,
 * Output_Matrix should be identical to Input_Matrix.
 *
 * @tparam T    Data type
 * @tparam TILE Tile size
 */
template<typename T, int TILE>
    __global__ void test_tile_mapping(const T* __restrict__ Input_Matrix,
                                    T* __restrict__ Output_Matrix,
                                    int M, int N) {

        __shared__ T LocalMatrix[TILE * TILE];

        int row = threadIdx.x + blockIdx.x * TILE;
        int col = threadIdx.y + blockIdx.y * TILE;

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        if (row < M && col < N) {
            LocalMatrix[tx + ty * TILE] = Input_Matrix[row + col * M];
        } else {
            LocalMatrix[tx + ty * TILE] = static_cast<T>(0);
        }

        __syncthreads();
        if (row < M && col < N) {
            Output_Matrix[row + col * M] = LocalMatrix[tx + ty * TILE];
        }
    }

/**
 * Function to test if matrix A do not mismatch with matrix B
 */
template<typename T>
    bool compare(const std::vector<T>& A, const std::vector<T>& B) {
        for (size_t i = 0; i < A.size(); ++i) {
            if (A[i] != B[i]) {
                std::cout << "Mismatch at " << i
                        << " : " << A[i] << " vs " << B[i] << "\n";
                return false;
            }
        }
        return true;
    }



#endif

