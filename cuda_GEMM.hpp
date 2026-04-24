#ifndef CUDA_GEMM_HPP
#define CUDA_GEMM_HPP


#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>


/**
 * @brief CUDA Kernel for Classical General Matrix Multiplication (GEMM).
 *
 * This kernel computes the matrix product:
 *      C = alpha * A * B + beta * C
 *
 * Matrices are stored in Column-Major format (BLAS standard):
 *      - A (M x K)
 *      - B (K x N)
 *      - C (M x N)
 *
 * Each thread computes one element of C using a dot product.
 * Boundary checks ensure safe execution for arbitrary matrix sizes.
 *
 * @tparam T           Data type (typically float or double).
 * @param[in]  A       Pointer to the input matrix A (M x K).
 * @param[in]  B       Pointer to the input matrix B (K x N).
 * @param[out] C       Pointer to the output matrix C (M x N).
 * @param[in]  M       Number of rows of A and C.
 * @param[in]  N       Number of columns of B and C.
 * @param[in]  K       Number of common dimensions (columns of A, rows of B).
 * @param[in]  alpha   Scalar multiplier for A * B.
 * @param[in]  beta    Scalar multiplier for C.
 */

template<typename T>
    __global__ void classic_gemm(const T* __restrict__ A,
                const T* __restrict__ B,
                T* __restrict__ C,
                int M, int N, int K, 
                const T alpha, const T beta) {
        int rows = blockIdx.y * blockDim.y + threadIdx.y;
        int cols = blockIdx.x * blockDim.x + threadIdx.x;
        if(rows < M && cols < N){
            T sum = static_cast<T>(0);
            for (int k = 0; k < K; ++k) {
                sum += A[rows + k * M] * B[k + cols * K];
            }
            C[rows + cols * M] = alpha * sum + beta * C[rows + cols * M];
        }
    }


/**
 * @brief Tiled CUDA kernel for General Matrix Multiplication (GEMM).
 *
 * This kernel computes the matrix product:
 *      C = alpha * A * B + beta * C
 *
 * Matrices are stored in Column-Major format (BLAS standard):
 *      - A (M x K)
 *      - B (K x N)
 *      - C (M x N)
 *
 * The implementation uses shared memory tiling to reduce global memory
 * accesses. Each thread block computes a TILE x TILE submatrix of C,
 * and each thread computes one element using a tiled dot product.
 *
 * @tparam T           Data type (typically float or double).
 * @tparam TILE        Tile size (blockDim.x = blockDim.y = TILE).
 * @param[in]  A       Pointer to the input matrix A (M x K).
 * @param[in]  B       Pointer to the input matrix B (K x N).
 * @param[out] C       Pointer to the output matrix C (M x N).
 * @param[in]  M       Number of rows of A and C.
 * @param[in]  N       Number of columns of B and C.
 * @param[in]  K       Number of common dimensions (columns of A, rows of B).
 * @param[in]  alpha   Scalar multiplier for A * B.
 * @param[in]  beta    Scalar multiplier for C.
 */

template<typename T, int TILE>
    __global__ void tile2D_gemm(const T* __restrict__ A,
                          const T* __restrict__ B,
                          T* __restrict__ C,
                          int M, int N, int K, 
                          const T alpha, const T beta) {

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
        // #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[tx + k * TILE] * Bs[k + ty * TILE];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the computed element to global memory
    if (row < M && col < N)
        C[row + col * M] = alpha * sum + beta * C[row + col * M];
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

