#ifndef BLAS_HPP
#define BLAS_HPP

#include <cblas.h>

/**
 * @brief BLAS wrapper for General Matrix Multiplication (GEMM).
 *
 * Computes:
 *      C = alpha * A * B + beta * C
 *
 * Matrices are assumed to be stored in Column-Major format
 * (BLAS/LAPACK convention).
 *
 * Matrix dimensions:
 *      A : (M x K)
 *      B : (K x N)
 *      C : (M x N)
 *
 * Specialized implementations are provided for float and double.
 *
 * @tparam T Scalar type (float or double)
 * @param[in]  A      Pointer to matrix A
 * @param[in]  B      Pointer to matrix B
 * @param[in,out] C   Pointer to matrix C
 * @param[in]  M      Number of rows of A and C
 * @param[in]  N      Number of columns of B and C
 * @param[in]  K      Shared dimension between A and B
 * @param[in]  alpha  Scalar multiplier applied to A * B
 * @param[in]  beta   Scalar multiplier applied to C
 */
template<typename T> void blas_gemm(const T* A, const T* B, T* C, int M, int N, int K, const T alpha, const T beta);


/**
 * @brief Single precision BLAS GEMM wrapper.
 *
 * Uses OpenBLAS SGEMM implementation.
 */
template<>
    inline void blas_gemm<float>(const float* A, const float* B, float* C, int M, int N, int K,
                                                const float alpha, const float beta) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
}


/**
 * @brief Double precision BLAS GEMM wrapper.
 *
 * Uses OpenBLAS DGEMM implementation.
 */
template<>
    inline void blas_gemm<double>(const double* A, const double* B, double* C, int M, int N, int K, 
                                                const double alpha, const double beta) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
}

#endif