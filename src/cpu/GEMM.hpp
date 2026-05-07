#ifndef GEMM_HPP
#define GEMM_HPP

#include "Matrix.hpp"
#include <algorithm>
#if defined(_OPENMP)
#include <omp.h>
#endif


/// @brief Computes the General Matrix Multiplication (GEMM):
///        Out = alpha * A * B + beta * C
///
/// This implementation uses a classical triple-loop algorithm.
/// Matrices are assumed to have compatible dimensions:
/// A: (M x K), B: (K x N), C: (M x N)
///
/// @tparam T Element type (float, double, etc.)
/// @param A Input matrix A (M x K)
/// @param B Input matrix B (K x N)
/// @param C Input matrix C (M x N), used for the beta term
/// @param alpha Scalar multiplier for A * B
/// @param beta Scalar multiplier for C
/// @return Matrix<T> Result matrix Out (M x N)

template<typename T>
    Matrix<T> classic_gemm(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C,
                            const T alpha, const T beta) {
        size_t M = C.n_rows;
        size_t N = C.n_cols;
        size_t K = A.n_cols; // or B.n_rows

        Matrix<T> Out(M, N, 0);

        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < M; ++i) {
                T sum = static_cast<T>(0);         
                for (size_t k = 0; k < K; ++k) {
                    sum += A(i, k) * B(k, j);      
                }
                Out(i, j) = alpha * sum + beta * C(i, j);
            }
        }

        return Out;
    }

/// @brief Compute a sub-block of the matrix product C = A * B
///        and accumulate the result into the output matrix.
/// @tparam T Element type (float, double, etc.)
/// @param iRowBlockA Starting row index of the block in A (and C)
/// @param iColBlockB Starting column index of the block in B (and C)
/// @param iColBlockA Starting column index of the block in A / row block in B
/// @param szBlock Size of the square tile (blockSize)
/// @param A Left-hand matrix (M x K)
/// @param B Right-hand matrix (K x N)
/// @param Out Output matrix (M x N), updated in-place
template<typename T>
    void prodSubBlocks(size_t iRowBlockA, size_t iColBlockB, size_t iColBlockA, size_t szBlock,
                   const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& Out) {
    
        size_t M = A.n_rows;
        size_t N = B.n_cols;
        size_t K = A.n_cols; // or B.n_rows

        const T* a = A.data();
        const T* b = B.data();
        T* out = Out.data();

        for (size_t j = iColBlockB; j < std::min(N, iColBlockB + szBlock); ++j)
            for (size_t k = iColBlockA; k < std::min(K, iColBlockA + szBlock); ++k) {
                T bkj = b[k + j * B.n_rows];
                for (size_t i = iRowBlockA; i < std::min(M, iRowBlockA + szBlock); ++i)
                    out[i + j * M] += a[i + k * M] * bkj;
            }
    }


/// @brief Compute matrix product using blocked (tiled) GEMM
///        and apply scaling: Out = alpha * (A * B) + beta * C
/// @tparam T Element type (float, double, etc.)
/// @param A Left-hand matrix (M x K)
/// @param B Right-hand matrix (K x N)
/// @param C Input matrix used for the beta scaling term
/// @param alpha Scalar applied to A * B
/// @param beta Scalar applied to C
/// @param blockSize Size of square blocks (tiling factor)
/// @return Matrix<T> Result matrix Out

template<typename T>
    Matrix<T> block_gemm(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C,
                        const T alpha, const T beta, size_t blockSize) {
        
        size_t M = C.n_rows;
        size_t N = C.n_cols;
        size_t K = A.n_cols; // or B.n_rows

        Matrix<T> Out(M, N, 0);

        #pragma omp parallel for schedule(dynamic)
        for (size_t j = 0; j < N; j += blockSize)
            for (size_t k = 0; k < K; k += blockSize)
                for (size_t i = 0; i < M; i += blockSize)
                    prodSubBlocks(i, j, k, blockSize, A, B, Out);

        for (size_t j = 0; j < N; ++j)
            for (size_t i = 0; i < M; ++i)
                Out(i, j) = alpha * Out(i, j) + beta * C(i, j);

        return Out;
    }


/// @brief Generic operator* for Matrix<T>
/// @tparam T Element type
/// @param A Left-hand matrix
/// @param B Right-hand matrix
/// @return Matrix<T> Result of A * B
template<typename T>
    Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.n_cols != B.n_rows) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        Matrix<T> C(A.n_rows, B.n_cols, static_cast<T>(0));

        return classic_gemm(A, B, C, static_cast<T>(1), static_cast<T>(0));
    }

#endif