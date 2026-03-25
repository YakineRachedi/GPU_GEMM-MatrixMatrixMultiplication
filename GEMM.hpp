#ifndef GEMM_HPP
#define GEMM_HPP

#include "Matrix.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>
#if defined(_OPENMP)
#include <omp.h>
#endif

enum class ProdAlgo { Classic, Block }; // Algorithm type for GEMM
static ProdAlgo current_algo = ProdAlgo::Classic; // Default selected GEMM algorithm
static size_t blockSize = 128; // Default value

/// @brief Set the block size for block GEMM
/// @param size Block size (number of rows/cols per block)
inline void setBlockSize(size_t size) {blockSize = size;}


/// @brief Initialize GEMM configuration from environment variables
/// 
/// Reads the following environment variables:
/// - GEMM_ALGO: "classic" or "block"
/// - GEMM_BLOCK: block size for blocked GEMM
inline void initGEMMConfig() {
    const char* algo = std::getenv("GEMM_ALGO");
    const char* block = std::getenv("GEMM_BLOCK");

    if (algo) {
        std::string s(algo);
        if (s == "classic") current_algo = ProdAlgo::Classic;
        else if (s == "block") current_algo = ProdAlgo::Block;
    }

    if (block) {
        blockSize = std::stoul(block);
    }
}


/// @brief Get a size_t value from an environment variable or default
/// @param name Environment variable name
/// @param default_val Value to use if the variable is not set
/// @return Value of environment variable converted to size_t, or default_val
size_t getEnvSize(const char* name, size_t default_val) {
    const char* val = std::getenv(name);
    if (val) return std::stoul(val);
    return default_val;
}


/// @brief Compute matrix product using classic (naive) triple loop
/// @tparam T Element type (float, double, int, etc.)
/// @param A Left-hand matrix
/// @param B Right-hand matrix
/// @return Matrix<T> Result of A * B

template<typename T>
    Matrix<T> classic_gemm(const Matrix<T> & A, const Matrix<T> & B) {
        size_t M = A.n_rows;
        size_t N = B.n_cols;
        size_t K = A.n_cols;

        Matrix<T> C(M, N, 0);

        const T* a = A.data();
        const T* b = B.data();
        T* c = C.data();

        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < K; ++k)
                for (size_t i = 0; i < M; ++i)
                    c[i + j * M] += a[i + k * M] * b[k + j * B.n_rows];

        return C;
    }


/// @brief Compute a sub-block of the blocked GEMM
/// @tparam T Element type
/// @param iRowBlockA Starting row index in A
/// @param iColBlockB Starting column index in B / C
/// @param iColBlockA Starting column index in A / row index in B
/// @param szBlock Size of the block
/// @param A Left-hand matrix
/// @param B Right-hand matrix
/// @param C Matrix to accumulate results
template<typename T>
    void prodSubBlocks(size_t iRowBlockA, size_t iColBlockB, size_t iColBlockA, size_t szBlock,
                    const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
        size_t M = A.n_rows;
        size_t K = A.n_cols;
        size_t N = B.n_cols;

        const T* a = A.data();
        const T* b = B.data();
        
        T* c = C.data();

        for (size_t j = iColBlockB; j < std::min(N, iColBlockB + szBlock); ++j)
            for (size_t k = iColBlockA; k < std::min(K, iColBlockA + szBlock); ++k) {
                T bkj = b[k + j * B.n_rows];
                for (size_t i = iRowBlockA; i < std::min(M, iRowBlockA + szBlock); ++i)
                    c[i + j * M] += a[i + k * M] * bkj;
            }
    }


/// @brief Compute matrix product using blocked GEMM
/// @tparam T Element type
/// @param A Left-hand matrix
/// @param B Right-hand matrix
/// @return Matrix<T> Result of A * B
template<typename T>
    Matrix<T> block_gemm(const Matrix<T>& A, const Matrix<T>& B) {
        size_t M = A.n_rows;
        size_t N = B.n_cols;
        size_t K = A.n_cols;

        Matrix<T> C(M, N, 0);

        #pragma omp parallel for
        for (size_t j = 0; j < N; j += blockSize)
            for (size_t k = 0; k < K; k += blockSize)
                for (size_t i = 0; i < M; i += blockSize)
                    prodSubBlocks(i, j, k, blockSize, A, B, C);

        return C;
    }



/// @brief Generic operator* for Matrix<T>
/// Dispatches to classic or block GEMM depending on current_algo
/// @tparam T Element type
/// @param A Left-hand matrix
/// @param B Right-hand matrix
/// @return Matrix<T> Result of A * B
template<typename T>
    Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.n_cols != B.n_rows) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        initGEMMConfig();
        switch (current_algo) {
            case ProdAlgo::Classic:
                return classic_gemm(A, B);
            case ProdAlgo::Block:
                return block_gemm(A, B);
        }

        return classic_gemm(A, B);
    }

#endif