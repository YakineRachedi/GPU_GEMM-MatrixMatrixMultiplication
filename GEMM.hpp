#ifndef GEMM_HPP
#define GEMM_HPP

#include "Matrix.hpp"
#include <algorithm>
#if defined(_OPENMP)
#include <omp.h>
#endif

enum class ProdAlgo { Classic, Block };

static ProdAlgo current_algo = ProdAlgo::Classic;
static size_t blockSize = 128;
inline void setProdMatMat(ProdAlgo algo) {current_algo = algo;}
inline void setBlockSize(size_t size) {blockSize = size;}

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


template<typename T>
    Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.n_cols != B.n_rows) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        switch (current_algo) {
            case ProdAlgo::Classic:
                return classic_gemm(A, B);
            case ProdAlgo::Block:
                return block_gemm(A, B);
        }

        return classic_gemm(A, B);
    }

#endif