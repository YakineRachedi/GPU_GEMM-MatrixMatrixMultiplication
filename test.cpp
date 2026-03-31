#include "GEMM.hpp"
#include <vector>
#include <random>
#include <cstdlib>
#include <string>
#include <ctime>
#include <cblas.h>
#include <chrono>
#include <fstream>

#ifndef SCALAR_TYPE
#define SCALAR_TYPE float
#endif

using Scalar = SCALAR_TYPE;
using namespace std;

template<typename T>void blas_gemm(const T* A, const T* B, T* C, int M, int N, int K);

template<> void blas_gemm<float>(const float* A, const float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);}

template<>void blas_gemm<double>(const double* A, const double* B, double* C, int M, int N, int K) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, M, B, K, 0.0, C, M);}

int main() {
    mt19937 RNG(time(nullptr));

    size_t M = getEnvSize("GEMM_M", 3);
    size_t N = getEnvSize("GEMM_N", 3);
    size_t K = getEnvSize("GEMM_K", 3);

    auto A = GenerateRandomMatrix<Scalar>(M, K, RNG);
    auto B = GenerateRandomMatrix<Scalar>(K, N, RNG);

    cout << "Matrix A =\n" << A << "\n";
    cout << "Matrix B =\n" << B << "\n";
    cout << "Type: " << (is_same<Scalar, float>::value ? "float" : "double") << "\n\n";

    // ── My GEMM ─────────────────────────────────────────────────
    auto start_my = chrono::high_resolution_clock::now();
    auto C_my = A * B;
    auto end_my = chrono::high_resolution_clock::now();
    cout << "Matrix C (my GEMM) =\n" << C_my << "\n";

    // ── OpenBLAS ─────────────────────────────────────────────────
    vector<Scalar> c_blas(M * N, static_cast<Scalar>(0));

    auto start_blas = chrono::high_resolution_clock::now();
    blas_gemm<Scalar>(A.data(), B.data(), c_blas.data(), (int)M, (int)N, (int)K);
    auto end_blas = chrono::high_resolution_clock::now();

    // ── Time ────────────────────────────────────────────────────
    double time_my   = chrono::duration<double>(end_my   - start_my).count();
    double time_blas = chrono::duration<double>(end_blas - start_blas).count();
    string filename = "results.csv";
    ofstream file(filename, std::ios::app);
    string type = is_same<Scalar, float>::value ? "float" : "double";
    string algo = getenv("GEMM_ALGO") ? getenv("GEMM_ALGO") : "unknown";
    string block = getenv("GEMM_BLOCK") ? getenv("GEMM_BLOCK") : "0";

      file << type << ";"
      << M << ";"
      << block << ";"
      << algo << ";"
      << time_my << ";"
      << time_blas << "\n";

      file.flush();
      file.close();

    return 0;
}