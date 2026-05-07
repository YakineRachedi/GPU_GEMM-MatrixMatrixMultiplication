#include "utils_matrix.hpp"
#include "env_config.hpp"
#include "GEMM.hpp"
#include "BLAS.hpp"
#include <vector>
#include <random>
#include <cstdlib>
#include <string>
#include <ctime>
#include <cblas.h>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <stdexcept>

#ifndef SCALAR_TYPE
#define SCALAR_TYPE float
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

using Scalar = SCALAR_TYPE;
using namespace std;

int main() {

    mt19937 RNG(time(nullptr));

    constexpr int N_RUNS = 5;

    size_t M         = getEnvSize("GEMM_M", 256);
    size_t N         = getEnvSize("GEMM_N", 256);
    size_t K         = getEnvSize("GEMM_K", 256);
    size_t blockSize = getEnvSize("GEMM_BLOCK", BLOCK_SIZE);

    const char* algo_env = getenv("GEMM_ALGO");
    string algo = algo_env ? algo_env : "classic";

    const Scalar alpha = static_cast<Scalar>(1);
    const Scalar beta  = static_cast<Scalar>(0);

    auto A = GenerateRandomMatrix<Scalar>(M, K, RNG);
    auto B = GenerateRandomMatrix<Scalar>(K, N, RNG);

    Matrix<Scalar> C(M, N, static_cast<Scalar>(0));
    vector<Scalar> c_blas(M * N, static_cast<Scalar>(0));

    if (M <= 8 && N <= 8 && K <= 8) {
        cout << "Matrix A =\n" << A << "\n";
        cout << "Matrix B =\n" << B << "\n";
    }

    cout << "Type  : "
         << (is_same<Scalar, float>::value ? "float" : "double") << "\n";

    cout << "Algo  : " << algo << "\n";
    cout << "Size  : " << M << " x " << N << " x " << K << "\n";
    cout << "Block : " << blockSize << "\n\n";

    double total_time_my   = 0.0;
    double total_time_blas = 0.0;

    Matrix<Scalar> C_my(M, N, static_cast<Scalar>(0));

    for (int run = 0; run < N_RUNS; ++run) {

        // ─────────────────────────────────────────────
        // My GEMM
        // ─────────────────────────────────────────────

        auto start_my = chrono::high_resolution_clock::now();

        if (algo == "block") {
            C_my = block_gemm<Scalar>(A, B, C, alpha, beta, blockSize);
        }
        else {
            C_my = classic_gemm<Scalar>(A, B, C, alpha, beta);
        }

        auto end_my = chrono::high_resolution_clock::now();

        double time_my = chrono::duration<double>(end_my - start_my).count();

        total_time_my += time_my;

        // ─────────────────────────────────────────────
        // OpenBLAS GEMM
        // ─────────────────────────────────────────────

        std::fill(c_blas.begin(), c_blas.end(), static_cast<Scalar>(0));

        auto start_blas = chrono::high_resolution_clock::now();

        blas_gemm<Scalar>(A.data(), B.data(), c_blas.data(), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), alpha, beta);

        auto end_blas = chrono::high_resolution_clock::now();

        double time_blas = chrono::duration<double>(end_blas - start_blas).count();

        total_time_blas += time_blas;

        cout << "Run " << run + 1 << "\n";
        cout << "My GEMM    : " << time_my << " s\n";
        cout << "OpenBLAS   : " << time_blas << " s\n\n";
    }

    double max_err = 0.0;

    const Scalar* my  = C_my.data();
    const Scalar* ref = c_blas.data();

    for (size_t idx = 0; idx < M * N; ++idx) {
        max_err = std::max(
            max_err,
            std::abs(static_cast<double>(my[idx] - ref[idx]))
        );
    }

    cout << "Max error vs OpenBLAS : " << max_err << "\n";

    double avg_my   = total_time_my   / N_RUNS;
    double avg_blas = total_time_blas / N_RUNS;

    double speedup = avg_my / avg_blas;

    cout << "=====================================\n";
    cout << "Average My GEMM   : " << avg_my   << " s\n";
    cout << "Average OpenBLAS  : " << avg_blas << " s\n";
    cout << "Speedup (OpenBLAS vs My GEMM) : "
         << speedup << "x\n";
    cout << "=====================================\n";

    string dir = "benchmarks";
    string filename = dir + "/results.csv";

    filesystem::create_directories(dir);

    bool write_header = false;

    {
        ifstream f(filename);
        write_header = !f.good();
    }

    ofstream file(filename, ios::app);
    
    if (!file.is_open()) {
        cerr << "Error: cannot open " << filename << "\n";
        return 1;
    }

    if (write_header) {
        file << "type;M;N;K;block;algo;"
             << "time_my;time_blas;speedup;max_err\n";
    }

    string type =
        is_same<Scalar, float>::value ? "float" : "double";

    file << type      << ";"
         << M         << ";"
         << N         << ";"
         << K         << ";"
         << blockSize << ";"
         << algo      << ";"
         << avg_my    << ";"
         << avg_blas  << ";"
         << speedup   << ";"
         << max_err   << "\n";

    file.flush();
    file.close();

    return 0;
}