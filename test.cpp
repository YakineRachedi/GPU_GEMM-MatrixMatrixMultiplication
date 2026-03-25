#include "GEMM.hpp"
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>

using namespace std;

int main() {
    mt19937 RNG(time(nullptr));
    
    size_t M = getEnvSize("GEMM_M", 3);
    size_t N = getEnvSize("GEMM_N", 3);
    size_t K = getEnvSize("GEMM_K", 3);

    auto A = GenerateRandomMatrix<float>(M, K, RNG);
    auto B = GenerateRandomMatrix<float>(K, N, RNG);
    
    cout << "Matrix A = \n" << A << "\n";
    cout << "Matrix B = \n" << B << "\n";
    cout << "Algo: " << (current_algo == ProdAlgo::Classic ? "Classic" : "Block")
          << ", Block size: " << blockSize << "\n";
    auto C = A * B;
    cout << "Matrix C = A * B \n" << C << "\n";
    return 0;
}