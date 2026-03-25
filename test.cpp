#include "GEMM.hpp"
#include <vector>
#include <random>
#include <ctime>

using namespace std;

/**
 * @brief Generates a random matrix of size Nrows x Ncols.
 * @tparam T Element type (float, double, int, etc.)
 * @param Nrows Number of rows
 * @param Ncols Number of columns
 * @param RNG Random number generator (mt19937)
 * @return Matrix<T> Matrix filled with random values
 */
template<typename T>
    Matrix<T> GenerateRandomMatrix(const int Nrows, const int Ncols, mt19937 & RNG) {
        uniform_int_distribution<int> dist(0, 10);
        vector<T> v(Nrows * Ncols);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<T>(dist(RNG));
        return Matrix<T>(v, Nrows, Ncols);
    }

int main() {
    mt19937 RNG(time(nullptr));
    
    size_t M = 3;
    size_t N = 3;
    size_t K = 3;

    auto A = GenerateRandomMatrix<float>(M, K, RNG);
    auto B = GenerateRandomMatrix<float>(K, N, RNG);
    
    cout << "Matrix A = \n" << A << "\n";
    cout << "Matrix B = \n" << B << "\n";

    setProdMatMat(ProdAlgo::Classic);
    auto C1 = A * B;

    setProdMatMat(ProdAlgo::Block);
    auto C2 = A * B;

    std::cout << "Classic calculs \n" << C1 << "\n";
    std::cout << "Block calculs \n"<< C2 << "\n";

    return 0;
}