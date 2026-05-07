#ifndef UTILS_MATRIX_INCLUDED
#define UTILS_MATRIX_INCLUDED

#pragma once

#include <random>
#include <vector>
#include <ctime>
#include "Matrix.hpp"

/**
 * @brief Generates a random matrix of size Nrows x Ncols.
 * @tparam T Element type (float, double, int, etc.)
 * @param Nrows Number of rows
 * @param Ncols Number of columns
 * @param RNG Random number generator (mt19937)
 * @return Matrix<T> Matrix filled with random values
 */
template<typename T>
    Matrix<T> GenerateRandomMatrix(const int Nrows, const int Ncols, std::mt19937 & RNG) {
        std::uniform_int_distribution<int> dist(0, 10);
        std::vector<T> v(Nrows * Ncols);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<T>(dist(RNG));
        return Matrix<T>(v, Nrows, Ncols);
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



template<typename T>
    void fill_random(std::vector<T>& A) {
        std::mt19937 gen(0);
        std::uniform_real_distribution<T> dist(0, 1);

        for (auto& x : A) x = dist(gen);
    }


#endif