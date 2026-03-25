#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <stdexcept>

/**
 * @brief Dense matrix class with column-major storage.
 *
 * Storage layout follows BLAS/LAPACK conventions:
 * element (i, j) is stored at index i + j * n_rows.
 *
 */


template<typename T> class Matrix; 
template<typename T> std::ostream & operator<<(std::ostream &, const Matrix<T> &);
template<typename T> Matrix<T> operator+(const Matrix<T> &, const Matrix<T> &);
template<typename T> Matrix<T> operator-(const Matrix<T> &, const Matrix<T> &);
template<typename T> Matrix<T> operator*(const Matrix<T> &, const Matrix<T> &);

template<typename T>
    class Matrix {
        private:
            std::vector<T> data_;   // Contiguous column-major storage

        public:
            size_t n_rows;          
            size_t n_cols; 
            /**
             * @brief Default constructor
             * Creates an empty matrix (0x0)
             */
            Matrix() : n_rows(0), n_cols(0) {}

            /**
             * @brief Destructor
             * Default is sufficient (std::vector handles memory)
             */
            ~Matrix() = default;

            /**
             * @brief Copy constructor (deleted)
             * Prevents expensive deep copies
             */
            Matrix(const Matrix& A) = delete;

            /**
             * @brief Move constructor
             * Transfers ownership of data (no copy)
             */
            Matrix(Matrix && A) noexcept = default;

            /**
             * @brief Copy assignment (deleted)
             * Prevents accidental copies
             */
            Matrix & operator=(const Matrix & A) = delete;

            /**
             * @brief Move assignment
             * Efficient transfer of resources
             */
            Matrix & operator=(Matrix && A) noexcept = default;

            /**
             * @brief Construct matrix from raw data
             * @param data Input data (must be size rows * cols)
             * @param rows Number of rows
             * @param cols Number of columns
             */
            Matrix(const std::vector<T> & data, size_t rows, size_t cols)
                : data_(data), n_rows(rows), n_cols(cols)
            {
                if (data.size() != rows * cols) {
                    throw std::invalid_argument("Data size does not match matrix dimensions");
                }
            }

            /**
             * @brief Construct uninitialized matrix
             * @param rows Number of rows
             * @param cols Number of columns
             */
            Matrix(size_t rows, size_t cols)
                : data_(rows * cols), n_rows(rows), n_cols(cols) {}

            /**
             * @brief Construct matrix filled with a constant value
             * @param rows Number of rows
             * @param cols Number of columns
             * @param value Initial value
             */
            Matrix(size_t rows, size_t cols, const T& value)
                : data_(rows * cols, value), n_rows(rows), n_cols(cols) {}
            /**
             * @brief Access element (read-only)
             * @param row Row index (i)
             * @param col Column index (j)
             *
             * Column-major indexing:
             * index = row + col * n_rows
             */
            T operator()(size_t row, size_t col) const {
                return data_[row + col * n_rows];
            }

            /**
             * @brief Access element (read/write)
             */
            T& operator()(size_t row, size_t col) {
                return data_[row + col * n_rows];
            }

            /**
             * @brief Return raw pointer
             */
            const T* data() const { return data_.data(); }
            T* data() { return data_.data(); }

            /**
             * @brief Leading dimension
             *
             * In column-major format, lda = number of rows.
             */
            size_t lda() const { return n_rows; }

            /**
             * @brief Pretty-print (for debugging only)
             */
            friend std::ostream & operator<< <>(std::ostream &, const Matrix<T> &);

            /**
             * @brief Simple arithmetic operators
             */
            friend Matrix<T> operator+ <>(const Matrix<T> &, const Matrix<T> &);
            friend Matrix<T> operator- <>(const Matrix<T> &, const Matrix<T> &);
        }; 


template<typename T> 
    std::ostream & operator<<(std::ostream & os, const Matrix<T> & mat) {
        if (mat.n_rows > 10 || mat.n_cols > 10) {
            throw std::runtime_error("Matrix too large to print");
        }

        for (size_t i = 0; i < mat.n_rows; ++i) {
            for (size_t j = 0; j < mat.n_cols; ++j) {
                os << mat(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }

template<typename T>
    Matrix<T> operator+(const Matrix<T> & A, const Matrix<T> & B) {
        if (A.n_rows != B.n_rows || A.n_cols != B.n_cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        std::vector<T> result(A.n_rows * A.n_cols);

        const T* a = A.data();
        const T* b = B.data();

        for (size_t k = 0; k < result.size(); ++k) {
            result[k] = a[k] + b[k];
        }

        return Matrix<T>(result, A.n_rows, A.n_cols);
}

template<typename T>
    Matrix<T> operator-(const Matrix<T> & A, const Matrix<T> & B) {
        if (A.n_rows != B.n_rows || A.n_cols != B.n_cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        std::vector<T> result(A.n_rows * A.n_cols);

        const T* a = A.data();
        const T* b = B.data();

        for (size_t k = 0; k < result.size(); ++k) {
            result[k] = a[k] - b[k];
        }

        return Matrix<T>(result, A.n_rows, A.n_cols);
}

#endif // MATRIX_HPP