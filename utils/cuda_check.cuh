#include <stdlib.h>
#include <cuda_runtime.h>


// Function that catches the error 
void CUDA_CHECK(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (CUDA_CHECK(error, __FILE__ , __LINE__))


// Function that checks error between gpu result and cpu ref result
template<typename T>
    bool check_error(const std::vector<T>& ref,
                    const std::vector<T>& gpu) {

        double max_err = 0.0;

        for (size_t i = 0; i < ref.size(); ++i) {
            double err = std::abs(ref[i] - gpu[i]);
            max_err = std::max(max_err, err);
        }

        std::cout << "Max error: " << max_err << "\n";
        return max_err < 1e-4;
    }