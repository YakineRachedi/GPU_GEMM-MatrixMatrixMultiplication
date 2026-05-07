#include <cstdlib>
#include <string>


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
    const char* val = getenv(name);
    return val ? static_cast<size_t>(atoi(val)) : default_val;
}

