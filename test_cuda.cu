#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    if (deviceCount > 0) {
        int device;
        error = cudaGetDevice(&device);
        if (error != cudaSuccess) {
            std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        
        cudaDeviceProp props;
        error = cudaGetDeviceProperties(&props, device);
        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        
        std::cout << "Using device " << device << ": " << props.name << std::endl;
        std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    }
    
    return 0;
}
