#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    std::cout << "cudaGetDeviceCount: " << cudaGetErrorString(error) << std::endl;
    return error != cudaSuccess ? 1 : 0;
}
