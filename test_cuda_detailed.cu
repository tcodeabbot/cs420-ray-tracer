#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // Try to reset the device first
    cudaError_t error = cudaDeviceReset();
    printf("cudaDeviceReset: %s\n", cudaGetErrorString(error));
    
    // Get last error
    error = cudaGetLastError();
    printf("cudaGetLastError: %s\n", cudaGetErrorString(error));
    
    // Try device count
    int deviceCount;
    error = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount: %s (count=%d)\n", cudaGetErrorString(error), deviceCount);
    
    // Try setting device explicitly
    error = cudaSetDevice(0);
    printf("cudaSetDevice(0): %s\n", cudaGetErrorString(error));
    
    return 0;
}
