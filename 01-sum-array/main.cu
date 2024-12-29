#include <iostream>
#include <cuda_runtime.h>

// Error handling macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error_code = call; \
        if (error_code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_code)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel function (we already wrote this in sum_array.cu)
__global__ void sumArrayKernel(const int *d_in, int *d_out, unsigned int size);

int main() {
    // Size of the array
    unsigned int size = 1000;

    // --- Host Memory Allocation ---
    int *h_in;  // Pointer to host input array
    int *h_out; // Pointer to host output 
    
    h_in = (int *)malloc(size * sizeof(int));
    h_out = (int *)malloc(sizeof(int)); // Only a single value is required

    // --- Device Memory Allocation ---
    int *d_in, *d_out; // Pointers to device input and output arrays
    CUDA_CHECK(cudaMalloc((void **)&d_in, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(int)));

    // Initialize host array
    for (unsigned int i = 0; i < size; i++) {
        h_in[i] = i;
    }
    
    *h_out = 0; // Initialise output to 0

    // --- Copy Data from Host to Device ---
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out, sizeof(int), cudaMemcpyHostToDevice));

    // --- Kernel Launch Configuration ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // --- Launch Kernel ---
    sumArrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, size);
    CUDA_CHECK(cudaGetLastError()); // Check for asynchronous errors during kernel launch

    // --- Copy Result from Device to Host ---
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // --- Print Result ---
    std::cout << "Sum: " << *h_out << std::endl;
    
    // Free host memory
    free(h_in);
    free(h_out);

    // Free device memory 
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
