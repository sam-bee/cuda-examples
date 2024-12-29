#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error_code = call; \
        if (error_code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_code)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0);

// Kernel (we already wrote this in the previous step)
__global__ void sieveKernel(bool *d_primes, unsigned int max_num, unsigned int prime);

int main() {
    // --- Set Search Space ---
    // long int max_num = 10000000000; // Approximate max with 12GB VRAM
    unsigned int max_num = 100000000;  // Faster running example

    // --- Allocate Host Memory ---
    bool *h_primes = (bool *)malloc((max_num + 1) * sizeof(bool));
    if (h_primes == NULL) {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // --- Initialize Host Array ---
    for (unsigned int i = 0; i <= max_num; i++) {
        h_primes[i] = true; // Initially assume all numbers are prime
    }
    h_primes[0] = false; // 0 is not prime
    h_primes[1] = false; // 1 is not prime

    // --- Allocate Device Memory ---
    bool *d_primes;
    CUDA_CHECK(cudaMalloc((void **)&d_primes, (max_num + 1) * sizeof(bool)));

    // --- Copy Data to Device ---
    CUDA_CHECK(cudaMemcpy(d_primes, h_primes, (max_num + 1) * sizeof(bool), cudaMemcpyHostToDevice));

    // --- Kernel Launch Configuration ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (max_num + threadsPerBlock - 1) / threadsPerBlock;

    // --- Sieve Execution (Outer Loop on CPU) ---
    for (unsigned int p = 2; p <= sqrt(max_num); p++) {
        // Check if p is still marked as prime on the host
        if (h_primes[p]) {
            // Launch kernel to mark multiples of p
            sieveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_primes, max_num, p);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // --- Copy Results Back to Host ---
    CUDA_CHECK(cudaMemcpy(h_primes, d_primes, (max_num + 1) * sizeof(bool), cudaMemcpyDeviceToHost));

    // --- Print Primes ---
    std::cout << "Prime numbers up to " << max_num << " are: " << std::endl;
    int count = 0;
    for (unsigned int i = 2; i <= max_num; i++) {
        if (h_primes[i]) {
            std::cout << i << " ";
            count++;
        }
    }
    std::cout << "\nTotal Primes: " << count << std::endl;

    // --- Free Memory ---
    free(h_primes);
    CUDA_CHECK(cudaFree(d_primes));

    return 0;
}
