
__global__ void sieveKernel(bool *d_primes, unsigned int max_num, unsigned int prime) {

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Will not find multiples of current prime below prime^2
    unsigned int start = prime * prime;

    // Offset added to tid to determine which element to update
    unsigned int x = (max_num - start) / prime;

    // Stride is the same for all threads in this configuration
    unsigned int stride = blockDim.x * gridDim.x;

    // Determine appropriate start location, offset by thread ID
    unsigned int i = start + (tid <= x ? tid : x) * prime;

    for (; i <= max_num; i += stride * prime) {
        d_primes[i] = false;
    }
}
