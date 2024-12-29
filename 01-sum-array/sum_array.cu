
__global__ void sumArrayKernel(const int *d_in, int *d_out, unsigned int size) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        // Each thread updates the d_out once
        atomicAdd(d_out, d_in[tid]);
    }
}
