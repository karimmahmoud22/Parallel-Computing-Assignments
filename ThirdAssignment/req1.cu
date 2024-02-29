%%cuda
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_THREADS 256
#define BLOCKS 1
__global__ void sumArray(float *d_array, float *d_sum, int N) {
    __shared__ float partialSum[BLOCK_THREADS];
    unsigned int t = threadIdx.x;
    unsigned int chunkStart = blockIdx.x * blockDim.x;  // start of the data chuck for the current thread
    
    // Each thread loads an element into shared memory
    if (chunkStart + t < N) partialSum[t] = d_array[chunkStart + t];
    
    // Ensure all threads have loaded their elements
    __syncthreads();

    // Compute the sum within the block
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (t % (2 * stride) == 0)  partialSum[t] += partialSum[t + stride];
        __syncthreads();
    }
    
    // Write the block sum to global memory ( t==0 , just to ensure that only one thread will write the sum in the gpu memory)
    if (t == 0) d_sum[blockIdx.x] = partialSum[0];
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Compiled file path %s: \n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    float *h_array, *d_array, *d_sum;
    float sum = 0;
    int N = 0;
    float dummy = 0;

    // Read array elements from file
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return 1;
    }
    // count the required size of array
    while (fscanf(file, "%f", &dummy) == 1) N++;
    fclose(file);

    // Allocate memory on host
    h_array = (float*)malloc(N * sizeof(float));

    // Read array elements from file
    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        fscanf(file, "%f", &h_array[i]);
    }
    fclose(file);

    // Allocate memory on device
    cudaMalloc(&d_array, N * sizeof(float));
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

    // kernel
    sumArray<<<BLOCKS, BLOCK_THREADS>>>(d_array, d_sum, N);
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%.1f\n", sum);

    // Free memory
    free(h_array);
    cudaFree(d_array);
    cudaFree(d_sum);

    return 0;
}