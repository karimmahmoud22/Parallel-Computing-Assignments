%%cuda
#include <stdio.h>

#define BLOCK_SIZE 256
#define BLOCKS 1

__global__ void sumArray(float *input, int size, float *output) {
    __shared__ float partialSum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;   // no needed as we use only one block

    float sum = 0.0;
    for (int i = global_tid; i < size; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    partialSum[tid] = sum;

    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1 /* efficent way to divide by 2*/) {
        if (tid < stride) partialSum[tid] += partialSum[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = partialSum[0]; // after the last loop, the sum will be at the first element partialSum[0]
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("compiled path: %s \n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", argv[1]);
        return 1;
    }

    float *host_input, *device_input, *device_output;
    int size = 0;
    float totalSum = 0.0;

    // Counting numbers in the input file
    while (fscanf(file, "%*f") != EOF) size++;
    rewind(file);  // return the pointer at the beginning of the file

    host_input = (float *)malloc(size * sizeof(float));
    cudaMalloc(&device_input, size * sizeof(float));
    cudaMalloc(&device_output, ceil(size / (float)BLOCK_SIZE) * sizeof(float));

    // Reading input data from the file
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &host_input[i]);
    }

    fclose(file);

    cudaMemcpy(device_input, host_input, size * sizeof(float), cudaMemcpyHostToDevice);
    sumArray<<<BLOCKS, BLOCK_SIZE>>>(device_input, size, device_output);
    cudaMemcpy(&totalSum, device_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%.1f\n", totalSum);

    // Cleanup
    free(host_input);
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
