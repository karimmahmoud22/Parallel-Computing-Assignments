%%cuda
#include <stdio.h>

#define BLOCK_SIZE 256
#define BLOCKS 1
#define MAX_INPUT_SIZE 10000

__global__ void binarySearch(float *input, int input_size, float target, int *output) {
    __shared__ int shared_result;
    __shared__ float shared_input[MAX_INPUT_SIZE];

    int tid = threadIdx.x;
    int start = 0;
    int end = input_size - 1;
    int mid;
    shared_result = -1;

    // Determine chunk size for each thread
    int chunk_size = (input_size + blockDim.x - 1) / blockDim.x;
    int chunk_start = tid * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, input_size);

    // Load input into shared memory chunk by chunk
    for (int i = chunk_start; i < chunk_end; ++i) {
        if (i < input_size) {
            shared_input[i] = input[i];
        }
    }

    __syncthreads();

    // Binary search within the chunk
    while (start <= end) {
        mid = (start + end) / 2;
        if (shared_input[mid] == target) {
            shared_result = mid;
            break;
        } else if (shared_input[mid] < target) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
        __syncthreads(); // Ensure all threads update start and end before continuing
    }

    // Update shared result if target found
    *output = shared_result;
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("compiled path: %s \n", argv[0]);
        return 1;
    }

    float target = atof(argv[2]);

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", argv[1]);
        return 1;
    }

    float *host_input, *device_input;
    int size = 0;
    int *device_output;

    // Counting numbers in the input file
    while (fscanf(file, "%*f") != EOF) size++;
    rewind(file);

    host_input = (float *)malloc(size * sizeof(float));
    cudaMalloc(&device_input, size * sizeof(float));
    cudaMalloc(&device_output, sizeof(int));

    // Reading input data from the file
    for (int i = 0; i < size; i++) {
        fscanf(file, "%f", &host_input[i]);
    }

    fclose(file);

    cudaMemcpy(device_input, host_input, size * sizeof(float), cudaMemcpyHostToDevice);
    binarySearch<<<BLOCKS, BLOCK_SIZE>>>(device_input, size, target, device_output);

    int gpu_result;
    cudaMemcpy(&gpu_result, device_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", gpu_result);

    // Cleanup
    free(host_input);
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}
