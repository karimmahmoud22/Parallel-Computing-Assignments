%%cuda
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMultiplicationKernel(double *matrix, double *vector, double *result, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[idx * cols + j] * vector[j];
        }
        result[idx] = sum;
    }
}

// Function to read matrix and vector from file
void readMatrixVector(FILE *file, double *matrix, double *vector, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            fscanf(file, "%lf", &matrix[i * cols + j]);
        }
    }
    for (i = 0; i < rows; i++) {
        fscanf(file, "%lf", &vector[i]);
    }
}

int main(int argc, char *argv[]) {
    printf("saving path will be : %s", argv[0]);

    FILE *inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        printf("Error opening input file.\n");
    }

    FILE *outputFile = fopen(argv[2], "w");
    if (outputFile == NULL) {
        printf("Error creating output file.\n");
        fclose(inputFile);
    }

    int numCases;
    fscanf(inputFile, "%d", &numCases);

    while (numCases--) {
        int rows, cols;
        fscanf(inputFile, "%d %d", &rows, &cols);

        // Allocate memory for matrix, vector, and result on host
        double *h_matrix = (double *)malloc(rows * cols * sizeof(double));
        double *h_vector = (double *)malloc(cols * sizeof(double));
        double *h_result = (double *)malloc(rows * sizeof(double));

        // Read matrix and vector from file
        readMatrixVector(inputFile, h_matrix, h_vector, rows, cols);

        // Allocate memory for matrix, vector, and result on device
        double *d_matrix, *d_vector, *d_result;
        cudaMalloc(&d_matrix, rows * cols * sizeof(double));
        cudaMalloc(&d_vector, cols * sizeof(double));
        cudaMalloc(&d_result, rows * sizeof(double));

        // Copy matrix and vector from host to device
        cudaMemcpy(d_matrix, h_matrix, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector, h_vector, cols * sizeof(double), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = 1;
        matrixVectorMultiplicationKernel<<<blocks, threads>>>(d_matrix, d_vector, d_result, rows, cols);

        // Copy result from device to host
        cudaMemcpy(h_result, d_result, rows * sizeof(double), cudaMemcpyDeviceToHost);

        // Write result to output file
        for (int i = 0; i < rows; i++) {
            fprintf(outputFile, "%.1lf\n", h_result[i]);
        }

        // Free memory on device
        cudaFree(d_matrix);
        cudaFree(d_vector);
        cudaFree(d_result);

        // Free memory on host
        free(h_matrix);
        free(h_vector);
        free(h_result);
    }

    fclose(inputFile);
    fclose(outputFile);

    return 0;
}
