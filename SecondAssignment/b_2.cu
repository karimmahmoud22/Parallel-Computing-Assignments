
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to read matrix from file
void readMatrix(FILE *file, double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%lf", &matrix[i * cols + j]);
        }
    }
}

// Function to write matrix to file
void writeMatrix(FILE *file, double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.2lf ", matrix[i * cols + j]);
        }
        fprintf(file, "\n");
    }
}

__global__ void MatAdd(double *A, double *B, double *C, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            C[index] = A[index] + B[index];
        }
    }
    
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("Usage: %s nrows ncols nrows*ncols numbers\n", argv[0]);
        return 1;
    }

    FILE *inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        printf("Error opening input file.\n");
        return 1;
    }

    FILE *outputFile = fopen(argv[2], "w");
    if (outputFile == NULL) {
        printf("Error creating output file.\n");
        fclose(inputFile);
        return 1;
    }

    int numCases;
    fscanf(inputFile, "%d", &numCases);

    while (numCases--) {
        int rows, cols;
        fscanf(inputFile, "%d %d", &rows, &cols);

        // Allocate memory for matrices on host
        double *matrix1 = (double *)malloc(rows * cols * sizeof(double));
        double *matrix2 = (double *)malloc(rows * cols * sizeof(double));
        double *result = (double *)malloc(rows * cols * sizeof(double));

        // Read matrices from file
        readMatrix(inputFile, matrix1, rows, cols);
        readMatrix(inputFile, matrix2, rows, cols);

        // Allocate memory for matrices on device
        double *matrix1_d, *matrix2_d, *result_d;
        cudaMalloc((void**)&matrix1_d, sizeof(double) * cols * rows);
        cudaMalloc((void**)&matrix2_d, sizeof(double) * cols * rows);
        cudaMalloc((void**)&result_d, sizeof(double) * cols * rows);

        // Transfer data from host to device memory
        cudaMemcpy(matrix1_d, matrix1, sizeof(double) * cols * rows, cudaMemcpyHostToDevice);
        cudaMemcpy(matrix2_d, matrix2, sizeof(double) * cols * rows, cudaMemcpyHostToDevice);

        // Perform matrix addition
        // Kernel invocation with one block of N * N * 1 threads
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((rows-1) / threadsPerBlock.x + 1, (cols-1) / threadsPerBlock.y + 1);
        MatAdd<<<numBlocks, threadsPerBlock>>>(matrix1_d, matrix2_d, result_d, rows, cols);

        // Transfer data from device to host memory
        cudaMemcpy(result, result_d, sizeof(double) * cols * rows, cudaMemcpyDeviceToHost);

        // Write result to output file
        writeMatrix(outputFile, result, rows, cols);

        // Deallocate device memory
        cudaFree(matrix1_d);
        cudaFree(matrix2_d);
        cudaFree(result_d);

        // Free memory for matrices
        free(matrix1);
        free(matrix2);
        free(result);
    }

    fclose(inputFile);
    fclose(outputFile);

    return 0;
}