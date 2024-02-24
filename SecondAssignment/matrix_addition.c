#include <stdio.h>
#include <stdlib.h>

// Function to read matrix from file
void readMatrix(FILE *file, double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%lf", &matrix[i * cols + j]);
            printf("%lf", &matrix[i * cols + j]);
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

void matrixAddition(double *matrix1, double *matrix2, double *result, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            result[i * cols + j] = matrix1[i * cols + j] + matrix2[i * cols + j];
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

        matrixAddition(matrix1, matrix2, result, rows, cols);

        // Write result to output file
        writeMatrix(outputFile, result, rows, cols);


        // Free memory for matrices
        free(matrix1);
        free(matrix2);
        free(result);
    }

    fclose(inputFile);
    fclose(outputFile);

    return 0;
}
