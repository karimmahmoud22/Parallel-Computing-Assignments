#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "stb_image_write.h"
#include "stb_image.h"
#include <dirent.h>

#define CHANNELS 3
#define THREADS_PER_BLOCK 16
#include <sys/stat.h>
#include <fstream>

// CUDA kernel to perform 2D convolution with a kernel for each channel separately
__global__ void convolutionKernalNoTiling(unsigned char *in, unsigned char *out, int w, int h, float* d_mask, int MASK_WIDTH) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        float pixVal = 0;

        int N_start_col = Col - (MASK_WIDTH / 2);
        int N_start_row = Row - (MASK_WIDTH / 2);

        // Get the starting index of the surrounding box
        for (int j = 0; j < MASK_WIDTH; ++j) {
            for (int k = 0; k < MASK_WIDTH; ++k) {
                int curRow = N_start_row + j;
                int curCol = N_start_col + k;

                // Verify we have a valid image pixel
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    int index = (curRow * w + curCol) * CHANNELS;

                    // Accumulate the convolution result for each channel
                    for (int c = 0; c < CHANNELS; ++c) {
                        pixVal += in[index + c] * d_mask[j * MASK_WIDTH + k];
                    }
                }
            }
        }

        // Write the average pixel value to the output image
        out[Row * w + Col] = (unsigned char)(pixVal / CHANNELS);
    }
}


// Function to read kernel values from a text file
void readKernelFromFile(const char* file_path, float **kernel, int *mask_width) {
    std::ifstream file(file_path); // Open the file for reading

    if (file.is_open()) {
        // Read the dimension of the mask
        file >> *mask_width;
        if (*mask_width <= 0) {
            printf("Error: Invalid mask width\n");
        }

        // Dynamically allocate memory for the kernel matrix
        *kernel = new float[*mask_width * *mask_width];

        // Read the kernel values row by row
        for (int i = 0; i < *mask_width * *mask_width; ++i) {
            file >> (*kernel)[i];
        }

        file.close(); // Close the file
    } else {
        printf("Error: Unable to open file\n");
    }
}

// Function to free dynamically allocated memory for the kernel matrix
void freeKernelMemory(float *kernel) {
    delete[] kernel;
}


// Function to perform 2D convolution for each channel and sum the results into a single channel
void PerformConvolution(unsigned char *input_image, int width, int height, unsigned char *output_image, float *d_mask, int MASK_WIDTH) {
    // Allocate device memory for input and output images
    unsigned char *d_input_image, *d_output_image;
    cudaMalloc(&d_input_image, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc(&d_output_image, width * height * sizeof(unsigned char));

    // Copy input image from host to device
    cudaMemcpy(d_input_image, input_image, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 ThreadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 BlocksPerGrid((width + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x, (height + ThreadsPerBlock.y - 1) / ThreadsPerBlock.y);

    // Launch the CUDA kernel for each channel separately
    convolutionKernalNoTiling<<<BlocksPerGrid, ThreadsPerBlock>>>(d_input_image, d_output_image, width, height, d_mask, MASK_WIDTH);

    // Copy the result back from device to host
    cudaMemcpy(output_image, d_output_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);
}

// Function to write a single-channel image
void writeSingleChannelImage(const char *filename, unsigned char *image, int width, int height) {
    stbi_write_png(filename, width, height, 1, image, width);
}


int main(int argc, char *argv[]) {
    // printf("Usage: %s \n", argv[0]);

    char *input_folder = argv[1];
    char *output_folder = argv[2];
    int batch_size = (argc >= 4) ? atoi(argv[3]) : 1;


// Read kernel values from file and get mask width
    float *blur_kernel;
    int mask_width;
    readKernelFromFile(argv[4], &blur_kernel, &mask_width);

    // Allocate memory for the kernel matrix on the device
    float *d_mask;
    cudaMalloc(&d_mask, mask_width * mask_width * sizeof(float));

// Copy the kernel matrix from host to device
    cudaMemcpy(d_mask, blur_kernel, mask_width * mask_width * sizeof(float), cudaMemcpyHostToDevice);

    // Open input folder
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_folder)) != NULL) {
        // Iterate over files in the folder
        int count = 0;
        while ((ent = readdir(dir)) != NULL && count < batch_size) {
            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
                // Construct input and output file paths
                char input_file_path[256];
                char output_file_path[256];
                sprintf(input_file_path, "%s/%s", input_folder, ent->d_name);
                sprintf(output_file_path, "%s/%s", output_folder, ent->d_name);

                // Load input image
                int width, height, channels;
                unsigned char *input_image = stbi_load(input_file_path, &width, &height, &channels, STBI_rgb);
                if (!input_image) {
                    printf("Failed to load image: %s\n", input_file_path);
                } else {
                    // Allocate memory for output image
                    unsigned char *output_image = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
                    if (!output_image) {
                        printf("Memory allocation failed for output image\n");
                        stbi_image_free(input_image);
                        continue;
                    }

                    PerformConvolution(input_image, width, height, output_image, d_mask, mask_width);

                    // Write the output image
                    writeSingleChannelImage(output_file_path, output_image, width, height);

                    // Free memory
                    stbi_image_free(input_image);
                    free(output_image);

                    count++;
                }
            }
        }
        closedir(dir);

        // Free dynamically allocated memory for the kernel matrix
    freeKernelMemory(blur_kernel);
    // Free memory allocated for the kernel matrix on the device
    cudaFree(d_mask);
    } else {
        // Could not open directory
        perror("Error opening input folder");
    }

    return 0;
}