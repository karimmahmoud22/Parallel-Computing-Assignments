

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "stb_image_write.h"
#include "stb_image.h"
#include <dirent.h>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>

#define CHANNELS 3
#include <sys/stat.h>
#include <fstream>

#define BLOCK_WIDTH 16

// a : height of the input image
// b : width of the input image
__global__ void tiledConvolution(unsigned char* input_image, unsigned char* output_image, int a, int b, float* d_mask, int maskWidth, int N_TILE_WIDTH)
{

    // define and initialize the variables that will be used for indexing - this is for brevity
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int n_row = blockIdx.y * N_TILE_WIDTH + ty;
    int n_col = blockIdx.x * N_TILE_WIDTH + tx;

    int m_row = n_row - maskWidth / 2;
    int m_col = n_col - maskWidth / 2;

    // define shared memory input array tile
    __shared__ unsigned char tile_red[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ unsigned char tile_green[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ unsigned char tile_blue[BLOCK_WIDTH][BLOCK_WIDTH];

    // if the input array index variables are within the bounds of the input array
    // then load the elements of input_image into their respective positions in the tile
    // otherwise just set the element of the tile to 0 (the element becomes a "ghost" element)
    int input_index = (m_row * b + m_col) * 3;

    if(m_row >= 0 && m_row < a && m_col >= 0 && m_col < b)
    {
        tile_red[ty][tx] = input_image[input_index];
        tile_green[ty][tx] = input_image[input_index + 1];
        tile_blue[ty][tx] = input_image[input_index + 2];
    }
    else
    {
        tile_red[ty][tx] = 0;
        tile_green[ty][tx] = 0;
        tile_blue[ty][tx] = 0;
    }


    __syncthreads();


    if(ty < N_TILE_WIDTH && tx < N_TILE_WIDTH && n_row < a && n_col < b)
    {
        int value_red = 0;
        int value_green = 0;
        int value_blue = 0;

        // calculate value of result element
        for(int i = 0; i < maskWidth; ++i)
        {
            int mask_indexing = i * maskWidth;
            for(int j = 0; j < maskWidth; ++j)
            {
                int tile_indexing_y = ty + i;
                int tile_indexing_x = tx + j;
                int mask_value = d_mask[mask_indexing + j];
                value_red += mask_value * tile_red[tile_indexing_y][tile_indexing_x];
                value_green += mask_value * tile_green[tile_indexing_y][tile_indexing_x];
                value_blue += mask_value * tile_blue[tile_indexing_y][tile_indexing_x];
            }
        }

        // write result variable to corresponding element of result array
        output_image[n_row * b + n_col] = ((unsigned char)(value_red) + (unsigned char)(value_green) + (unsigned char)(value_blue)) / 3;
    }
}



// Function to read kernel values from a text file
int readKernelFromFile(const char* file_path, float **kernel, int *mask_width) {
    std::ifstream file(file_path); // Open the file for reading

    if (file.is_open()) {
        // Read the dimension of the mask
        file >> *mask_width;
        if (*mask_width <= 0) {
            printf("Error: Invalid mask width\n");
            return 0; // Return 0 to indicate failure
        }

        // Dynamically allocate memory for the kernel matrix
        *kernel = new float[*mask_width * *mask_width];

        // Read the kernel values row by row
        for (int i = 0; i < *mask_width * *mask_width; ++i) {
            file >> (*kernel)[i];
        }

        file.close(); // Close the file
        return 1; // Return 1 to indicate success
    } else {
        printf("Error: Unable to open file\n");
        return 0; // Return 0 to indicate failure
    }
}

// Function to free dynamically allocated memory for the kernel matrix
void freeKernelMemory(float *kernel) {
    delete[] kernel;
}


// Function to write a single-channel image
void writeImage(const char *filename, unsigned char *image, int width, int height) {
    // stbi_write_png(filename, width, height, 1, image, width);
    stbi_write_jpg(filename, width, height, 1, image, 100);
}


int main(int argc, char *argv[]) {
    // printf("Usage: %s \n", argv[0]);

    char *input_folder = argv[1];
    char *output_folder = argv[2];
    int batch_size = (argc >= 4) ? atoi(argv[3]) : 1;


// Read kernel values from file and get mask width
    float *blur_kernel;
    int mask_width;
    if (!readKernelFromFile(argv[4], &blur_kernel, &mask_width)) {
        printf("Failed to read kernel from file\n");
        return 1;
    }

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
                unsigned char *input_image = stbi_load(input_file_path, &width, &height, &channels, 0);
                if (!input_image) {
                    printf("Failed to load image: %s\n", input_file_path);
                } else {
                    //printf("Image loaded successfully: %s, width = %d, height = %d, channels = %d\n", input_file_path, width, height, channels);

                    // Allocate memory for output image
                    unsigned char *output_image = (unsigned char*)malloc(width * height * sizeof(unsigned char) );
                    if (!output_image) {
                        printf("Memory allocation failed for output image\n");
                        stbi_image_free(input_image);
                        continue;
                    }

                    // Perform 2D convolution on each channel
                    ///////////
                    // Allocate device memory for input and output images
                    unsigned char *d_input_image, *d_output_image;
                    cudaMalloc(&d_input_image, width * height * channels);
                    cudaMalloc(&d_output_image, width * height);

                    // Copy input image from host to device
                    cudaMemcpy(d_input_image, input_image, width * height * channels, cudaMemcpyHostToDevice);

                    // Calculate grid and block dimensions
                    int N_TILE_WIDTH = BLOCK_WIDTH - (mask_width - 1);
                    dim3 ThreadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
                    dim3 BlocksPerGrid(ceil(width / (float) N_TILE_WIDTH), ceil(height / (float) N_TILE_WIDTH), 1);

                    // Launch the CUDA kernel for each channel separately
                    
                    tiledConvolution<<<BlocksPerGrid, ThreadsPerBlock>>>(d_input_image, d_output_image, height, width, d_mask, mask_width, N_TILE_WIDTH);

                    // Copy the result back from device to host
                    cudaMemcpy(output_image, d_output_image, width * height, cudaMemcpyDeviceToHost);

                    // Free device memory
                    cudaFree(d_input_image);
                    cudaFree(d_output_image);
                    ////////////

                    // Write the output image
                    writeImage(output_file_path, output_image, width, height);

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
        return 1;
    }

    return 0;
}