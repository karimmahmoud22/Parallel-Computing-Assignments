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
#define TILE_SIZE 16


__global__ void convolutionTiledOutputKernel_2(unsigned char * input_image, unsigned char * out, int w, int h, float* mask,  int maskwidth) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    int Col_start = blockIdx.x * blockDim.x ;
    int Row_start = blockIdx.y * blockDim.y ;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tiling_size = TILE_SIZE + maskwidth - 1;
    extern __shared__ unsigned char shared_memory[];

    unsigned char *tile_red = shared_memory;
    unsigned char *tile_green  = tile_red + tiling_size * tiling_size; 
    unsigned char *tile_blue  = tile_green + tiling_size * tiling_size; 

    // Loop to load multiple elements per thread

    int x_index, y_index, tile_idx,temp;
    for (int i = ty; i < TILE_SIZE + maskwidth - 1; i += blockDim.y) {
            for (int j = tx; j < TILE_SIZE + maskwidth - 1; j += blockDim.x) {
                x_index = Col_start - (maskwidth / 2) + j;
                y_index = Row_start - (maskwidth / 2) + i;

                // Calculate the linear index for accessing shared memory
                tile_idx =  i * tiling_size  + j;

                // Check bounds and load data into tiles
                if (x_index >= 0 && x_index < w && y_index >= 0 && y_index < h) {
                    temp = (y_index * w + x_index) * 3;
                    tile_red[tile_idx] = input_image[temp];
                    tile_green[tile_idx] = input_image[temp + 1];
                    tile_blue[tile_idx] = input_image[temp + 2];
                } else {
                    tile_red[tile_idx] = 0;
                    tile_green[tile_idx] = 0;
                    tile_blue[tile_idx] = 0;
                }
            }
        }
    __syncthreads();

    // Perform convolution
    if (Col < w && Row < h) {
        int r_value = 0;
        int g_value = 0;
        int b_value = 0;
        int temp;
        for(int j = 0; j < maskwidth; ++j) {
            for(int k = 0; k < maskwidth; ++k) {
                tile_idx = (ty + j)* tiling_size  + (tx + k);
                temp = mask[j * maskwidth + k];
                r_value += tile_red[tile_idx] * temp;
                g_value += tile_green[tile_idx] * temp;
                b_value += tile_blue[tile_idx] * temp;
            }
        }

        // Write the new pixel values to the output
        out[Row * w + Col] = ((unsigned char)(r_value) + (unsigned char)(g_value) + (unsigned char)(b_value)) / 3;
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
    //printf("Usage: %s \n", argv[0]);

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
                    unsigned char *output_image = (unsigned char*)malloc(width * height  * sizeof(unsigned char));
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
                    dim3 ThreadsPerBlock(TILE_SIZE, TILE_SIZE);
                    dim3 BlocksPerGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

                    // Launch the CUDA kernel for each channel separately
                    int tiling_sizee =  TILE_SIZE + mask_width - 1;
                    int sharedMemSize = (tiling_sizee * tiling_sizee) * sizeof(unsigned char) * 3;
                    convolutionTiledOutputKernel_2<<<BlocksPerGrid, ThreadsPerBlock, sharedMemSize>>>(d_input_image, d_output_image, width, height, d_mask, mask_width);

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