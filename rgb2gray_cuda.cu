#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define MAX_PATH 255
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

__global__ void rgba_to_grayscale(uint8_t *d_rgba_image, uint8_t *d_gray_image, int image_width, int image_height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y > image_height || x > image_width) return;
    int index = y * image_width + x;
    d_gray_image[index * 2] = (uint8_t)(.299f * d_rgba_image[index * 4] + .587f * d_rgba_image[index * 4 + 1] + .114f * d_rgba_image[index * 4 + 2]);
    d_gray_image[index * 2 + 1] = d_rgba_image[index * 4 + 3];
}

__global__ void rgb_to_grayscale(uint8_t *d_rgba_image, uint8_t *d_gray_image, int image_width, int image_height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y > image_height || x > image_width) return;
    int index = y * image_width + x;
    d_gray_image[index] = (uint8_t)(.299f * d_rgba_image[index * 3] + .587f * d_rgba_image[index * 3 + 1] + .114f * d_rgba_image[index * 3 + 2]);
}

const char *get_file_ext(char *file_path) {
    const char *p, *dot = file_path;
    while (p = strchr(dot, '.')) dot = p + 1;
    if (dot == file_path) return "";
    return dot;
}

int main(int argc, char **argv) {
    char input_file[MAX_PATH + 1], output_file[MAX_PATH + 1];
    const char *input_file_extension;
    if (argc < 2) {
        printf("Usage: ./rgb2gray input_file [output_file]\n");
        exit(1);
    }
    strncpy(input_file, argv[1], MAX_PATH);
    input_file[MAX_PATH] = '\0';
    input_file_extension = get_file_ext(input_file);
    /* if only input_file is passed then default output_file to input_file_grayscale.ext
    input_file without extension + "_grayscale." + input_file's extension */
    if (argc == 2) {
        int l = strnlen(input_file, MAX_PATH) - strnlen(input_file_extension, MAX_PATH) - 1;
        strncpy(output_file, input_file, l);
        output_file[l] = '\0';
        strncat(strncat(output_file, "_grayscale.", MAX_PATH), input_file_extension, MAX_PATH);
    } else {
        strncpy(output_file, argv[2], MAX_PATH);
        output_file[MAX_PATH] = '\0';
    }
    int width, height, channels;
    stbi_info(input_file, &width, &height, &channels);
    if (channels != 4 && channels != 3) {
        printf("Invalid input image '%s' has %d channel%s, expected 3 or 4\n", input_file, channels, channels > 1 ? "s" : "");
        exit(1);
    }
    uint8_t *input_img = stbi_load(input_file, &width, &height, &channels, 0);
    if (!input_img) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image '%s' with a width of %dpx, a height of %dpx and %d channels\n", input_file, width, height, channels);
    size_t img_size = width * height * channels;
    int gray_channels = channels == 4 ? 2 : 1;
    size_t output_img_size = width * height * gray_channels;
    uint8_t *output_img = (uint8_t *)malloc(output_img_size);
    if (!output_img) {
        printf("Unable to allocate memory for the output image\n");
        exit(1);
    }
    uint8_t *d_input_img, *d_output_img;
    cudaEvent_t start, stop;
    float time_spent;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    size_t input_img_size = width * height * channels;
    cudaMalloc((void **)&d_input_img, input_img_size);
    cudaMalloc((void **)&d_output_img, output_img_size);
    cudaMemcpy(d_input_img, input_img, input_img_size, cudaMemcpyHostToDevice);
    const dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    unsigned int nb_blocksx = (unsigned int)(width / BLOCK_WIDTH + 1);
    unsigned int nb_blocksy = (unsigned int)(height / BLOCK_HEIGHT + 1);
    const dim3 grid_size(nb_blocksx, nb_blocksy, 1);
    cudaEventRecord(start, 0);
    if (channels == 4)
        rgba_to_grayscale<<<grid_size, block_size>>>(d_input_img, d_output_img, width, height);
    else
        rgb_to_grayscale<<<grid_size, block_size>>>(d_input_img, d_output_img, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_spent, start, stop);
    cudaMemcpy(output_img, d_output_img, output_img_size, cudaMemcpyDeviceToHost);
    const char *output_file_extension = get_file_ext(output_file);
    if (!(strcmp(output_file_extension, "jpg") || strcmp(output_file_extension, "jpeg") || strcmp(output_file_extension, "JPG") || strcmp(output_file_extension, "JPEG")))
        stbi_write_jpg(output_file, width, height, gray_channels, output_img, 100);
    else if (!(strcmp(output_file_extension, "bmp") || strcmp(output_file_extension, "BMP")))
        stbi_write_bmp(output_file, width, height, gray_channels, output_img);
    else
        stbi_write_png(output_file, width, height, gray_channels, output_img, width * gray_channels);
    stbi_image_free(input_img);
    free(output_img);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input_img);
    cudaFree(d_output_img);
    printf("Check '%s' (took %fms)\n", output_file, time_spent);
    return 0;
}
