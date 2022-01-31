#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define MAX_PATH 255

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
    } else
        strncpy(output_file, argv[2], MAX_PATH);
    int width, height, channels;
    uint8_t *input_img = stbi_load(input_file, &width, &height, &channels, 0);
    if (!input_img) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image '%s' with a width of %dpx, a height of %dpx and %d channels\n", input_file, width, height, channels);
    size_t img_size = width * height * channels;
    int gray_channels = channels == 4 ? 2 : 1;
    size_t output_img_size = width * height * gray_channels;
    uint8_t *output_img = malloc(output_img_size);
    if (!output_img) {
        printf("Unable to allocate memory for the output image\n");
        exit(1);
    }
    clock_t begin = clock();
    for (uint8_t *p = input_img, *pg = output_img; p != input_img + img_size; p += channels, pg += gray_channels) {
        *pg = (uint8_t)(.299f * *p + .587f * *(p + 1) + .114f * *(p + 2));
        if (channels == 4 && gray_channels == 2)
            *(pg + 1) = *(p + 3);
    }
    clock_t end = clock();
    // time is milliseconds
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
    const char *output_file_extension = get_file_ext(output_file);
    if (!(strcmp(output_file_extension, "jpg") || strcmp(output_file_extension, "jpeg") || strcmp(output_file_extension, "JPG") || strcmp(output_file_extension, "JPEG")))
        stbi_write_jpg(output_file, width, height, gray_channels, output_img, 100);
    else if (!(strcmp(output_file_extension, "bmp") || strcmp(output_file_extension, "BMP")))
        stbi_write_bmp(output_file, width, height, gray_channels, output_img);
    else
        stbi_write_png(output_file, width, height, gray_channels, output_img, width * gray_channels);
    stbi_image_free(input_img);
    stbi_image_free(output_img);
    printf("check '%s' (took %fms)\n", output_file, time_spent);
    return 0;
}
