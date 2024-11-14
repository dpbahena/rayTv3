#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
// #include <cstdlib>
// #include <iostream>

class rtw_image {
    public:
        __device__ __host__ 
        rtw_image(){}
        __device__ __host__ 
        rtw_image(const char* image_filename){
            // // auto filename = std::string(image_filename);
            // // auto imagedir = getenv("RTW_IMAGES");
            // Hunt for the image file in some likely locations.
            // // if(imagedir && load(std::string(imagedir) + "/" +image_filename)) return;
            if(load(image_filename)) return;
            if(load("images/")) return;
            if(load("../images/")) return;
            printf("ERROR: Could not load image file '%s'\n",image_filename );

        }
        __device__ __host__ 
        ~rtw_image() {
            delete[] bdata;
            STBI_FREE(fdata);
        }

        __device__ __host__ 
        bool load(const char* filepath) {
            /**
             * *Loads the linear (gamma=1) image data.
             * @return true if load succeeded.
             * * Resulting data buffer contains the three floating point values (0.0 - 1.0) for the first
             * * pixel (red, green, blue). 
             * * Pixels are contiguous, going from left to right for the width then the next row
             * * for the full height of the image
             */

            auto n = bytes_per_pixel; // dummy parameter: original components per pixel
            fdata = stbi_loadf(filepath, &image_width, &image_height, &n, bytes_per_pixel);
            if (fdata == nullptr) return false;

            bytes_per_scanline = image_width * bytes_per_pixel;
            convert_to_bytes();
            return true;

        }

        __device__ __host__ int width() const {return (fdata == nullptr) ? 0 : image_width; }
        __device__ __host__ int height() const {return (fdata == nullptr) ? 0 : image_height; }

        /**
         * @return the address of the three RGB bytes of the pixel at x, y.
         * @return magenta if there is no image data
         */
        __device__ __host__
        const unsigned char* pixel_data(int x, int y) const {
            static unsigned char magenta[] = {255, 0, 255};
            if (bdata == nullptr) {
                return magenta;
            }
            x = clamp(x, 0, image_width);
            y = clamp(y, 0, image_height);

            return bdata + y * bytes_per_scanline + x * bytes_per_pixel;
        }

    private:
        const int       bytes_per_pixel = 3;
        float*          fdata = nullptr;           // linear floating point pixel data
        unsigned char*  bdata = nullptr;           // linear 8-bit pixel data
        int             image_width = 0;
        int             image_height = 0;
        int             bytes_per_scanline = 0;

        /**
         * @return the value clamped to the range [low, high]
         */
        __device__ __host__ 
        static int clamp(int x, int low, int high) {
            if (x < low) return low;
            if (x < high) return x;
            return high - 1;
        }
        __device__ __host__ 
        static unsigned char float_to_byte(float value) {
            if (value <= 0.0) return 0;
            if (1.0 <= value) return 255;
            return static_cast<unsigned char>(256.0 * value);
        }

        /**
         * * Convert the linear floating point pixel data to bytes
         * * storing the resulting byte data in the 'bdata' member;
         */
        __device__ __host__ 
        void convert_to_bytes() {
            int total_bytes = image_width * image_height * bytes_per_pixel;
            bdata = new unsigned char[total_bytes];

            /** iterate through all pixel components, converting from [0.0, 1.0] flaot values to 
            ** unsigned [0, 255] bytes values
            */
            auto *bptr = bdata;
            auto *fptr = fdata;
            for (auto i = 0; i < total_bytes; i++, fptr++, bptr++){
                *bptr = float_to_byte(*fptr);
            }
        }

};