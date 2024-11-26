#pragma once
#include "types.h"
#include "rtw_stb_image.h"
#include "perlin.h"

struct texture;

struct checkerTexture_data {
    float inv_scale;
    texture* even;
    texture* odd;
    __device__ __host__
    glm::vec3 value(float u, float v, const glm::vec3& p) const;
};



struct solidColor_data {
    glm::vec3 albedo;
    __device__ __host__
    glm::vec3 value(float u, float v, const glm::vec3& p) const { return albedo;};
};

struct imageTexture_data {
    unsigned char* bdata;
    int image_width;
    int image_height;
    int bytes_per_scanline;
    int bytes_per_pixel;
    __device__ __host__
    glm::vec3 value(float u, float v, const glm::vec3& p) const;
    __device__ __host__
    const unsigned char* pixel_data(int x, int y) const;
    __device__ __host__
    int clamp(int x, int low, int high) const;
};

struct noiseTexture_data {
    Perlin noisy;
    float scale;
    glm::vec3 albedo;
    __device__ __host__
    glm::vec3 value(glm::vec3 albedo, float u, float v, const glm::vec3& p);

};


__device__ __host__
static glm::vec3 checkeredTexture_value(float u, float v, const glm::vec3& p, checkerTexture_data& checkered);


struct texture {
    Type type;

    union {
        solidColor_data solidColor;
        checkerTexture_data checkerTexture;
        imageTexture_data imageTexture;
        noiseTexture_data noiseTexture;
    };

    /* Default Constructor */
    texture() : type(Type::NONE) {}


    // *Constructors for type SOLID
    static texture solid_texture(const glm::vec3& albedo){
        texture obj;
        obj.type = Type::SOLID;
        obj.solidColor.albedo = albedo;
        return obj;
    }

    static texture solid_texture(float red, float green, float blue) {
        texture obj;
        obj.type = Type::SOLID;
        obj.solidColor.albedo = glm::vec3(red, green, blue);
        return obj;
    }

    // * Constructor for type CHECKER
    static texture checker_texture(float scale, texture* even, texture* odd) {
        texture obj;
        obj.type = Type::CHECKER;
        obj.checkerTexture.inv_scale = (1.0f / scale);
        obj.checkerTexture.even = even;
        obj.checkerTexture.odd = odd;
        return obj;
        
    }

    // static texture checker_texture(float scale, const glm::vec3& color1, const glm::vec3& color2) {
    //     texture obj;
    //     obj.type = Type::CHECKER;
    //     obj.checkerTexture.inv_scale = (1.0f / scale);
    //     obj.checkerTexture.even =  new texture(solid_texture(color1));
    //     obj.checkerTexture.odd =  new texture(solid_texture(color2));
        
    //     return obj;
        
    // }
    
    // *Constructors for type IMAGE
    static texture image_texture(unsigned char* bdata, int width, int height, int vScan, int bytes_per_pixel) {
        texture obj;
        obj.type = Type::IMAGE;
        obj.imageTexture.bdata = bdata;
        obj.imageTexture.image_width = width;
        obj.imageTexture.image_height = height;
        obj.imageTexture.bytes_per_scanline = vScan;
        obj.imageTexture.bytes_per_pixel = bytes_per_pixel;
        

        return obj; 
    }

    // *Constructors for type NOISE
    static texture noise_texture(Perlin &noisy, glm::vec3 albedo, float scale = 1.0f) {
        texture obj;
        obj.type = Type::NOISE;
        obj.noiseTexture.albedo = albedo;
        obj.noiseTexture.scale = scale;
        obj.noiseTexture.noisy = noisy;
        


        return obj;
    }

   

    //* Value function to dispatch based on texture type
    __device__ __host__
    glm::vec3 value (float u, float v, const glm::vec3& p) {
        switch(type) {
            case Type::SOLID:
                return solidColor.value(u, v, p);
            case Type::CHECKER:
                return checkeredTexture_value(u, v, p, checkerTexture);
                // return checkerTexture.value(u, v, p);
            case Type::IMAGE:
                return imageTexture.value(u, v, p);
            case Type::NOISE:
                // auto color = noiseTexture.albedo;
                return noiseTexture.value(noiseTexture.albedo, u, v, p);
            default: 
                return glm::vec3(0.0f);   // default value if none
            
        }
    }


};





