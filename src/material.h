#pragma once
// #include "types.h"
#include "texture.h"

// enum class Type {NONE, SPHERE, BBOX, LAMBERTIAN, METAL, DIELECTRIC};


// Define a struct for lambertian material
struct lambertian_data {
    glm::vec3 albedo;
    texture* tex;
};

struct metal_data {
    glm::vec3 albedo;
    float fuzz;
};

struct dielectric_data {
    /*  Refractive index (ri) in vacuum or air, or the ratio of the material's ri over the ri of the enclosing media */
    double refraction_index;
};

struct diffuseLight_data {
    texture* tex;
    
};

struct isotropic_data {
    
    texture* tex;
};



struct material {
    Type type;
    
    union {
        lambertian_data lambertian;
        metal_data metal;
        dielectric_data dielectric;
        diffuseLight_data diffuseLight;
        isotropic_data isotropic;
        
    };


    /* Default constructor */
    material(): type(Type::NONE) {}

    // ~material() {
    //     // Clean up dynamically allocated memory based on type
    //     if (type == Type::LAMBERTIAN && lambertian.tex != nullptr){
    //         delete lambertian.tex;
    //         lambertian.tex = nullptr;   // set to nullptr to avoid dangling pointer
    //     }
    //     if (type == Type::DIFFUSE && diffuseLight.tex != nullptr){
    //         delete diffuseLight.tex;
    //         diffuseLight.tex = nullptr;   // set to nullptr to avoid dangling pointer
    //     } 

    // }

    // constructor for type lambertian
    static material lambertian_material(const glm::vec3& albedo) {
        material obj;
        obj.type = Type::LAMBERTIAN;
        obj.lambertian.albedo = albedo;
        obj.lambertian.tex = new texture(texture::solid_texture(albedo));
        return obj;
    }

    static material lambertian_material(texture* tex) {
        material obj;
        obj.type = Type::LAMBERTIAN;
        obj.lambertian.tex = tex;
        return obj;
    }
    // constructor for type metal
    static material metal_material(const glm::vec3& albedo, double fuzz) {
        material obj;
        obj.type = Type::METAL;
        obj.metal.albedo = albedo;
        obj.metal.fuzz = fuzz < 1 ? fuzz : 1.0;
        return obj;
    }
    // constructor for type dielectric
    static material dielectric_material(double refraction_index) {
        material obj;
        obj.type = Type::DIELECTRIC;
        obj.dielectric.refraction_index = refraction_index;
        return obj;
    }

    // constructor for diffuseLight
    static material diffuseLight_material(texture* tex) {
        material obj;
        obj.type = Type::DIFFUSE;
        obj.diffuseLight.tex = tex;

        return obj;
    }

    static material diffuseLight_material(const glm::vec3& emit) {
        material obj;
        obj.type = Type::DIFFUSE;
        obj.diffuseLight.tex = new texture(texture::solid_texture(emit));

        return obj;
    }

    static material isotropic_material(const glm::vec3& albedo) {
        material obj;
        obj.type = Type::ISOTROPIC;
        obj.isotropic.tex = new texture(texture::solid_texture(albedo));

        return obj;
    }

    static material isotropic_material(texture* tex) {
        material obj;
        obj.type = Type::ISOTROPIC;
        obj.isotropic.tex = tex;

        return obj;
    }



};

