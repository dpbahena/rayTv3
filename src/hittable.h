#pragma once


#include "material.h"


class hit_record {
    public:
        glm::vec3 p;
        glm::vec3 normal;
        material* mat;
        double t;
        Type type;
        bool front_face;
        __device__
        void set_face_normal(const ray& r, const glm::vec3& outward_normal){
            /* Sets the hit record normal vector. NOTE: the parameter outward_normal is assumed to be of unit length */
            front_face = glm::dot(r.direction, outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

// Define a struct for sphere data
struct sphere_data {
    ray center;
    float radius;
    material* mat;
};

struct hittable {

    Type type;

    // Use the define struct in the union
    union {
        sphere_data sphere;
    };

    // default constructor
    hittable() : type(Type::NONE) {}

    // Constructor for each type
    static hittable make_sphere(const glm::vec3& static_center, float radius, material* mat) {
        hittable obj;
        obj.type = Type::SPHERE;
        obj.sphere.center = ray(static_center, glm::vec3(0.0f, 0.0f, 0.0f));
        obj.sphere.radius = radius;
        obj.sphere.mat = mat;
        return obj;
    }

    static hittable make_sphere(const glm::vec3& center1, const glm::vec3& center2, float radius, material* mat) {
        hittable obj;
        obj.type = Type::SPHERE;
        obj.sphere.center = ray(center1, center2 - center1);
        obj.sphere.radius = radius;
        obj.sphere.mat = mat;
        return obj;
    }


};
     
