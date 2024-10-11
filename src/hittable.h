#pragma once


#include "material.h"
#include "aabb.h"
#include <vector>


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

/* Stores a bounding box, the indices for the left and right children, and whether the node is a leaf */
struct BVHNode {
    AaBb bbox;
    int left_child_index; // Index left child in the array, -1 if it's a leaf
    int right_child_index; // index right child in the array, -1 if it's a leaf
    int object_index;       // index of the object the leaf represent (if it's a leaf)
    bool is_leaf;           // is this node a leaf?


};

// Define a struct for sphere data
struct sphere_data {
    ray center;
    float radius;
    material* mat;
    AaBb bbox;
};

// Define a struct for bvh_node
struct bvh_data {

    BVHNode node;    
    
};

struct hittable {

    Type type;

    // Use the define struct in the union
    union {
        sphere_data sphere;
        bvh_data BVH;

    };

    // default constructor
    hittable() : type(Type::NONE) {}

    // Constructor for each type

    /* STATIONARY SPHERE */
    static hittable make_sphere(const glm::vec3& static_center, float radius, material* mat) {
        hittable obj;
        obj.type = Type::SPHERE;
        obj.sphere.center = ray(static_center, glm::vec3(0.0f, 0.0f, 0.0f));
        obj.sphere.radius = radius;
        obj.sphere.mat = mat;
        auto rvec = glm::vec3(radius, radius, radius);
        obj.sphere.bbox = AaBb(static_center - rvec, static_center + rvec);
        
        return obj;
    }
    /* MOVING SPHERE */
    static hittable make_sphere(const glm::vec3& center1, const glm::vec3& center2, float radius, material* mat) {
        hittable obj;
        obj.type = Type::SPHERE;
        obj.sphere.center = ray(center1, center2 - center1);
        obj.sphere.radius = radius;
        obj.sphere.mat = mat;
        auto rvec = glm::vec3(radius, radius, radius);
        AaBb box1(obj.sphere.center.at(0) - rvec, obj.sphere.center.at(0) + rvec);
        AaBb box2(obj.sphere.center.at(1) - rvec, obj.sphere.center.at(1) + rvec);
        obj.sphere.bbox = AaBb(box1, box2);

        return obj;
    }


};
     
