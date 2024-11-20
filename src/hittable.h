#pragma once


#include "material.h"
#include "texture.h"
#include "aabb.h"
#include <vector>

#include <curand_kernel.h>


struct BVHNode;
struct material;

class hit_record {
    public:
        glm::vec3 p;
        glm::vec3 normal;
        material* mat;
        double t;
        float u;
        float v;
        Type type;
        bool front_face;
        __device__ __host__
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
    AaBb bbox;
    __device__ __host__
    AaBb bounding_box() const {return bbox;}
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    __device__ __host__
    void get_sphere_uv(const glm::vec3& p, float& u, float& v) const;
};

//* Quadrilateral struct
struct quad_data {
    glm::vec3 Q;
    glm::vec3 u, v, w;
    glm::vec3 normal;
    float D;
    
    material* mat;
    AaBb bbox;
    
    __device__ __host__
    AaBb bounding_box() const {return bbox;}
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    __device__ __host__
    bool is_interior(float a, float b, hit_record& rec) const;
    __device__ __host__
    void set_boundig_box();
    
};

struct hittable;  // predeclare the existance of a hittable struct
// Define a struct of a list of hittables
struct hittableList_data {
    hittable* objects;
    BVHNode* nodeObjects;
    int objects_size;
    AaBb bbox;
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec, curandState_t* state,  int i, int j) const;
    __device__ __host__
    void setList(hittable* hittables, size_t list_size);
    
    void setNodes(BVHNode* nodes, hittable* hittables);
    __device__ __host__
    AaBb bounding_box() const {return bbox;}
};


//* translate struct
struct translate_data {
    hittable* object;
    glm::vec3 offset;
    AaBb bbox;
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    __device__ __host__
    AaBb bounding_box() const {return bbox;}
};

//* Rotate struc
struct rotateY_data {
    hittable* object;
    float sin_theta;
    float cos_theta;
    AaBb bbox;
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    __device__ __host__
    AaBb bounding_box() const {return bbox;}

};

//* Constant_Medium struct
struct constantMedium_data {
    hittable* boundary;
    float neg_inv_density;
    material* phase_function;
    

    __device__
    bool hit(const ray& r, interval ray_t, hit_record& rec, curandState_t* states,  int i, int j);
    __device__ __host__
    AaBb bounding_box() const;

};



struct hittable {

    Type type;

    // Use the define struct in the union
    union {
        sphere_data sphere;
        quad_data quad;
        hittableList_data hittableList;
        translate_data translate;
        rotateY_data rotateY;
        constantMedium_data constantMedium;
    };

    // default constructor
    __device__ __host__
    hittable() : type(Type::NONE) {}

    // Constructor for each type

    /* STATIONARY SPHERE */
    __device__ __host__
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
    __device__ __host__
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

    //* Quad Constructor
    static hittable make_quad(const glm::vec3& Q, const glm::vec3& u, const glm::vec3& v, material* mat){
        hittable obj;
        obj.type = Type::QUAD;
        obj.quad.Q = Q;
        obj.quad.u = u;
        obj.quad.v = v;
        obj.quad.mat = mat;
        auto n = glm::cross(u, v);
        obj.quad.normal = glm::normalize(n);
        obj.quad.D = glm::dot(obj.quad.normal, obj.quad.Q);
        obj.quad.w = n / glm::dot(n, n);
        obj.quad.set_boundig_box();

        return obj;
    }

    //* hittable list constructor
    static hittable make_hittableList() {
        hittable obj;
        obj.type = Type::LIST;
         
        return obj;
    }

    //* Translate object constructor
    static hittable make_translate(hittable* object, const glm::vec3& offset) {
        hittable obj;
        obj.type = Type::TRANSLATE;
        
        obj.translate.offset = offset;
        switch (object->type){
            case Type::SPHERE:
                obj.translate.bbox = object->sphere.bounding_box() + offset;
                break;
            case Type::QUAD:
                obj.translate.bbox = object->quad.bounding_box() + offset;
                break;
            case Type::ROTATE_Y:
                obj.translate.bbox = object->rotateY.bounding_box() + offset;
                break;
            default:
                break;
        }

        obj.translate.object = object;

        return obj;

    }

    //* RotateY object constructor
    static hittable make_rotateY(hittable* object, float angle) {
        hittable obj;
        obj.type = Type::ROTATE_Y;
        auto radians = glm::radians(angle);
        obj.rotateY.sin_theta = sin(radians);
        obj.rotateY.cos_theta = cos(radians);
        
        switch (object->type){
            case Type::SPHERE:
                obj.rotateY.bbox = object->sphere.bounding_box();
                break;
            case Type::QUAD:
                obj.rotateY.bbox = object->quad.bounding_box();
                break;
            case Type::TRANSLATE:
                obj.rotateY.bbox = object->translate.bounding_box();
                break;
            default:
                break;
        }
        obj.rotateY.object = object;

        glm::vec3 min(MAXFLOAT, MAXFLOAT, MAXFLOAT);
        glm::vec3 max(-MAXFLOAT, -MAXFLOAT, -MAXFLOAT);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i * obj.rotateY.bbox.x.max + (1 - i) * obj.rotateY.bbox.x.min;
                    auto y = j * obj.rotateY.bbox.y.max + (1 - j) * obj.rotateY.bbox.y.min;
                    auto z = k * obj.rotateY.bbox.z.max + (1 - k) * obj.rotateY.bbox.z.min;

                    auto newx =  obj.rotateY.cos_theta * x + obj.rotateY.sin_theta * z;
                    auto newz = -obj.rotateY.sin_theta * x + obj.rotateY.cos_theta * z;

                    glm::vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = fminf(min[c], tester[c]);
                        max[c] = fmaxf(max[c], tester[c]);
                    }

                }
            }
        }
        
        obj.rotateY.bbox = AaBb(min, max);
        
        return obj;

    }


    static hittable make_constantMedium(hittable* boundary, float density, const glm::vec3& albedo) {
        hittable obj;
        obj.type = Type::MEDIUM;
        obj.constantMedium.boundary = boundary;
        obj.constantMedium.neg_inv_density = -1.0f / density;
        obj.constantMedium.phase_function = new material(material::isotropic_material(albedo));

        return obj;
    }

    static hittable make_constantMedium(hittable* boundary, float density, texture* tex) {
        hittable obj;
        obj.type = Type::MEDIUM;
        obj.constantMedium.boundary = boundary;
        obj.constantMedium.neg_inv_density = -1.0f / density;
        obj.constantMedium.phase_function = new material(material::isotropic_material(tex));

        return obj;
    }

    

    

};
     
