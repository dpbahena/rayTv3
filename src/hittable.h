#pragma once

#include "interval.h"
#include "material.h"
#include "texture.h"
#include "aabb.h"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <rtw_stb_image.h>

typedef struct  {
        uint32_t  width;
        uint32_t height;
} Extent2D;

struct Vertex0 {
    glm::vec3 point;
};
struct Face {
    uint32_t indices[3];
    glm::vec3 color;
};

struct Builder0 {
  std::vector<Vertex0> vertices;
  std::vector<Face> indices;
};


struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec3 normal{};
    glm::vec2 uv{};     // short of 2-Dimensional texture coordinates
    size_t material_id;

    bool operator==(const Vertex &other) const {
        return position == other.position && color == other.color && normal == other.normal && uv == other.uv && material_id == other.material_id;
    }
};

// struct Texture {
//     std::vector<uint32_t> data;
//     int width;
//     int height;

// };

struct Texture {
    unsigned char* bdata{};
    int width{};
    int height{};
    int scanline{};
    int pixel_size{};
};

struct Material {
    std::string name;
    glm::vec3 diffuseColor;
    // std::string diffuseTexName;   // map_Kd
    // std::string roughnessTexName; // map_Ns (used for roughness in your case)
    // std::string normalTexName;    // map_Bump
    const char* diffuseTexName;   // map_Kd
    const char* roughnessTexName; // map_Ns (used for roughness in your case)
    const char* normalTexName;    // map_Bump
    
    
    // Add handles or IDs for loaded textures if needed
    Texture diffuseTexture;
    Texture roughnessTexture;
    Texture normalTexture;
    // unsigned char* r_diffuseTexture;
    // unsigned char* r_roughnessTexture;
    // unsigned char* r_normalTexture;

    // rtw_image r_diffuseTexture;
    // rtw_image r_roughnessTexture;
    // rtw_image r_normalTexture;
};


class Builder {
    public:
        std::vector<Vertex> vertices{};
        std::vector<uint32_t> indices{};
        std::vector<uint32_t> textureBuffer;
        std::vector<Material> materialList; // list of Materials
        glm::uvec2 extent;
        // Extent2D extent;

        void loadModel(const std::string &filepath);
        Texture loadTexture(const std::string &filepath);

    private:
        std::string resolvePath(const std::string& baseDir, const std::string& relativePath);
    
};

// from https:://stackoverflow.com/a/57595105
    template <typename T, typename... Rest>
    void hashCombine(std::size_t& seed, const T& v, const Rest&... rest) {
        seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        (hashCombine(seed, rest), ...);
    };


struct alignas(16) BVHNode {
    AaBb bbox;
    int left_child_index;     // Index of left child in the BVH array (-1 if it's a leaf)
    int right_child_index;    // Index of right child in the BVH array (-1 if it's a leaf)
    int rope_index;
    bool is_leaf;             // Is this node a leaf?
    size_t start;
    size_t end;
    int object_index;         // Index of the object (used if it's a leaf)
    
};


struct alignas(16) StackNode {
            size_t start, end;
            int nodeIndex; // indext of the node being processed
            int parentIndex;
            bool isLeftChild;           
            int ropeIndex;
};



class hit_record {
    public:
        glm::vec3 p;
        glm::vec3 normal;
        material* mat;
        double t;
        float u;
        float v;
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

//* Triangle struct
struct triangle_data {
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
    int objects_size;
    AaBb bbox;
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec, float randNumber) const;
    bool ah();
    __device__ __host__
    AaBb bounding_box() const {return bbox;}
};

struct bvhNode_data {
    hittable* objects;
    BVHNode* nodes;
    size_t objects_size;
    AaBb bbox;

    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec, float randNumber) const;
    __device__ __host__
    AaBb bounding_box() const {return bbox;}
    __device__ __host__
    void build_bvh();
};


//* translate struct
struct translate_data {
    hittable* object;
    glm::vec3 offset;
    AaBb bbox;
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec, float randNumber) const;
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
    bool hit(const ray& r, interval ray_t, hit_record& rec, float randNumber) const;
    __device__ __host__
    AaBb bounding_box() const {return bbox;}
    void calculateBbox();

};

//* Constant_Medium struct
struct constantMedium_data {
    hittable* boundary;
    float neg_inv_density;
    material* phase_function;
    

    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec, float randNumber) const;
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
        bvhNode_data bvhNode;
        triangle_data triangle;
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

    //* Triangle Constructor
    static hittable make_triangle(const glm::vec3& Q, const glm::vec3& u, const glm::vec3& v, material* mat){
        hittable obj;
        obj.type = Type::TRI;
        obj.triangle.Q = Q;
        obj.triangle.u = u;
        obj.triangle.v = v;
        obj.triangle.mat = mat;
        auto n = glm::cross(u, v);
        obj.triangle.normal = glm::normalize(n);
        obj.triangle.D = glm::dot(obj.triangle.normal, obj.triangle.Q);
        obj.triangle.w = n / glm::dot(n, n);
        obj.triangle.set_boundig_box();

        return obj;
    }


    //* hittable list constructor
    static hittable make_hittableList() {
        hittable obj;
        obj.type = Type::LIST;
         
        return obj;
    }

    //* hittable list constructor
    static hittable make_hittableList(hittable* objects, size_t objects_size) {
        hittable obj;
        obj.type = Type::LIST;
        obj.hittableList.objects = objects;
        obj.hittableList.objects_size = objects_size;
         
        return obj;
    }

    
    
    static hittable make_bvhNode(BVHNode* nodes, hittable* objects, size_t objects_size){
        hittable obj;
        obj.type = Type::BVH;
        obj.bvhNode.nodes = nodes;
        obj.bvhNode.objects = objects;
        obj.bvhNode.objects_size = objects_size;
        
        obj.bvhNode.build_bvh();

        return obj;
    }

    //* Translate object constructor
    static hittable make_translate(hittable* object, const glm::vec3& offset) {
        hittable obj;
        obj.type = Type::TRANSLATE;
        obj.translate.object = object;
        
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
            case Type::LIST:
                obj.translate.bbox = object->hittableList.bounding_box() + offset;
                break;
            case Type::MEDIUM:
                obj.translate.bbox = object->constantMedium.bounding_box() + offset;
                break;
            default:
                break;
        }
        

        return obj;
    }

    //* RotateY object constructor
    static hittable make_rotateY(hittable* object, float angle) {
        hittable obj;
        obj.type = Type::ROTATE_Y;
        auto radians = glm::radians(angle);
        obj.rotateY.sin_theta = sin(radians);
        obj.rotateY.cos_theta = cos(radians);
        obj.rotateY.object = object;
        switch (object->type){
            case Type::SPHERE:
                obj.rotateY.bbox = object->sphere.bounding_box();
                obj.rotateY.calculateBbox();
                break;
            case Type::QUAD:
                obj.rotateY.bbox = object->quad.bounding_box();
                obj.rotateY.calculateBbox();
                break;
            case Type::TRANSLATE:
                obj.rotateY.bbox = object->translate.bounding_box();
                obj.rotateY.calculateBbox();
                break;
            case Type::LIST:
                obj.rotateY.bbox = object->hittableList.bounding_box();
                obj.rotateY.calculateBbox();
                break;
            case Type::MEDIUM:
                obj.rotateY.bbox = object->constantMedium.bounding_box();
                obj.rotateY.calculateBbox();
                break;
            default:
                break;
        }
        
        return obj;
    }


    static hittable make_constantMedium(hittable* boundary, float density, material* mat) {
        hittable obj;
        obj.type = Type::MEDIUM;
        obj.constantMedium.boundary = boundary;
        obj.constantMedium.neg_inv_density = -1.0f / density;
        obj.constantMedium.phase_function = mat;

        return obj;
    }
  
};


     
