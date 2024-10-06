#pragma once


#include <thrust/sort.h>
#include <curand_kernel.h>
#include <memory>

__device__
inline float random_float(curandState_t* state);
// struct hitRecord; 

// inline double random_double();

struct material;

struct hitRecord {
    glm::vec3 p;
    glm::vec3 normal;
    material* mat_ptr;
    float t;
    bool front_face;

    __host__ __device__
    void set_face_normal(const ray& r, const glm::vec3& outward_normal) {
        front_face = glm::dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class AaBb {
    public:
        interval x, y, z;
        __device__ __host__
        AaBb() {} // The default AABB is empty, since intervals are empty by default
        __device__ __host__
        AaBb(const interval& x, const interval& y, const interval& z): x(x), y(y), z(z) {}
        __device__ __host__
        AaBb(const glm::vec3& a, const glm::vec3& b) {
            /* Treat the 2 points a & b as extrema for the bounding box, so we don't require a particular min/max coordinate order */
            x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]); // same as a[0] same as a.x
            y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
            z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
        }
        __device__ __host__
        AaBb(const AaBb& box0, const AaBb& box1) {
            x = interval(box0.x, box1.x);
            y = interval(box0.y, box1.y);
            z = interval(box0.z, box1.z);
        }
        __device__ __host__
        const interval& axis_interval(int n) const {
            if (n == 1) return y;
            if (n == 2) return z;
            return x;
        }
        __device__ __host__
        bool hit(const ray& r, interval ray_t) const {
            const glm::vec3& ray_orig = r.origin;
            const glm::vec3& ray_dir  = r.direction;

            for (int axis = 0; axis < 3; axis++) {
                const interval& ax = axis_interval(axis);
                const double adinv = 1.0 / ray_dir[axis];

                auto t0 = (ax.min - ray_orig[axis]) * adinv;
                auto t1 = (ax.max - ray_orig[axis]) * adinv;

                if (t0 < t1) {
                    if (t0 > ray_t.min) ray_t.min = t0;
                    if (t1 < ray_t.max) ray_t.max = t1;
                } else {
                    if (t1 > ray_t.min) ray_t.min = t1;
                    if (t0 < ray_t.max) ray_t.max = t0;
                }

                if (ray_t.max <= ray_t.min)
                    return false;
            }

            return true;
        }
};

class BVH_Node {
    public:

        // __device__ __host__
        // BVH_Node(hittable_boxes boxes, curandState_t* states, int i): BVH_Node(boxes.list, 0, boxes.list_size, states, i) {} 
        // BVH_Node(std::vector<BVH_Node*>list, curandState_t* states, int i): BVH_Node(list.data(), 0, list.size(), states, i) {} 

        __device__ 
        BVH_Node(BVH_Node** nodeObjects, size_t start, size_t end, curandState_t* states, int i) {
            auto x = states[i];
            int axis = int(3 * random_float(&x));  // randomly choose an axis
            // int axis = int(3 * random_double());  // randomly choose an axis
            states[i] = x;  // save the value back after use
            auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;
            
            
            size_t object_span = end - start;

            if (object_span == 1) {
                left = right = nodeObjects[start];
            } else if (object_span == 2) {
                left = nodeObjects[start];
                right = nodeObjects[start + 1];
            } else { 
                thrust::sort(thrust::device, nodeObjects + start, nodeObjects + end, comparator );
                auto mid = start + object_span / 2;
                left = new BVH_Node(nodeObjects, start, mid, states, i);
                right = new BVH_Node(nodeObjects, mid, end, states, i);
            }
            
            bBox = AaBb(left->bounding_box(), right->bounding_box());

        }

        __host__ __device__
        bool hit(const ray& r, interval ray_t, hitRecord &rec) const  {
            if(!bBox.hit(r, ray_t))
                return false;
            bool hit_left   = left->hit(r, ray_t, rec);
            bool hit_right  = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

            return hit_left || hit_right;
        }

        __device__ __host__
        AaBb bounding_box() const {return bBox; }
    private:
        AaBb bBox;
        BVH_Node* left;
        BVH_Node* right;

        static bool box_compare(const BVH_Node* a, const BVH_Node* b, int axis_index){
            auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
            auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
            

            return a_axis_interval.min < b_axis_interval.min;
        }

        static bool box_x_compare (const BVH_Node* a, const BVH_Node* b){
            return box_compare(a, b, 0);
        }
        static bool box_y_compare (const BVH_Node* a, const BVH_Node* b){
            return box_compare(a, b, 1);
        }
        static bool box_z_compare (const BVH_Node* a, const BVH_Node* b){
            return box_compare(a, b, 2);
        }

         

};

struct BVHNode {
    AaBb bbox;
    int left_child_index; // Index left child in the array, -1 if it's a leaf
    int right_child_index; // index right child in the array, -1 if it's a leaf
    int object_index;       // index of the object the leaf represent (if it's a leaf)
    bool is_leaf;


};

// class BVH {
//     public:
//         BVH(const std::vector<shared_ptr<hittable_boxes>>& objects) {
//             build_bh
//         }
// }