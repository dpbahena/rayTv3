#pragma once

#include "sphere.h"
#include "aabb.h"
#include <thrust/sort.h>
#include <curand_kernel.h>

const int MAX_STACK_SIZE = 12;

__device__ int random_int_device(curandState_t* state, int min, int max);

struct box_compare {
    int axis;

    __host__ __device__
    box_compare(int axis_index) : axis(axis_index) {}

    __host__ __device__
    bool operator()(const hittable& a, const hittable& b) const {
        auto a_axis_interval = a.sphere.bounding_box().axis_interval(axis);
        auto b_axis_interval = b.sphere.bounding_box().axis_interval(axis);
        return a_axis_interval.min < b_axis_interval.min;
    }
};

struct BVHNodeSoA {
    int* left_child_index = nullptr;     // Index of left child in the BVH array (-1 if it's a leaf)
    int* right_child_index = nullptr;    // Index of right child in the BVH array (-1 if it's a leaf)
    int* object_index = nullptr;         // Index of the object (used if it's a leaf)
    bool* is_leaf = nullptr;             // Is this node a leaf?
    AaBb* bbox = nullptr;
};

struct StackNode {
            size_t start{}, end{};
            int parentIndex{};
            bool isLeftChild{};
};

__device__ static bool box_compare(const hittable& a, const hittable& b, int axis_index);
__device__ static bool box_x_compare (const hittable& a, const hittable& b);
__device__ static bool box_y_compare (const hittable& a, const hittable& b);
__device__ static bool box_z_compare (const hittable& a, const hittable& b);

__global__ void build_bvh_NR(BVHNodeSoA* nodes, hittable* hittables, size_t N) {
    
    int index = 0;
    const int MAX = 15;
    StackNode traversalStack[MAX];
    int top = -1 ;  // initialize stack
    
    traversalStack[++top] ={0, N, -1, true};  // push()

    while (top >= 0 ) {
        
        StackNode current = traversalStack[top--];  // equals top then pop()

        size_t object_span = current.end - current.start;
    

        // **Compute the bounding box of the current node upfront**
        AaBb bbox = AaBb::empty(); 
        for (size_t i = current.start; i < current.end; ++i) {
            bbox = AaBb(bbox, hittables[i].sphere.bounding_box());
        }

        if (object_span == 1) {
            // **Create a leaf node**
            // BVHNode node;
            nodes->is_leaf[index] = true;
            nodes->object_index[index] = current.start;
            nodes->left_child_index[index] = -1;
            nodes->right_child_index[index] = -1;
            nodes->bbox[index] = bbox;  // Bounding box of the single object
            int node_index = index++;

            // **Update parent's child index**
            if (current.parentIndex != -1) {

                auto& parent_node_left_child = nodes->left_child_index[current.parentIndex];
                auto& parent_node_right_child = nodes->right_child_index[current.parentIndex];

                if (current.isLeftChild) {
                    parent_node_left_child = node_index;
                } else {
                    parent_node_right_child = node_index;
                }
            }

        } else if (object_span == 2) {
            // **Create leaf nodes for both objects**
            // Left leaf node
            nodes->is_leaf[index] = true;
            nodes->object_index[index] = current.start;
            nodes->left_child_index[index] = -1;
            nodes->right_child_index[index] = -1;
            nodes->bbox[index] = hittables[nodes->object_index[index]].sphere.bounding_box();  // comment later for "optimization 3.10"
            int left_node_index = index++; 

            // Right leaf node
            nodes->is_leaf[index] = true;
            nodes->object_index[index] = current.start + 1;
            nodes->left_child_index[index] = -1;
            nodes->right_child_index[index] = -1;
            nodes->bbox[index] = hittables[nodes->object_index[index]].sphere.bounding_box();  // comment later for "optimization 3.10"
            int right_node_index = index++; 

            // bbox = AaBb(nodes->bbox[left_node_index], nodes->bbox[right_node_index]);

            // **Create parent node with bounding box**
          
            nodes->is_leaf[index] = false;
            nodes->left_child_index[index] = left_node_index;
            nodes->right_child_index[index] = right_node_index;
            nodes->bbox[index] = bbox;  // Bounding box of the two objects
            int parent_index = index++; 

            // **Update parent's child index**
            if (current.parentIndex != -1) {

                auto& grandparent_node_left_child = nodes->left_child_index[current.parentIndex];
                auto& grandparent_node_righ_child = nodes->right_child_index[current.parentIndex];
                
                if (current.isLeftChild) {
                    grandparent_node_left_child = parent_index;
                } else {
                    grandparent_node_righ_child = parent_index;
                }
            }

        } else {
            // **object_span > 2**
            // **Select the longest axis based on the bounding box**
            int axis = bbox.longest_axis();  // Implement this function in your aabb class

            // **Sort objects along the selected axis**
            auto comparator = (axis == 0) ? box_x_compare
                            : (axis == 1) ? box_y_compare
                                            : box_z_compare;

            thrust::sort(thrust::device, hittables + current.start, hittables + current.end, comparator);

            // **Split the objects into two halves**
            size_t mid = current.start + object_span / 2;

            // **Create an internal node**
            nodes->is_leaf[index] = false;
            nodes->left_child_index[index] = -1;  // Will be set after processing children
            nodes->right_child_index[index] = -1; // Will be set after processing children
            nodes->bbox[index] = bbox;  // Bounding box of all objects in the span
            int node_index = index++;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                auto& parent_node_left_child = nodes->left_child_index[current.parentIndex];
                auto& parent_node_right_child = nodes->right_child_index[current.parentIndex];

                if (current.isLeftChild) {
                    parent_node_left_child = node_index;
                } else {
                    parent_node_right_child = node_index;
                }
            }
            if (top + 2 >= MAX) {
                printf("Error: Stack overflow in the build_bvh_NR\n");
                return;
            }

            // **Push child nodes onto the stack for further processing**
            
            traversalStack[++top] = {mid, current.end, node_index, false};        // Right child
            traversalStack[++top] = {current.start, mid, node_index, true};       // Left child
        }
    }
    
}


__device__
static bool box_compare(const hittable& a, const hittable& b, int axis_index) {
    
    auto a_axis_interval = a.sphere.bounding_box().axis_interval(axis_index);
    auto b_axis_interval = b.sphere.bounding_box().axis_interval(axis_index);
    
    return a_axis_interval.min < b_axis_interval.min;
}

__device__
static bool box_x_compare (const hittable& a, const hittable& b) {
    return box_compare(a, b, 0);
}
__device__
static bool box_y_compare (const hittable& a, const hittable& b) {
    return box_compare(a, b, 1);
}
__device__
static bool box_z_compare (const hittable& a, const hittable& b) {
    return box_compare(a, b, 2);
}

__device__
bool object_hit(const ray& r, interval ray_t, hit_record& rec, BVHNodeSoA* nodes, hittable* hittables, int* node_stack_arr) {
    // const int MAX = 12;
    // int node_stack_arr[MAX];
    int top = -1;

    node_stack_arr[++top] = 0;  // Start with the root node index

    bool hit_anything = false;
    hit_record temp_rec;

    while (top >= 0) {
        int node_index = node_stack_arr[top--];  // Pop the node index

        const AaBb& current_bbox = nodes->bbox[node_index];
        const int current_left_child = nodes->left_child_index[node_index];
        const int current_right_child = nodes->right_child_index[node_index];
        const bool current_is_leaf = nodes->is_leaf[node_index];
        const int current_node_object_index = nodes->object_index[node_index];

        if (!current_bbox.hit(r, ray_t)) {
            continue;
        }

        if (current_is_leaf) {
            if (hittables[current_node_object_index].sphere.hit(r, ray_t, temp_rec)) {
                hit_anything = true;
                ray_t.max = temp_rec.t;
                rec = temp_rec;
            }
        } else {
            if (current_left_child != -1) {
                if (top + 1 >= MAX_STACK_SIZE) {
                    printf("Error: Stack overflow in object_hit()  increase STACK SIZE\n");
                    return false;
                }
                node_stack_arr[++top] = current_left_child;
            }
            if (current_right_child != -1) {
                if (top + 1 >= MAX_STACK_SIZE) {
                    printf("Error: Stack overflow in object_hit()  increase STACK SIZE\n");
                    return false;
                }
                node_stack_arr[++top] = current_right_child;
            }
        }
    }
    return hit_anything;
}


__device__
bool hit(const ray& r, interval ray_t, hit_record& rec, BVHNodeSoA* &nodes, hittable* &hittables, int* node_stack_arr )  {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
   

    if(object_hit(r, interval(ray_t.min, closest_so_far), temp_rec, nodes, hittables, node_stack_arr)){
        
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
    }
    

    return hit_anything;
}


