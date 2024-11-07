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
    int* rope_index = nullptr;                 // rope index
    bool* is_leaf = nullptr;             // Is this node a leaf?

    AaBb* bbox = nullptr;
};

struct StackNode {
            size_t start{}, end{};
            int parentIndex{};
            bool isLeftChild{};
            int parentRopeIndex{};
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
    
    traversalStack[++top] ={0, N, -1, true, -1};  // push()

    while (top >= 0 ) {
        
        StackNode current = traversalStack[top--];  // equals top then pop()

        size_t object_span = current.end - current.start;
    

        // **Compute the bounding box of the current node upfront**
        AaBb bbox = AaBb::empty(); 
        for (size_t i = current.start; i < current.end; ++i) {
            bbox = AaBb(bbox, hittables[i].sphere.bounding_box());
        }

        int node_index = index++;
        nodes->bbox[node_index] = bbox;
        nodes->rope_index[node_index] = -1; //* initialize rope to null

        if (object_span <= 2) {
            // **Handle leaf nodes**
            nodes->is_leaf[node_index] = true;
            nodes->object_index[node_index] = current.start;
            nodes->left_child_index[node_index] = -1;
            nodes->right_child_index[node_index] = -1;
            // nodes->rope_index[index] = -1;
            // nodes->bbox[index] = bbox;  // Bounding box of the single object
            // int node_index = index++;

            //* Set ropt to parent's node
            nodes->rope_index[node_index] = current.parentRopeIndex;


            // **Update parent's child index**
            if (current.parentIndex != -1) {

                auto& parent_node_left_child_index = nodes->left_child_index[current.parentIndex];
                auto& parent_node_right_child_index = nodes->right_child_index[current.parentIndex];
                // auto& parent_node_rope_index = nodes->rope_index[current.parentIndex];

                if (current.isLeftChild) {
                    parent_node_left_child_index = node_index;
                    // nodes->rope_index[node_index] = parent_node_right_child_index;
                } else {
                    parent_node_right_child_index = node_index;
                    // nodes->rope_index[node_index] = parent_node_rope_index;
                }
            }

       

        } else {
            //* Handle internal nodes

            nodes->is_leaf[node_index] = false;
            nodes->left_child_index[node_index] = -1;
            nodes->right_child_index[node_index] = -1;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                auto& parent_node_left_child_index = nodes->left_child_index[current.parentIndex];
                auto& parent_node_right_child_index = nodes->right_child_index[current.parentIndex];
                // auto& parent_node_rope_index =  nodes->rope_index[current.parentIndex];

                if (current.isLeftChild) {
                    parent_node_left_child_index = node_index;
                    // nodes->rope_index[node_index] = parent_node_right_child_index;
                } else {
                    parent_node_right_child_index = node_index;
                    // nodes->rope_index[node_index] = parent_node_rope_index;
                }
            }



            // **Select the longest axis based on the bounding box**
            int axis = bbox.longest_axis();  // Implement this function in your aabb class

            // **Sort objects along the selected axis**
            auto comparator = (axis == 0) ? box_x_compare
                            : (axis == 1) ? box_y_compare
                                            : box_z_compare;

            thrust::sort(thrust::device, hittables + current.start, hittables + current.end, comparator);

            // **Split the objects into two halves**
            size_t mid = current.start + object_span / 2;

            // **Set ropes for child nodes

            int leftRopeIndex = nodes->right_child_index[node_index];
            int rightRopeIndex = current.parentRopeIndex;

            // nodes->is_leaf[index] = false;
            // nodes->left_child_index[index] = -1;  // Will be set after processing children
            // nodes->right_child_index[index] = -1; // Will be set after processing children
            // nodes->rope_index[index] = -1; 
            // nodes->bbox[index] = bbox;  // Bounding box of all objects in the span
            // int node_index = index++;

            // **Update parent's child index**
            // if (current.parentIndex != -1) {
            //     auto& parent_node_left_child_index = nodes->left_child_index[current.parentIndex];
            //     auto& parent_node_right_child_index = nodes->right_child_index[current.parentIndex];
            //     auto& parent_node_rope_index =  nodes->rope_index[current.parentIndex];

            //     if (current.isLeftChild) {
            //         parent_node_left_child_index = node_index;
            //         nodes->rope_index[node_index] = parent_node_right_child_index;
            //     } else {
            //         parent_node_right_child_index = node_index;
            //         nodes->rope_index[node_index] = parent_node_rope_index;
            //     }
            // }

            // **Push child nodes onto the stack with appropriate rope indices for further processing**
            
            traversalStack[++top] = {mid, current.end, node_index, false, rightRopeIndex};        // Right child
            traversalStack[++top] = {current.start, mid, node_index, true, leftRopeIndex};       // Left child
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

// __device__
// bool object_hit(const ray& r, interval ray_t, hit_record& rec, const BVHNodeSoA* __restrict__ nodes, const hittable* __restrict__ hittables, int* node_stack_arr) {
//     // const int MAX = 12;
//     // int node_stack_arr[MAX];
//     int top = -1;

//     node_stack_arr[++top] = 0;  // Start with the root node index

//     bool hit_anything = false;
//     hit_record temp_rec;

//     while (top >= 0) {
//         int node_index = node_stack_arr[top--];  // Pop the node index
//         //* OPTION 1. Let the compiler optimize optimize memory access _restrict__  ---> UNCOMMENT OR COMMENT 
//         //! FASTER
//         const AaBb& current_bbox = nodes->bbox[node_index];
//         const int current_left_child = nodes->left_child_index[node_index];
//         const int current_right_child = nodes->right_child_index[node_index];
//         const int current_is_leaf = nodes->is_leaf[node_index];
//         const int current_node_object_index = nodes->object_index[node_index];
        

//         //* OPTION 2:  You load data directly using __ldg   ---> UNCOMMENT or COMMENT
//         //! SLOWER
//         // double min_x = __ldg(&nodes->bbox[node_index].x.min);  // needs conversion to accepted types
//         // double min_y = __ldg(&nodes->bbox[node_index].y.min);
//         // double min_z = __ldg(&nodes->bbox[node_index].z.min);
//         // double max_x = __ldg(&nodes->bbox[node_index].x.max);
//         // double max_y = __ldg(&nodes->bbox[node_index].y.max);
//         // double max_z = __ldg(&nodes->bbox[node_index].z.max);
//         // interval x = interval(min_x, max_x);
//         // interval y = interval(min_y, max_y);
//         // interval z = interval(min_z, max_z);
//         // AaBb current_bbox = AaBb(x, y, z);  //! faster
//         // AaBb current_bbox = AaBb(glm::vec3(min_x, min_y, min_z), glm::vec3(max_x, max_y, max_z)); //! slower
//         // const int current_left_child = __ldg(&nodes->left_child_index[node_index]);
//         // const int current_right_child = __ldg(&nodes->right_child_index[node_index]);
//         // int8_t is_leaf = __ldg(reinterpret_cast<const int8_t*>(&nodes->is_leaf[node_index])); // needs conversion to accepted types
//         // bool current_is_leaf = static_cast<bool>(is_leaf);
//         // const int current_node_object_index = __ldg(&nodes->object_index[node_index]);

//         if (!current_bbox.hit(r, ray_t)) {
//             continue;
//         }

//         if (current_is_leaf) {
            
//             if (hittables[current_node_object_index].sphere.hit(r, ray_t, temp_rec)) {
//                 hit_anything = true;
//                 ray_t.max = temp_rec.t;
//                 rec = temp_rec;
//             }
//         } else {
//             if (current_left_child != -1) {
//                 if (top + 1 >= MAX_STACK_SIZE) {
//                     printf("Error: Stack overflow in object_hit()  increase STACK SIZE\n");
//                     return false;
//                 }
//                 node_stack_arr[++top] = current_left_child;
//             }
//             if (current_right_child != -1) {
//                 if (top + 1 >= MAX_STACK_SIZE) {
//                     printf("Error: Stack overflow in object_hit()  increase STACK SIZE\n");
//                     return false;
//                 }
//                 node_stack_arr[++top] = current_right_child;
//             }
//         }
//     }
//     return hit_anything;
// }

__device__
bool object_hit_rope(const ray& r, interval ray_t, hit_record& rec, const BVHNodeSoA* __restrict__ nodes, const hittable* __restrict__ hittables, int* node_stack_arr) {
    
  
    BVHNodeSoA current;
    current.bbox = nodes->bbox;
    current.is_leaf = nodes->is_leaf;
    current.left_child_index = nodes->left_child_index;
    current.right_child_index = nodes->right_child_index;
    current.rope_index = nodes->rope_index;
    current.object_index = nodes->object_index;
    

    bool hit_anything = false;
    hit_record temp_rec;

    while (*current.rope_index != -1) {
               
        if (current.bbox->hit(r, ray_t)) {
            
            if (*current.is_leaf) {
                printf("current_object_indes %d\n", *current.object_index);
                if (hittables[*current.object_index].sphere.hit(r, ray_t, temp_rec)) {
                    
                    hit_anything = true;
                    ray_t.max = temp_rec.t;
                    rec = temp_rec;
                }
                // Move to the next node using the rope
               
                if (*current.rope_index != -1) {
                    
                    current.bbox = &nodes->bbox[*current.rope_index];
                    current.is_leaf = &nodes->is_leaf[*current.rope_index];
                    current.left_child_index = &nodes->left_child_index[*current.rope_index];
                    current.right_child_index = &nodes->right_child_index[*current.rope_index];
                    current.rope_index = &nodes->rope_index[*current.rope_index];
                    current.object_index = &nodes->object_index[*current.rope_index];
                } else {
                    
                    *current.bbox = AaBb::empty(); 
                    *current.is_leaf = -1;
                    *current.left_child_index = -1; 
                    *current.right_child_index = -1; 
                    *current.rope_index = -1; 
                    *current.object_index = -1;
                }




            } else {
                // If it is not a leaf node, first go to the left child
                
                current.bbox = &nodes->bbox[*current.left_child_index];
                current.is_leaf = &nodes->is_leaf[*current.left_child_index];
                current.left_child_index = &nodes->left_child_index[*current.left_child_index];
                current.right_child_index = &nodes->right_child_index[*current.left_child_index];
                current.rope_index = &nodes->rope_index[*current.left_child_index];
                current.object_index = &nodes->object_index[*current.left_child_index];
                
            }
        } else {
            // If there's no intersection with the current node's bounding box, follow the rope
            
            if (*current.rope_index != -1) {
                current.bbox = &nodes->bbox[*current.rope_index];
                current.is_leaf = &nodes->is_leaf[*current.rope_index];
                current.left_child_index = &nodes->left_child_index[*current.rope_index];
                current.right_child_index = &nodes->right_child_index[*current.rope_index];
                current.rope_index = &nodes->rope_index[*current.rope_index];
                current.object_index = &nodes->object_index[*current.rope_index];
            } else {
                *current.bbox = AaBb::empty();
                *current.is_leaf = -1; 
                *current.left_child_index = -1; 
                *current.right_child_index = -1; 
                *current.rope_index = -1; 
                *current.object_index = -1; 
            }
        }
    }
    
    return hit_anything;
}


__device__
bool hit(const ray& r, interval ray_t, hit_record& rec, const BVHNodeSoA* __restrict__ nodes, const hittable* __restrict__ hittables, int* node_stack_arr )  {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
   

    if(object_hit_rope(r, interval(ray_t.min, closest_so_far), temp_rec, nodes, hittables, node_stack_arr)){
        
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
    }
    

    return hit_anything;
}


