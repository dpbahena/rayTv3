#pragma once

#include "sphere.h"
#include "aabb.h"
#include <thrust/sort.h>

// struct BVHNode {
//     int left_child_index = -1;     // Index of left child in the BVH array (-1 if it's a leaf)
//     int right_child_index = -1;    // Index of right child in the BVH array (-1 if it's a leaf)
//     int object_index = -1;         // Index of the object (used if it's a leaf)
//     bool is_leaf;             // Is this node a leaf?
//     AaBb bbox;
// };


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
            // index++;
            // int node_index = index - 1;
            int node_index = index++;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                // auto& parent_node = nodes[current.parentIndex];

                // auto& parent_node_is_leaf = nodes->is_leaf[current.parentIndex];
                // auto& parent_node_left_child = nodes->left_child_index[current.parentIndex];
                // auto& parent_node_right_child = nodes->right_child_index[current.parentIndex];
                // auto& parent_node_object_index = nodes->object_index[current.parentIndex];
                // auto& parent_node_bbox = nodes->bbox[current.parentIndex];

                // printf("current ParentIndex: %d\n", (int)node_index);
                if (current.isLeftChild) {
                    // parent_node.left_child_index = node_index;
                    nodes->left_child_index[current.parentIndex] = node_index;  // parent node
                    // parent_node_left_child = node_index;
                } else {
                    // parent_node_right_child = node_index;
                    nodes->right_child_index[current.parentIndex] = node_index;  // parent node
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
            index++;
            int left_node_index = index - 1;

            // Right leaf node
            nodes->is_leaf[index] = true;
            nodes->object_index[index] = current.start + 1;
            nodes->left_child_index[index] = -1;
            nodes->right_child_index[index] = -1;
            nodes->bbox[index] = hittables[nodes->object_index[index]].sphere.bounding_box();  // comment later for "optimization 3.10"
            index++;
            int right_node_index = index - 1;

            // bbox = AaBb(nodes->bbox[left_node_index], nodes->bbox[right_node_index]);

            // **Create parent node with bounding box**
            // BVHNode parent_node;
            nodes->is_leaf[index] = false;
            nodes->left_child_index[index] = left_node_index;
            nodes->right_child_index[index] = right_node_index;
            // nodes->object_index[index] = -1; // internal nodes have not objects in them
            nodes->bbox[index] = bbox;  // Bounding box of the two objects
            index++;
            // node_stack.push_back(new BVHNode(parent_node));
            int parent_index = index - 1;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                // auto& grandparent_node = nodes[current.parentIndex];
                // auto& grandparent_node_left_child = nodes->left_child_index[current.parentIndex];
                // auto& grandparent_node_righ_child = nodes->right_child_index[current.parentIndex];
                
                if (current.isLeftChild) {
                    nodes->left_child_index[current.parentIndex] = parent_index;
                    // grandparent_node_left_child = parent_index;
                } else {
                    // grandparent_node_righ_child = parent_index;
                    nodes->right_child_index[current.parentIndex] = parent_index;
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
            // BVHNode node;
            nodes->is_leaf[index] = false;
            nodes->left_child_index[index] = -1;  // Will be set after processing children
            nodes->right_child_index[index] = -1; // Will be set after processing children
            // nodes->object_index[index] = -1;  // internal nodes have not objects in them
            nodes->bbox[index] = bbox;  // Bounding box of all objects in the span
            // index++;
            // int node_index = index - 1;
            int node_index = index++;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                // auto& parent_node = nodes[current.parentIndex];
                // auto& parent_node_left_child = nodes->left_child_index[current.parentIndex];
                // auto& parent_node_right_child = nodes->right_child_index[current.parentIndex];
                if (current.isLeftChild) {
                    // parent_node_left_child = node_index;
                    nodes->left_child_index[current.parentIndex] = node_index;
                } else {
                    // parent_node_right_child = node_index;
                    nodes->right_child_index[current.parentIndex] = node_index;
                }
            }
            if (top + 2 >= MAX) {
                printf("Error: Stack overflow in the build_bvh_NR\n");
                return;
            }

            // **Push child nodes onto the stack for further processing**
            
            // printf("mid: %d, end: %d\n", (int)mid, (int)current.end);
            // printf("start: %d, mid: %d\n", (int)current.start, (int)mid);
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



// __device__ 
// bool hit31(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
//     const int MAX = 15;
//     __shared__ int node_stack_arr[MAX];  // Store node indices instead of pointers
//     __shared__ int top;
    
//     if (threadIdx.x == 0) top = 0; // Initialize top only once per block
//     __syncthreads();

//     bool hit_anything = false;
//     hit_record temp_rec;

//     // Initial push by thread 0, using index 0 for the root node
//     if (threadIdx.x == 0 && top < MAX) {
//         node_stack_arr[atomicAdd(&top, 1)] = 0; // Push root node index
//     }
//     __syncthreads();

//     // Loop until stack is empty
//     while (top > 0) {
//         int current_top = atomicAdd(&top, -1) - 1;  // Pop a node and decrement top
//         if (current_top < 0) break;  // Exit if stack is empty

//         // Access the current node using its index
//         BVHNode* current = &nodes[node_stack_arr[current_top]];

//         if (!current->bbox.hit(r, ray_t)) continue;

//         // Push children onto stack if they exist and within bounds
//         if (current->left_child_index != -1 && top < MAX) {
//             node_stack_arr[atomicAdd(&top, 1)] = current->left_child_index;
//         }
//         if (current->right_child_index != -1 && top < MAX) {
//             node_stack_arr[atomicAdd(&top, 1)] = current->right_child_index;
//         }

//         if (current->is_leaf && (hittables + current->object_index)->sphere.hit(r, ray_t, temp_rec)) {
//             hit_anything = true;
//             ray_t.max = temp_rec.t;
//             rec = temp_rec;
//         }

//         __syncthreads();
//     }
//     return hit_anything;
// }


// __device__
// bool hit3(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
    
//     const int MAX = 15;
//     BVHNode* node_stack_arr[MAX];  // created on stack 
//     int top = -1;        
    
//     node_stack_arr[++top] = (nodes + 0);
    
//     bool hit_anything = false;
    
//     // Initialize a temporary record to store the closest hit found during the traversal.
//     hit_record temp_rec;
//     // Loop until there are no more nodes to process in the stack.
//     while (top >= 0) {
        
//         // Retrieve and remove the top node from the stack.
//         const BVHNode* current = node_stack_arr[top];
//         --top;
//         // Check if the ray intersects the bounding box of the current node.
//         // If not, skip further processing for this node.
//         if (!current->bbox.hit(r, ray_t)) {
//             continue;

//         }
//         // If the current node has a left child node, add it to the stack for further processing.
//         if (current->left_child_index != -1 ) {
//             node_stack_arr[++top] = (nodes + current->left_child_index);
//         }
//         // If the current node has a right child node, add it to the stack for further processing.
//         if (current->right_child_index != -1 ) {
//             node_stack_arr[++top] = (nodes + current->right_child_index);
//         }
//         // Perform a hit test on the left child if it is a leaf node (i.e., it contains an actual object).
//         if (current->is_leaf && (hittables + current->object_index)->sphere.hit(r, ray_t, temp_rec)) {
            
//             hit_anything = true; // A hit was found; update the hit_left flag and the closest hit record.
//             // Update the maximum boundary of the ray interval to the hit point, ensuring that any subsequent hits are closer than the current hit.
//             ray_t.max = temp_rec.t;
//             rec = temp_rec;  // Update the closest hit record with the details of the new closest hit.
//         }
//     }
//     return hit_anything;  // Return true if a hit was detected

// }

// __device__
// bool hit3x(const ray& r, interval ray_t, hit_record& rec, BVHNodeSoA* &nodes, hittable* &hittables) {
    
//     const int MAX = 12;
//     int node_stack_arr[MAX];  // created on stack 
//     int top = -1;        
    
//     // node_stack_arr[++top] = (nodes + 0);
//     node_stack_arr[++top] = 0;  // or any
    
//     bool hit_anything = false;
    
//     // Initialize a temporary record to store the closest hit found during the traversal.
//     hit_record temp_rec;
//     // Loop until there are no more nodes to process in the stack.
//     while (top >= 0) {
        
//         // Retrieve and remove the top node from the stack.
//         // const BVHNode* current = node_stack_arr[top];
//         const AaBb current_bbox = nodes->bbox[node_stack_arr[top]];
//         const int current_left_child = nodes->left_child_index[node_stack_arr[top]];
//         const int current_right_child = nodes->right_child_index[node_stack_arr[top]];
//         const bool current_is_leaf = nodes->is_leaf[node_stack_arr[top]];
//         const int current_node_object_index = nodes->object_index[top];
//         --top;
//         // Check if the ray intersects the bounding box of the current node.
//         // If not, skip further processing for this node.
//         if (!current_bbox.hit(r, ray_t)) {
//             continue;

//         }
//         // If the current node has a left child node, add it to the stack for further processing.
//         // const int current_left_child = nodes->left_child_index[node_stack_arr[top]];
//         if (current_left_child != -1 ) {
//             node_stack_arr[++top] =  nodes->left_child_index[current_left_child];   //(nodes + current->left_child_index);
//         }
//         // If the current node has a right child node, add it to the stack for further processing.
//         // const int current_right_child = nodes->right_child_index[node_stack_arr[top]];
//         if (current_right_child != -1 ) {
//             node_stack_arr[++top] = nodes->right_child_index[current_right_child];  //(nodes + current->right_child_index);
//         }
        
//         // Perform a hit test on the left child if it is a leaf node (i.e., it contains an actual object).
//         // const bool current_is_leaf = nodes->is_leaf[node_stack_arr[top]];
        
//         if (current_is_leaf && hittables[current_node_object_index].sphere.hit(r, ray_t, temp_rec)) {
            
//             hit_anything = true; // A hit was found; update the hit_left flag and the closest hit record.
//             // Update the maximum boundary of the ray interval to the hit point, ensuring that any subsequent hits are closer than the current hit.
//             ray_t.max = temp_rec.t;
//             rec = temp_rec;  // Update the closest hit record with the details of the new closest hit.
//         }
//     }
//     return hit_anything;  // Return true if a hit was detected

// }


__device__
bool hit3x(const ray& r, interval ray_t, hit_record& rec, BVHNodeSoA* nodes, hittable* hittables) {
    const int MAX = 15;
    int node_stack_arr[MAX];
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
                if (top + 1 >= MAX) {
                    printf("Error: Stack overflow in hit3x\n");
                    return false;
                }
                node_stack_arr[++top] = current_left_child;
            }
            if (current_right_child != -1) {
                if (top + 1 >= MAX) {
                    printf("Error: Stack overflow in hit3x\n");
                    return false;
                }
                node_stack_arr[++top] = current_right_child;
            }
        }
    }
    return hit_anything;
}


__device__
bool hit(const ray& r, interval ray_t, hit_record& rec, BVHNodeSoA* &nodes, hittable* &hittables )  {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
   

    if(hit3x(r, interval(ray_t.min, closest_so_far), temp_rec, nodes, hittables)){
        
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
    }
    

    return hit_anything;
}


