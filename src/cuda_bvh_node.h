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


struct StackNode {
            size_t start, end;
            int parentIndex;
            bool isLeftChild;           
            // int parentRopeIndex;
            int ropeIndex;
};

__device__ static bool box_compare(const hittable& a, const hittable& b, int axis_index);
__device__ static bool box_x_compare (const hittable& a, const hittable& b);
__device__ static bool box_y_compare (const hittable& a, const hittable& b);
__device__ static bool box_z_compare (const hittable& a, const hittable& b);


__global__ void build_bvh_NR(BVHNode* nodes, hittable* hittables, size_t N) {
    
    //std::vector<BVHNode*> node_stack;
    int index = 0;
    const int MAX = 15;
    StackNode traversalStack[MAX];
    int top = -1 ;  // initialize stack
    
    traversalStack[++top] ={0, N, -1, true};  // push()

    // while (top >= 0) {
    for (;top >= 0;) {
        
        StackNode current = traversalStack[top];
        --top; // pop()

        size_t object_span = current.end - current.start;

        // **Compute the bounding box of the current node upfront**
        AaBb bbox = AaBb::empty(); // = AaBb::empty;
        for (size_t i = current.start; i < current.end; ++i) {
            bbox = AaBb(bbox, (hittables + i)->sphere.bounding_box());
        }

        if (object_span == 1) {
            // **Create a leaf node**
            // BVHNode node;
            nodes[index].is_leaf = true;
            nodes[index].object_index = current.start;
            nodes[index].left_child_index = -1;
            nodes[index].right_child_index = -1;
            nodes[index].bbox = bbox;  // Bounding box of the single object
            index++;
            // node_stack.push_back(new BVHNode(node));
            int node_index = index - 1;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                auto& parent_node = nodes[current.parentIndex];
                if (current.isLeftChild) {
                    parent_node.left_child_index = node_index;
                } else {
                    parent_node.right_child_index = node_index;
                }
            }

        } else if (object_span == 2) {
            // **Create leaf nodes for both objects**
            // Left leaf node
            // BVHNode left_node;
            nodes[index].is_leaf = true;
            nodes[index].object_index = current.start;
            nodes[index].left_child_index = -1;
            nodes[index].right_child_index = -1;
            nodes[index].bbox = (hittables + nodes[index].object_index)->sphere.bounding_box();  // comment later for "optimization 3.10"
            index++;
            // node_stack.push_back(new BVHNode(left_node));
            int left_node_index = index - 1;

            // Right leaf node
            // BVHNode right_node;
            nodes[index].is_leaf = true;
            nodes[index].object_index = current.start + 1;
            nodes[index].left_child_index = -1;
            nodes[index].right_child_index = -1;
            nodes[index].bbox = (hittables + nodes[index].object_index)->sphere.bounding_box();  // comment later for "optimization 3.10"
            index++;
            // node_stack.push_back(new BVHNode(right_node));
            int right_node_index = index - 1;

            // **Create parent node with bounding box**
            // BVHNode parent_node;
            nodes[index].is_leaf = false;
            nodes[index].left_child_index = left_node_index;
            nodes[index].right_child_index = right_node_index;
            nodes[index].bbox = bbox;  // Bounding box of the two objects
            index++;
            // node_stack.push_back(new BVHNode(parent_node));
            int parent_index = index - 1;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                auto& grandparent_node = nodes[current.parentIndex];
                if (current.isLeftChild) {
                    grandparent_node.left_child_index = parent_index;
                } else {
                    grandparent_node.right_child_index = parent_index;
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
            nodes[index].is_leaf = false;
            nodes[index].left_child_index = -1;  // Will be set after processing children
            nodes[index].right_child_index = -1; // Will be set after processing children
            nodes[index].bbox = bbox;  // Bounding box of all objects in the span
            index++;
            // node_stack.push_back(new BVHNode(node));
            int node_index = index - 1;

            // **Update parent's child index**
            if (current.parentIndex != -1) {
                auto& parent_node = nodes[current.parentIndex];
                if (current.isLeftChild) {
                    parent_node.left_child_index = node_index;
                } else {
                    parent_node.right_child_index = node_index;
                }
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
bool hit3(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
    
    const int MAX = 15;
    BVHNode* node_stack_arr[MAX];  // created on stack 
    int top = -1;        
    
    node_stack_arr[++top] = (nodes + 0);
    
    bool hit_anything = false;
    
    // Initialize a temporary record to store the closest hit found during the traversal.
    hit_record temp_rec;
    // Loop until there are no more nodes to process in the stack.
    while (top >= 0) {
        
        // Retrieve and remove the top node from the stack.
        const BVHNode* current = node_stack_arr[top];
        --top;
        // Check if the ray intersects the bounding box of the current node.
        // If not, skip further processing for this node.
        if (!current->bbox.hit(r, ray_t)) {
            continue;

        }
        // If the current node has a left child node, add it to the stack for further processing.
        if (current->left_child_index != -1 ) {
            node_stack_arr[++top] = (nodes + current->left_child_index);
        }
        // If the current node has a right child node, add it to the stack for further processing.
        if (current->right_child_index != -1 ) {
            node_stack_arr[++top] = (nodes + current->right_child_index);
        }
        // Perform a hit test on the left child if it is a leaf node (i.e., it contains an actual object).
        if (current->is_leaf && (hittables + current->object_index)->sphere.hit(r, ray_t, temp_rec)) {
            
            hit_anything = true; // A hit was found; update the hit_left flag and the closest hit record.
            // Update the maximum boundary of the ray interval to the hit point, ensuring that any subsequent hits are closer than the current hit.
            ray_t.max = temp_rec.t;
            rec = temp_rec;  // Update the closest hit record with the details of the new closest hit.
        }
    }
    return hit_anything;  // Return true if a hit was detected

}


// __global__ void build_bvh_NR_ROPE(BVHNode* nodes, hittable* hittables, size_t N) {
//     int index = 0;
//     const int MAX = 15;
//     StackNode traversalStack[MAX];
//     int top = -1;

//     // Initialize stack with the root node
//     traversalStack[++top] = {0, N, -1, true, -1, -1};

//     while (top >= 0) {
//         StackNode current = traversalStack[top];
//         --top;

//         size_t object_span = current.end - current.start;
//         AaBb bbox = AaBb::empty();

//         for (size_t i = current.start; i < current.end; ++i) {
//             bbox = AaBb(bbox, (hittables + i)->sphere.bounding_box());
//         }

//         if (object_span == 1) {
//             // Create a leaf node
//             nodes[index].is_leaf = true;
//             nodes[index].object_index = current.start;
//             nodes[index].bbox = bbox;
//             nodes[index].rope_index = current.ropeIndex;  // Set rope to the parent rope
//             index++;

//             int node_index = index - 1;
//             if (current.parentIndex != -1) {
//                 auto& parent_node = nodes[current.parentIndex];
//                 if (current.isLeftChild) {
//                     parent_node.left_child_index = node_index;
//                 } else {
//                     parent_node.right_child_index = node_index;
//                     parent_node.rope_index = current.parentRopeIndex;  // Link to parent rope
//                 }
//             }
//         } else {
//             int axis = bbox.longest_axis();
//             auto comparator = (axis == 0) ? box_x_compare
//                                           : (axis == 1) ? box_y_compare
//                                                         : box_z_compare;

//             thrust::sort(thrust::device, hittables + current.start, hittables + current.end, comparator);
//             size_t mid = current.start + object_span / 2;

//             nodes[index].is_leaf = false;
//             nodes[index].bbox = bbox;
//             nodes[index].rope_index = current.ropeIndex;
//             index++;

//             int node_index = index - 1;
//             if (current.parentIndex != -1) {
//                 auto& parent_node = nodes[current.parentIndex];
//                 if (current.isLeftChild) {
//                     parent_node.left_child_index = node_index;
//                 } else {
//                     parent_node.right_child_index = node_index;
//                 }
//             }

//             // Set ropes for children
//             traversalStack[++top] = {mid, current.end, node_index, false, current.ropeIndex, node_index};  // Right child
//             traversalStack[++top] = {current.start, mid, node_index, true, (int)mid, node_index};  // Left child
//         }
//     }
// }


__device__
bool hit_rope(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
    const int MAX = 15;
    BVHNode* node_stack_arr[MAX];
    int top = -1;

    // Push the root node onto the stack
    node_stack_arr[++top] = (nodes + 0);
    bool hit_anything = false;
    hit_record temp_rec;

    while (top >= 0) {
        // Pop the current node from the stack
        const BVHNode* current = node_stack_arr[top];
        --top;

        // Check if the ray intersects the bounding box of the current node
        if (!current->bbox.hit(r, ray_t)) {
            continue;
        }

        if (current->is_leaf) {
            // If the current node is a leaf, test for intersection with the object
            if ((hittables + current->object_index)->sphere.hit(r, ray_t, temp_rec)) {
                hit_anything = true;
                ray_t.max = temp_rec.t;
                rec = temp_rec;
            }
        } else {
            // If not a leaf, push child nodes onto the stack
            if (current->left_child_index != -1 && top < MAX - 1) {
                node_stack_arr[++top] = (nodes + current->left_child_index);
            }
            if (current->right_child_index != -1 && top < MAX - 1) {
                node_stack_arr[++top] = (nodes + current->right_child_index);
            }
            // Use rope if no right child and top is within bounds
            else if (current->rope_index != -1 && top < MAX - 1) {
                node_stack_arr[++top] = (nodes + current->rope_index);
            }
        }
    }

    return hit_anything;
}

// __global__ void build_bvh_NR_ROPE2(BVHNode* nodes, hittable* hittables, size_t N) {
//     int index = 0;
//     const int MAX = 15;
//     StackNode traversalStack[MAX];
//     int top = -1;

//     // Initialize stack with the root node
//     traversalStack[++top] = {0, N, -1, true, -1, -1};

//     while (top >= 0) {
//         StackNode current = traversalStack[top];
//         --top;

//         size_t object_span = current.end - current.start;
//         AaBb bbox = AaBb::empty();

//         for (size_t i = current.start; i < current.end; ++i) {
//             bbox = AaBb(bbox, (hittables + i)->sphere.bounding_box());
//         }

//         if (object_span == 1) {
//             // Create a leaf node
//             nodes[index].is_leaf = true;
//             nodes[index].object_index = current.start;
//             nodes[index].bbox = bbox;
//             nodes[index].rope_index = current.ropeIndex;  // Leaf nodes point to parent's rope
//             index++;

//             int node_index = index - 1;
//             if (current.parentIndex != -1) {
//                 auto& parent_node = nodes[current.parentIndex];
//                 if (current.isLeftChild) {
//                     parent_node.left_child_index = node_index;
//                 } else {
//                     parent_node.right_child_index = node_index;
//                 }
//             }
//         } else {
//             int axis = bbox.longest_axis();
//             auto comparator = (axis == 0) ? box_x_compare
//                                           : (axis == 1) ? box_y_compare
//                                                         : box_z_compare;

//             thrust::sort(thrust::device, hittables + current.start, hittables + current.end, comparator);
//             size_t mid = current.start + object_span / 2;

//             nodes[index].is_leaf = false;
//             nodes[index].bbox = bbox;
//             nodes[index].rope_index = current.ropeIndex;  // Internal nodes inherit parent's rope
//             index++;

//             int node_index = index - 1;
//             if (current.parentIndex != -1) {
//                 auto& parent_node = nodes[current.parentIndex];
//                 if (current.isLeftChild) {
//                     parent_node.left_child_index = node_index;
//                 } else {
//                     parent_node.right_child_index = node_index;
//                 }
//             }

//             // Assign ropes: left child points to the right child, right child points to the parent's rope
//             traversalStack[++top] = {mid, current.end, node_index, false, current.ropeIndex, node_index};  // Right child
//             traversalStack[++top] = {current.start, mid, node_index, true, index, node_index};  // Left child
//         }
//     }
// }


__device__
bool hit_rope2(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
    int current_index = 0;
    bool hit_anything = false;
    hit_record temp_rec;

    while (current_index != -1) {
        const BVHNode* current = nodes + current_index;

        // Only print debug information for thread (0, 0)
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("Traversing Node Index: %d, Rope Index: %d\n", current_index, current->rope_index);
        }

        if (current->bbox.hit(r, ray_t)) {
            if (current->is_leaf) {
                if ((hittables + current->object_index)->sphere.hit(r, ray_t, temp_rec)) {
                    hit_anything = true;
                    ray_t.max = temp_rec.t;
                    rec = temp_rec;

                    if (threadIdx.x == 0 && threadIdx.y == 0) {
                        printf("Hit Sphere at Object Index: %d, Distance: %f\n", current->object_index, temp_rec.t);
                    }
                }
                current_index = current->rope_index;
            } else {
                if (current->left_child_index != -1) {
                    current_index = current->left_child_index;
                } else if (current->right_child_index != -1) {
                    current_index = current->right_child_index;
                } else {
                    current_index = current->rope_index;
                }
            }
        } else {
            current_index = current->rope_index;
        }
    }

    return hit_anything;
}

__global__ void build_bvh_NR_ROPE3(BVHNode* nodes, hittable* hittables, size_t N) {
    int index = 0;
    const int MAX = 64;  // Adjust the size as needed
    StackNode traversalStack[MAX];
    int top = -1;

    traversalStack[++top] = {0, N, -1, true, -1};  // {start, end, parentIndex, isLeftChild, ropeIndex}

    while (top >= 0) {
        StackNode current = traversalStack[top--];
        size_t object_span = current.end - current.start;

        // Compute the bounding box of the current node
        AaBb bbox = AaBb::empty();
        for (size_t i = current.start; i < current.end; ++i) {
            bbox = AaBb(bbox, (hittables + i)->sphere.bounding_box());
        }

        int node_index = index++;
        BVHNode& node = nodes[node_index];
        node.bbox = bbox;
        node.start = current.start;
        node.end = current.end;
        node.rope_index = current.ropeIndex;
        node.is_leaf = (object_span <= 2);
        node.left_child_index = -1;
        node.right_child_index = -1;

        // Update parent's child index
        if (current.parentIndex != -1) {
            BVHNode& parent_node = nodes[current.parentIndex];
            if (current.isLeftChild) {
                parent_node.left_child_index = node_index;
            } else {
                parent_node.right_child_index = node_index;
            }
        }

        if (node.is_leaf) {
            // Leaf node logic
            node.object_index = current.start;  // Assuming one object per leaf
            // for (int i = 0; i < index; ++i) {
            //     BVHNode& node = nodes[i];
            //     printf("Node %d: bbox min (%f, %f, %f), max (%f, %f, %f)\n",
            //         i, node.bbox.x.min, node.bbox.y.min, node.bbox.z.min,
            //         node.bbox.x.max, node.bbox.y.max, node.bbox.z.max);
            // }
            // for (int i = 0; i < index; ++i) {
            //     BVHNode& node = nodes[i];
            //     printf("Node %d: is_leaf=%d, start=%zu, end=%zu, object_index=%d\n",
            //         i, node.is_leaf, node.start, node.end, node.object_index);
            // }

            // Ropes for leaf nodes are already set via current.ropeIndex
        } else {
            // Internal node logic

            // Select axis and sort
            int axis = bbox.longest_axis();
            auto comparator = (axis == 0) ? box_x_compare
                            : (axis == 1) ? box_y_compare
                                          : box_z_compare;
            thrust::sort(thrust::device, hittables + current.start, hittables + current.end, comparator);

            // Split the objects into two halves
            size_t mid = current.start + object_span / 2;

            // Prepare rope indices for child nodes
            int left_rope_index = index;               // Right child will be the next node
            int right_rope_index = node.rope_index;    // Inherit from current node

            // Push child nodes onto the stack
            // Right child
            traversalStack[++top] = {mid, current.end, node_index, false, right_rope_index};
            // Left child
            traversalStack[++top] = {current.start, mid, node_index, true, left_rope_index};
        }
    }
}


__device__
bool hit_rope3(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
    const BVHNode* current = nodes;  // Start at root node
    bool hit_anything = false;
    hit_record temp_rec;

    while (current != nullptr) {
        if (current->bbox.hit(r, ray_t)) {
            if (current->is_leaf) {
                // Loop over objects in the leaf node
                for (size_t i = current->start; i < current->end; ++i) {
                    hittable* obj = hittables + i;
                    if (obj->sphere.hit(r, ray_t, temp_rec)) {
                        hit_anything = true;
                        ray_t.max = temp_rec.t;
                        rec = temp_rec;
                    }
                }
                // Move to the next node via the rope
                if (current->rope_index != -1) {
                    current = nodes + current->rope_index;
                } else {
                    current = nullptr;  // End of traversal
                }
            } else {
                // Move to the left child
                current = nodes + current->left_child_index;
            }
        } else {
            // No intersection; follow the rope
            if (current->rope_index != -1) {
                current = nodes + current->rope_index;
            } else {
                current = nullptr;  // End of traversal
            }
        }
    }
    return hit_anything;
}

__global__ void build_bvh_NR_ROPE5(BVHNode* nodes, hittable* hittables, size_t N) {
    int index = 0;
    const int MAX = 64;
    StackNode traversalStack[MAX];
    int top = -1;

    traversalStack[++top] = {0, N, -1, true, -1};

    while (top >= 0) {
        StackNode current = traversalStack[top--];
        size_t object_span = current.end - current.start;

        // Compute the bounding box
        AaBb bbox = AaBb::empty();
        for (size_t i = current.start; i < current.end; ++i) {
            bbox = AaBb(bbox, (hittables + i)->sphere.bounding_box());
        }

        int node_index = index++;
        BVHNode& node = nodes[node_index];
        node.bbox = bbox;
        node.start = current.start;
        node.end = current.end;
        node.rope_index = current.ropeIndex;

        if (object_span <= 2) {
            // Leaf node
            node.is_leaf = true;
            node.left_child_index = -1;
            node.right_child_index = -1;

            // Update parent's child index
            if (current.parentIndex != -1) {
                BVHNode& parent_node = nodes[current.parentIndex];
                if (current.isLeftChild) {
                    parent_node.left_child_index = node_index;
                } else {
                    parent_node.right_child_index = node_index;
                }
            }

        } else {
            // Internal node
            node.is_leaf = false;
            node.left_child_index = -1;
            node.right_child_index = -1;

            // Update parent's child index
            if (current.parentIndex != -1) {
                BVHNode& parent_node = nodes[current.parentIndex];
                if (current.isLeftChild) {
                    parent_node.left_child_index = node_index;
                } else {
                    parent_node.right_child_index = node_index;
                }
            }

            // Select axis and sort
            int axis = bbox.longest_axis();
            thrust::sort(thrust::device, hittables + current.start, hittables + current.end, BoxCompare(axis));

            // Split the objects
            size_t mid = current.start + object_span / 2;

            // Prepare rope indices for child nodes
            int left_rope_index = index;               // Right child will be the next node
            int right_rope_index = node.rope_index;    // Inherit from current node

            // Push child nodes onto the stack
            traversalStack[++top] = {mid, current.end, node_index, false, right_rope_index};
            traversalStack[++top] = {current.start, mid, node_index, true, left_rope_index};
        }
    }
}


__device__
bool hit_rope5(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
    const BVHNode* current = nodes;  // Start at the root node
    bool hit_anything = false;
    hit_record temp_rec;

    while (current != nullptr) {
        if (current->bbox.hit(r, ray_t)) {
            if (current->is_leaf) {
                // Loop over objects in the leaf node
                for (size_t i = current->start; i < current->end; ++i) {
                    hittable* obj = hittables + i;
                    if (obj->sphere.hit(r, ray_t, temp_rec)) {
                        hit_anything = true;
                        ray_t.max = temp_rec.t;
                        rec = temp_rec;
                    }
                }
                // Move to the next node via the rope
                if (current->rope_index != -1) {
                    current = nodes + current->rope_index;
                } else {
                    current = nullptr;  // End of traversal
                }
            } else {
                // Move to the left child
                current = nodes + current->left_child_index;
            }
        } else {
            // No intersection; follow the rope
            if (current->rope_index != -1) {
                current = nodes + current->rope_index;
            } else {
                current = nullptr;  // End of traversal
            }
        }
    }
    return hit_anything;
}



__device__
bool hit(const ray& r, interval ray_t, hit_record& rec, BVHNode* &nodes, hittable* &hittables )  {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
    // bool gotHit;
    
    // if(hit3(r, interval(ray_t.min, closest_so_far), temp_rec, nodes, hittables)){
    if(hit_rope5(r, interval(ray_t.min, closest_so_far), temp_rec, nodes, hittables)){
        
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
    }
    

    return hit_anything;
}


