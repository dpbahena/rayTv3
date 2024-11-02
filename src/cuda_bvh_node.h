#pragma once

#include "sphere.h"
#include "aabb.h"
#include <thrust/sort.h>



// __device__ __host__
// BVHNode* bvh_nodes;


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
};

__device__ static bool box_compare(const hittable& a, const hittable& b, int axis_index);
__device__ static bool box_x_compare (const hittable& a, const hittable& b);
__device__ static bool box_y_compare (const hittable& a, const hittable& b);
__device__ static bool box_z_compare (const hittable& a, const hittable& b);



const interval interval::empty = interval(+MAXFLOAT, -MAXFLOAT);
const interval interval::universe = interval(-MAXFLOAT, + MAXFLOAT);

const AaBb AaBb::empty    = AaBb(interval::empty,    interval::empty,    interval::empty);
const AaBb AaBb::universe = AaBb(interval::universe, interval::universe, interval::universe);





__global__ void build_bvh_NR(BVHNode* nodes, hittable* hittables, size_t N) {
    

    //std::vector<BVHNode*> node_stack;
    int index = 0;
    const int MAX = 1024;
    StackNode traversalStack[MAX];
    int top = -1 ;  // initialize stack
    
    traversalStack[++top] ={0, N, -1, true};  // push()

    // while (top >= 0) {
    for (;top >= 0;) {
        
        StackNode current = traversalStack[top];
        --top; // pop()

        size_t object_span = current.end - current.start;

        // **Compute the bounding box of the current node upfront**
        AaBb bbox; // = AaBb::empty;
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

            thrust::sort(hittables + current.start, hittables + current.end, comparator);

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

    // **Assign the built nodes to the class member**
    // if (!node_stack.empty()) {
    //     nodes = node_stack;
    // }
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
       