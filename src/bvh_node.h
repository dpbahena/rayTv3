#pragma once

// #include "aabb.h"
// #include "sphere.h"

#include <algorithm>
#include <stack>

inline double random_double(float min, float max);

/* Stores a bounding box, the indices for the left and right children, and whether the node is a leaf */
// struct BVHNode {
//     AaBb bbox;
//     int left_child_index; // Index left child in the array, -1 if it's a leaf
//     int right_child_index; // index right child in the array, -1 if it's a leaf
//     int object_index;       // index of the object the leaf represent (if it's a leaf)
//     bool is_leaf;


// };


class BVH {
    public:
        BVH(const hittable_list& objects) {
            build_bvh(objects);
        }
        // bool hit(const ray& r, interval ray_t, hit_record& rec, const hittable_list& objects) const {
        //     return hit_bvh(r, ray_t, rec, objects);
        // }

        AaBb bounding_box() const {
            return nodes[0].bbox;  // Root node's bounding box
        }



    private:
    std::vector<BVHNode> nodes;

    void build_bvh(const hittable_list& objects) {
    nodes.clear();

    struct StackNode {
        size_t start, end;
        int parentIndex;
        bool isLeftChild;
    };

    std::vector<BVHNode> node_stack;
    std::stack<StackNode> traversalStack;

    traversalStack.push({(0, objects.objects_size, -1, true)});

    while (!traversalStack.empty()) {
        auto current = traversalStack.top();
        traversalStack.pop();

        size_t object_span = current.end - current.start;
        BVHNode node;

        if (object_span == 1) {
            node.is_leaf = true;
            node.object_index = current.start;
            node.left_child_index = -1;
            node.right_child_index = -1;
        } else {
            int axis = int(random_double(0, 2));
            auto comparator = (axis == 0) ? box_x_compare :
                              (axis == 1) ? box_y_compare : box_z_compare;

            std::sort(objects.list + current.start, objects.list + current.end, comparator);
            
            size_t mid = current.start + object_span / 2;
            traversalStack.push({mid, current.end, int(node_stack.size()), false});
            traversalStack.push({current.start, mid, int(node_stack.size()), true});
            
            node.is_leaf = false;
        }

        if (current.parentIndex != -1) {
            if (current.isLeftChild) {
                node_stack[current.parentIndex].left_child_index = int(node_stack.size());
            } else {
                node_stack[current.parentIndex].right_child_index = int(node_stack.size());
            }
        }

        node_stack.push_back(node);
    }

    nodes = node_stack;
}


    // void build_bvh(const hittable_list& objects) {
    //     nodes.clear();

    //     std::vector<BVHNode> node_stack;
    //     node_stack.reserve(objects.objects_size);

    //     /* Build the BVH with the objects from the hittable_list */
    //     recursive_build(objects, 0, objects.objects_size, node_stack);

    //     /* Store the resulting BVH in a flattened array */
        
    //     nodes = node_stack;

    // }

    // void recursive_build(const hittable_list& objects, size_t start, size_t end, std::vector<BVHNode>& node_stack) {
    //     size_t object_span = end - start;
    //     BVHNode node;

    //     if(object_span == 1) {
    //         node.is_leaf = true;
    //         node.object_index = start; // Directly use index as object reference
    //         node.left_child_index = -1;
    //         node.right_child_index = -1;
    //     } else {

    //         /* Internal node case: sort the objects by a random axis */
    //         int axis = int(random_double(0, 2)); // choose an axis to split
    //         auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

    //         /* Sort the objects along the chosen axis */
            
           
    //         std::sort(objects.list + start, objects.list + end, comparator);
            
            
    //         /* Divide the objects into 2 groups and recursively build left and right subtrees */
    //         auto mid = start + object_span / 2;

    //         recursive_build(objects, start, mid, node_stack);  // build left subtree
    //         recursive_build(objects, mid, end, node_stack);     // build right subtree

    //         /* Set left and right child indices */
    //         node.left_child_index = node_stack.size() - 2;
    //         node.right_child_index = node_stack.size() - 1;
            
    //         node.is_leaf = false;

    //         /* Conpute bounding box for interneal node by combining left and right child boxes */
    //         node.bbox = AaBb(node_stack[node.left_child_index].bbox, node_stack[node.right_child_index].bbox);
    //     }

    //     /* Add the node to the stack (flat BVH array )*/
    //     node_stack.push_back(node);

    // }
    
    bool hit_bvh(const ray& r, interval ray_t, hit_record& rec, const hittable_list& objects) const {

        bool hit_anything = false;
        hit_record temp_rec;
        int current_node_index = 0;  // start from the root node

        while (current_node_index != -1) {
            const BVHNode& node = nodes[current_node_index];
            
            if (!node.bbox.hit(r, ray_t)) {
                break ; // Skip if the bounding box is not hit
            }
            if (node.is_leaf) {
                // check for object hit using hittable_list passed as a parameter 
                // if (objects.list[node.object_index].hit(r, ray_t, temp_rec)){
                if (objects.list[node.object_index].type == Type::SPHERE) { 
                    if( hit_sphere(r, ray_t, objects.list[node.object_index].sphere, temp_rec )){
                    hit_anything = true;
                    ray_t.max = temp_rec.t;  // Update interval for closer hit
                    rec = temp_rec;
                    }
                }
                break;
            } else {
                /* Internal node, check children */
                bool hit_left =  nodes[node.left_child_index].bbox.hit(r, ray_t);
                bool hit_right = nodes[node.right_child_index].bbox.hit(r, ray_t);

                if (hit_left && hit_right) {
                    current_node_index = node.left_child_index; // Go to the left first

                } else if (hit_left) {
                    current_node_index = node.left_child_index;
                } else if (hit_right) {
                    current_node_index = node.right_child_index;
                } else {
                    break;  // Neither child is hit, end transversal
                }
            }
        }

        return hit_anything;
    }



    
    static bool box_compare(const hittable& a, const hittable& b, int axis_index) {
        
        /* Compare the bounding boxes of two objects along the specific axis */
       
        auto a_axis_interval = a.sphere.bbox->axis_interval(axis_index);
        auto b_axis_interval = b.sphere.bbox->axis_interval(axis_index);

        return a_axis_interval.min < b_axis_interval.min; 
    }

    // static bool box_x_compare ( BVHNode* a,  BVHNode* b){
    static bool box_x_compare (const hittable& a, const hittable& b){
        return box_compare(a, b, 0);
    }
    // static bool box_y_compare ( BVHNode* a,  BVHNode* b){
    static bool box_y_compare (const hittable& a, const hittable& b){
        return box_compare(a, b, 1);
    }
    // static bool box_z_compare ( BVHNode* a,  BVHNode* b){
    static bool box_z_compare (const hittable& a, const hittable& b){
        return box_compare(a, b, 2);
    }

};

