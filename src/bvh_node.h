#pragma once

// #include "aabb.h"
// #include "sphere.h"
#include "hittable_list.h"
#include "sphere.h"
#include "hittable.h"
#include <algorithm>

inline int random_int(int min, int max);




class BVH {
    public:
        

        BVH(hittable_list list, std::vector<BVH*> &allocated_nodes) : BVH(list.hittables, 0, list.objects_size, allocated_nodes) 
        {
            type = Type::BBOX;
        }
        BVH(hittable* &objects, size_t start, size_t end, std::vector<BVH*> &allocated_nodes) 
        {
            // int axis = random_int(0,2);
            // Build the bounding box of the span of source objects.
            bbox = AaBb::empty;
            for (size_t object_index=start; object_index < end; object_index++)
                bbox = AaBb(bbox, (objects + object_index)->sphere.bounding_box());

            int axis = bbox.longest_axis();
            
            auto comparator = (axis == 0) ? box_x_compare
                            : (axis == 1) ? box_y_compare
                                          : box_z_compare;

            size_t object_span = end - start;
            
            if (object_span == 1) {
                left = right = objects + start;
            } else if (object_span == 2) {
                left = objects + start;
                right = objects + start + 1;
            } else {
                
                std::sort(objects + start , objects + end, comparator);
                
                auto mid = start + object_span / 2;
                leftNode = new BVH(objects, start, mid, allocated_nodes);
                allocated_nodes.push_back(leftNode);
                rightNode = new BVH(objects, mid, end, allocated_nodes);
                allocated_nodes.push_back(rightNode);
            }
            // if (object_span > 2){
            //     bbox = AaBb(leftNode->bounding_box(), rightNode->bounding_box() );
            // } else
            //     bbox = AaBb(left->sphere.bounding_box(), right->sphere.bounding_box());
        }
        
        // NON RECURSIVE - Define the hit function which determines if a ray intersects any object within the BVH structure.
        __device__ __host__
        bool hit(const ray& r, interval ray_t, hit_record& rec) {
            // Initialize a stack to manage the BVH nodes to be processed.
            // std::stack<const BVH*> nodes;
            const int MAX = 1024;
            BVH* stack_arr[MAX];
            int top = -1;  // clear() stack
            
            // Start the traversal with the root node of the BVH.
            stack_arr[++top] = this;  // push() first element into stack

            // Initialize flags to track hits on the left and right children.
            bool hit_left = false;
            bool hit_right = false;
            // Initialize a temporary record to store the closest hit found during the traversal.
            hit_record temp_rec;

            // Loop until there are no more nodes to process in the stack.
            while (top >= 0) {   // empty() stack
                // Retrieve and remove the top node from the stack.
                const BVH* current = stack_arr[top]; // top() from stack
                --top; // pop()

                // Check if the ray intersects the bounding box of the current node.
                // If not, skip further processing for this node.
                if (!current->bbox.hit(r, ray_t)) {
                    continue;
                }

                // If the current node has a left child node, add it to the stack for further processing.
                if (current->leftNode) {
                    stack_arr[++top] = current->leftNode;   // push() to stack
                }

                // If the current node has a right child node, add it to the stack for further processing.
                if (current->rightNode) {
                    stack_arr[++top] = current->rightNode;  // push() into stack
                }

                // Perform a hit test on the left child if it is a leaf node (i.e., it contains an actual object).
                if (current->left && current->left->sphere.hit(r, ray_t, temp_rec)) {
                    // A hit was found; update the hit_left flag and the closest hit record.
                    hit_left = true;
                    // Update the maximum boundary of the ray interval to the hit point,
                    // ensuring that any subsequent hits are closer than the current hit.
                    ray_t.max = temp_rec.t;
                    // Update the closest hit record with the details of the new closest hit.
                    rec = temp_rec;
                }

                // Perform a hit test on the right child if it is a leaf node,
                // using an updated interval that accounts for any hit found on the left.
                if (current->right && current->right->sphere.hit(r, interval(ray_t.min, hit_left ? temp_rec.t : ray_t.max), temp_rec)) {
                    // A hit was found; update the hit_right flag and the closest hit record.
                    hit_right = true;
                    // Update the closest hit record with the details of the new closest hit.
                    rec = temp_rec;
                    // Update the maximum boundary of the ray interval to this new hit point.
                    ray_t.max = rec.t;
                }
            }

            // Return true if a hit was detected on either the left or the right,
            // indicating that the ray intersects at least one object within the BVH.
            return hit_left || hit_right;
        }
        

        
        AaBb bounding_box() { return bbox; }

    private:
        AaBb bbox;
        Type type;
        hittable* left = NULL;
        hittable* right = NULL;
        BVH* leftNode = NULL;
        BVH* rightNode = NULL;
       
        static bool box_compare(const hittable &a, const hittable &b, int axis_index) {
            auto a_axis_interval = a.sphere.bounding_box().axis_interval(axis_index);
            auto b_axis_interval = b.sphere.bounding_box().axis_interval(axis_index);
            return a_axis_interval.min < b_axis_interval.min;
        }

        static bool box_x_compare(const hittable &a, const hittable &b) {
            return box_compare(a, b, 0);
        }

        static bool box_y_compare(const hittable &a, const hittable &b) {
            return box_compare(a, b, 1);
        }

        static bool box_z_compare(const hittable &a, const hittable &b) {
            return box_compare(a, b, 2);
        }



};
