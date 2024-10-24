#pragma once

// #include "aabb.h"
// #include "sphere.h"
#include "hittable_list.h"
#include <algorithm>
#include "sphere.h"
#include "hittable.h"

inline int random_int(int min, int max);


// class BVH {
//     public:
//         BVH(const hittable_list& objects) {
//             build_bvh(objects);
//         }
//         // bool hit(const ray& r, interval ray_t, hit_record& rec, const hittable_list& objects) const {
//         //     return hit_bvh(r, ray_t, rec, objects);
//         // }

//         AaBb bounding_box() const {
//             return nodes[0].bbox;  // Root node's bounding box
//         }



//     private:
//     std::vector<BVHNode> nodes;

//     void build_bvh(const hittable_list& objects) {
//         nodes.clear();

//         std::vector<BVHNode> node_stack;
//         node_stack.reserve(objects.objects_size);

//         /* Build the BVH with the objects from the hittable_list */
//         recursive_build(objects, 0, objects.objects_size, node_stack);

//         /* Store the resulting BVH in a flattened array */
        
//         nodes = node_stack;

//     }

//     void recursive_build(const hittable_list& objects, size_t start, size_t end, std::vector<BVHNode>& node_stack) {
//         size_t object_span = end - start;
//         BVHNode node;

//         if(object_span == 1) {
//             node.is_leaf = true;
//             node.object_index = start; // Directly use index as object reference
//             node.left_child_index = -1;
//             node.right_child_index = -1;
//         } else {

//             /* Internal node case: sort the objects by a random axis */
//             int axis = int(random_double(0, 2)); // choose an axis to split
//             auto comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

//             /* Sort the objects along the chosen axis */
            
//             std::sort(objects.list + start, objects.list + end, comparator);
            
           

//             /* Divide the objects into 2 groups and recursively build left and right subtrees */
//             auto mid = start + object_span / 2;

//             recursive_build(objects, start, mid, node_stack);  // build left subtree
//             recursive_build(objects, mid, end, node_stack);     // build right subtree

//             /* Set left and right child indices */
//             node.left_child_index = node_stack.size() - 2;
//             node.right_child_index = node_stack.size() - 1;
            
//             node.is_leaf = false;

//             /* Conpute bounding box for interneal node by combining left and right child boxes */
//             node.bbox = AaBb(node_stack[node.left_child_index].bbox, node_stack[node.right_child_index].bbox);
//         }

//         /* Add the node to the stack (flat BVH array )*/
//         node_stack.push_back(node);

//     }
    
//     // bool hit_bvh(const ray& r, interval ray_t, hitRecord& rec, const hittable_list& objects) const {

//     //     bool hit_anything = false;
//     //     hitRecord temp_rec;
//     //     int current_node_index = 0;  // start from the root node

//     //     while (current_node_index != -1) {
//     //         const BVHNode& node = nodes[current_node_index];
            
//     //         if (!node.bbox.hit(r, ray_t)) {
//     //             break ; // Skip if the bounding box is not hit
//     //         }
//     //         if (node.is_leaf) {
//     //             // check for object hit using hittable_list passed as a parameter 
//     //             if (objects.list[node.object_index].hit(r, ray_t, temp_rec)){
//     //                 hit_anything = true;
//     //                 ray_t.max = temp_rec.t;  // Update interval for closer hit
//     //                 rec = temp_rec;
//     //             }
//     //             break;
//     //         } else {
//     //             /* Internal node, check children */
//     //             bool hit_left =  nodes[node.left_child_index].bbox.hit(r, ray_t);
//     //             bool hit_right = nodes[node.right_child_index].bbox.hit(r, ray_t);

//     //             if (hit_left && hit_right) {
//     //                 current_node_index = node.left_child_index; // Go to the left first

//     //             } else if (hit_left) {
//     //                 current_node_index = node.left_child_index;
//     //             } else if (hit_right) {
//     //                 current_node_index = node.right_child_index;
//     //             } else {
//     //                 break;  // Neither child is hit, end transversal
//     //             }
//     //         }
//     //     }

//     //     return hit_anything;
//     // }



//     // static bool box_compare(const BVHNode* a, const BVHNode* b, int axis_index) {
//     static bool box_compare(const hittable& a, const hittable& b, int axis_index) {
        
//         /* Compare the bounding boxes of two objects along the specific axis */
//         // a.sphere.bbox.axis_interval(
//         // auto a_axis_interval = a->bbox.axis_interval(axis_index);
//         // auto b_axis_interval = b->bbox.axis_interval(axis_index);
//         auto a_axis_interval = a.sphere.bbox->axis_interval(axis_index);
//         auto b_axis_interval = b.sphere.bbox->axis_interval(axis_index);

//         return a_axis_interval.min < b_axis_interval.min; 
//     }

//     // static bool box_x_compare ( BVHNode* a,  BVHNode* b){
//     static bool box_x_compare (const hittable& a, const hittable& b){
//         return box_compare(a, b, 0);
//     }
//     // static bool box_y_compare ( BVHNode* a,  BVHNode* b){
//     static bool box_y_compare (const hittable& a, const hittable& b){
//         return box_compare(a, b, 1);
//     }
//     // static bool box_z_compare ( BVHNode* a,  BVHNode* b){
//     static bool box_z_compare (const hittable& a, const hittable& b){
//         return box_compare(a, b, 2);
//     }

// };



// working here


// const int  MAX_DEPTH = 20;

// class BVH {
//     public:
//         BVH(hittable_list list, std::vector<BVH*> &allocated_nodes) : BVH(list.list, (size_t)0, list.objects_size, allocated_nodes) {}
        
//         BVH(hittable* objects, size_t start, size_t end, std::vector<BVH*> &allocated_nodes) : type(Type::BBOX)
//         {
//             int axis = random_int(0,2);

//             // build the bounding box of the span of source objects.
//             // bbox = aabb::empty;
//             // for(size_t object_index = start; object_index < end; object_index++){
//             //     bbox = aabb(bbox, objects[object_index]->sphere.bounding_box());
//             // }
//             // int axis = bbox.longest_axis();            


//             auto comparator = (axis == 0) ? box_x_compare
//                             : (axis == 1) ? box_y_compare
//                                         : box_z_compare;

//             size_t object_span = end - start;
//             // printf("Start: %d, end: %d\n", (int)start, (int)end);
//             if (object_span == 1) {
//                 left = right = objects[start];
//             } else if (object_span == 2) {
//                 left = &objects[start];
//                 right = &objects[start+1];
//             } else {
//                 std::sort(objects + start, objects + end, comparator);
                
//                 auto mid = start + object_span/2;
//                 leftNode = new BVH(objects, start, mid, allocated_nodes);
//                 allocated_nodes.push_back(leftNode);
//                 // ln = true;
//                 rightNode = new BVH(objects, mid, end, allocated_nodes);
//                 allocated_nodes.push_back(rightNode);
//                 // rn = true;

//             }
            
            
//             if (object_span > 2){
//                 bbox = AaBb(leftNode->bounding_box(), rightNode->bounding_box() );
//             } else
//                 bbox = AaBb(*left->sphere.bbox, *right->sphere.bbox);
//         }
      
//         // NON RECURSIVE - Define the hit function which determines if a ray intersects any object within the BVH structure.
//         __device__
//         bool hit(const ray& r, interval ray_t, hit_record& rec)  {
//             // Initialize a stack to manage the BVH nodes to be processed.
//             const BVH* stack[MAX_DEPTH];
//             int stackIndex = 0;  // stack pointer
//             // Start the traversal with the root node of the BVH.
//             stack[stackIndex++] = this;

//             // Initialize flags to track hits on the left and right children.
//             bool hit_left = false;
//             bool hit_right = false;
//             // Initialize a temporary record to store the closest hit found during the traversal.
//             hit_record temp_rec;

//             // Loop until there are no more nodes to process in the stack.
//             while (stackIndex > 0) {
//                 // Retrieve and remove the top node from the stack.
//                 const BVH* current = stack[--stackIndex]; // pop the top node
                

//                 // Check if the ray intersects the bounding box of the current node.
//                 // If not, skip further processing for this node.
//                 if (!current->bbox.hit(r, ray_t)) {
//                     continue;
//                 }

//                 // If the current node has a left child node, add it to the stack for further processing.
//                 if (current->leftNode) {
//                     stack[stackIndex++] = current->leftNode;  // push left child
//                 }

//                 // If the current node has a right child node, add it to the stack for further processing.
//                 if (current->rightNode) {
//                     stack[stackIndex++] = current->rightNode;  // push right child
//                 }

//                 // Perform a hit test on the left child if it is a leaf node (i.e., it contains an actual object).
                
//                 if (current->left && hit_sphere(r, ray_t, current->left->sphere, temp_rec)) {
//                     // A hit was found; update the hit_left flag and the closest hit record.
//                     hit_left = true;
//                     // Update the maximum boundary of the ray interval to the hit point,
//                     // ensuring that any subsequent hits are closer than the current hit.
//                     ray_t.max = temp_rec.t;
//                     // Update the closest hit record with the details of the new closest hit.
//                     rec = temp_rec;
//                 }

//                 // Perform a hit test on the right child if it is a leaf node,
//                 // using an updated interval that accounts for any hit found on the left.
//                 if (current->right && hit_sphere(r, interval(ray_t.min, hit_left ? temp_rec.t : ray_t.max), current->right->sphere, temp_rec)) {
//                     // A hit was found; update the hit_right flag and the closest hit record.
//                     hit_right = true;
//                     // Update the closest hit record with the details of the new closest hit.
//                     rec = temp_rec;
//                     // Update the maximum boundary of the ray interval to this new hit point.
//                     ray_t.max = rec.t;
//                 }
//             }

//             // Return true if a hit was detected on either the left or the right,
//             // indicating that the ray intersects at least one object within the BVH.
//             return hit_left || hit_right;
//         }

        
//         AaBb bounding_box() { return bbox; }

//     private:
//         AaBb bbox;
//         Type type;
//         hittable* left = NULL;
//         hittable* right = NULL;
//         BVH* leftNode = NULL;
//         BVH* rightNode = NULL;

//         static bool box_compare(  hittable* a,   hittable* b, int axis_index) {
           
//             auto a_axis_interval = a->sphere.bbox->axis_interval(axis_index);
//             auto b_axis_interval = b->sphere.bbox->axis_interval(axis_index);
           
//             return a_axis_interval.min < b_axis_interval.min;
//         }

//         static bool box_x_compare ( hittable* a,  hittable* b) {
//             return box_compare(a, b, 0);
//         }

//         static bool box_y_compare (  hittable* a,   hittable* b) {
//             return box_compare(a, b, 1);
//         }

//         static bool box_z_compare (  hittable* a,  hittable* b) {
//             return box_compare(a, b, 2);
//         }

// };

