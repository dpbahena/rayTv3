#pragma once


#include "hittable_list.h"
#include "sphere.h"
#include "hittable.h"
#include <algorithm>

inline int random_int(int min, int max);

struct BVHNode {
    AaBb bbox;
    int left_child_index;     // Index of left child in the BVH array (-1 if it's a leaf)
    int right_child_index;    // Index of right child in the BVH array (-1 if it's a leaf)
    int rope_index;
    bool is_leaf;             // Is this node a leaf?
    size_t start;
    size_t end;
    int object_index;         // Index of the object (used if it's a leaf)
    
};


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
            bbox = AaBb::empty();
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
            const int MAX = 15;
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



class BVH2 {
    public:
        std::vector<BVHNode*> nodes;
        hittable* hittables;
        size_t hittables_size;
        
        struct StackNode {
            size_t start, end;
            int parentIndex;
            bool isLeftChild;
        };
        
        
        BVH2(hittable_list list, std::vector<BVHNode*> nodes) :  BVH2(list.hittables, list.objects_size, nodes)  {}
            
        BVH2(hittable* &objects, size_t size, std::vector<BVHNode*> nodes) : nodes(nodes), hittables(objects), hittables_size(size)
        {
            
            build_bvh_NR();
        }

        // NON RECURSIVE - Define the hit function which determines if a ray intersects any object within the BVH structure.
        __device__ __host__
        bool hit(const ray& r, interval ray_t, hit_record& rec, BVHNode** nodes, /* const std::vector<BVHNode*> &nodes, */ hittable* &hittables) {
            
            const int MAX = 15;
            BVHNode* node_stack_arr[MAX];  // created on stack 
            int top = -1;        
          
            node_stack_arr[++top] = *(nodes + 0);
            
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
                    node_stack_arr[++top] = *(nodes + current->left_child_index);
                }
                // If the current node has a right child node, add it to the stack for further processing.
                if (current->right_child_index != -1 ) {
                    node_stack_arr[++top] = *(nodes + current->right_child_index);
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

        __device__ __host__
        static bool hit2(const ray& r, interval ray_t, hit_record& rec, BVHNode* nodes, hittable* hittables) {
            
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





        AaBb bounding_box() { return bbox; }

    private:

        AaBb bbox;

        // SUGGESTED OPTIMIZATION FROM CHATGPT
        void build_bvh_NR() {
            nodes.clear();

            std::vector<BVHNode*> node_stack;
          
            const int MAX = 1024;
            StackNode traversalStack[MAX];
            int top = -1 ;  // initialize stack
            
            traversalStack[++top] ={0, hittables_size, -1, true};  // push()

            // while (top >= 0) {
            for (;top >= 0;) {
               
                StackNode current = traversalStack[top];
                --top; // pop()

                size_t object_span = current.end - current.start;

                // **Compute the bounding box of the current node upfront**
                AaBb bbox = AaBb::empty();
                for (size_t i = current.start; i < current.end; ++i) {
                    bbox = AaBb(bbox, (hittables + i)->sphere.bounding_box());
                }

                if (object_span == 1) {
                    // **Create a leaf node**
                    BVHNode node;
                    node.is_leaf = true;
                    node.object_index = current.start;
                    node.left_child_index = -1;
                    node.right_child_index = -1;
                    node.bbox = bbox;  // Bounding box of the single object
                    node_stack.push_back(new BVHNode(node));
                    int node_index = int(node_stack.size() - 1);

                    // **Update parent's child index**
                    if (current.parentIndex != -1) {
                        auto& parent_node = node_stack[current.parentIndex];
                        if (current.isLeftChild) {
                            parent_node->left_child_index = node_index;
                        } else {
                            parent_node->right_child_index = node_index;
                        }
                    }

                } else if (object_span == 2) {
                    // **Create leaf nodes for both objects**
                    // Left leaf node
                    BVHNode left_node;
                    left_node.is_leaf = true;
                    left_node.object_index = current.start;
                    left_node.left_child_index = -1;
                    left_node.right_child_index = -1;
                    // left_node.bbox = (hittables + left_node.object_index)->sphere.bounding_box();
                    node_stack.push_back(new BVHNode(left_node));
                    int left_node_index = int(node_stack.size() - 1);

                    // Right leaf node
                    BVHNode right_node;
                    right_node.is_leaf = true;
                    right_node.object_index = current.start + 1;
                    right_node.left_child_index = -1;
                    right_node.right_child_index = -1;
                    // right_node.bbox = (hittables + right_node.object_index)->sphere.bounding_box();
                    node_stack.push_back(new BVHNode(right_node));
                    int right_node_index = int(node_stack.size() - 1);

                    // **Create parent node with bounding box**
                    BVHNode parent_node;
                    parent_node.is_leaf = false;
                    parent_node.left_child_index = left_node_index;
                    parent_node.right_child_index = right_node_index;
                    parent_node.bbox = bbox;  // Bounding box of the two objects
                    node_stack.push_back(new BVHNode(parent_node));
                    int parent_index = int(node_stack.size() - 1);

                    // **Update parent's child index**
                    if (current.parentIndex != -1) {
                        auto& grandparent_node = node_stack[current.parentIndex];
                        if (current.isLeftChild) {
                            grandparent_node->left_child_index = parent_index;
                        } else {
                            grandparent_node->right_child_index = parent_index;
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
                    std::sort(hittables + current.start, hittables + current.end, comparator);

                    // **Split the objects into two halves**
                    size_t mid = current.start + object_span / 2;

                    // **Create an internal node**
                    BVHNode node;
                    node.is_leaf = false;
                    node.left_child_index = -1;  // Will be set after processing children
                    node.right_child_index = -1; // Will be set after processing children
                    node.bbox = bbox;  // Bounding box of all objects in the span
                    node_stack.push_back(new BVHNode(node));
                    int node_index = int(node_stack.size() - 1);

                    // **Update parent's child index**
                    if (current.parentIndex != -1) {
                        auto& parent_node = node_stack[current.parentIndex];
                        if (current.isLeftChild) {
                            parent_node->left_child_index = node_index;
                        } else {
                            parent_node->right_child_index = node_index;
                        }
                    }

                    // **Push child nodes onto the stack for further processing**
                    traversalStack[++top] = {mid, current.end, node_index, false};        // Right child
                    traversalStack[++top] = {current.start, mid, node_index, true};       // Left child
                }
            }

            // **Assign the built nodes to the class member**
            if (!node_stack.empty()) {
                nodes = node_stack;
            }
        }



        static bool box_compare(const hittable& a, const hittable& b, int axis_index) {
           
            auto a_axis_interval = a.sphere.bounding_box().axis_interval(axis_index);
            auto b_axis_interval = b.sphere.bounding_box().axis_interval(axis_index);
           
            return a_axis_interval.min < b_axis_interval.min;
        }

        static bool box_x_compare (const hittable& a, const hittable& b) {
            return box_compare(a, b, 0);
        }

        static bool box_y_compare (const hittable& a, const hittable& b) {
            return box_compare(a, b, 1);
        }

        static bool box_z_compare (const hittable& a, const hittable& b) {
            return box_compare(a, b, 2);
        }
       
        
// RANDOM AXIS
        // void build_bvh_NR() {
        //     nodes.clear();

        //     // Enum to track the processing state of a node
        //     enum class NodeState { Unprocessed, Processing };

        //     struct StackNode {
        //         size_t start, end;
        //         int parentIndex;
        //         bool isLeftChild;
        //         NodeState state;
        //         int nodeIndex;
        //     };

        //     std::vector<std::shared_ptr<BVHNode>> node_stack;
        //     std::stack<StackNode> traversalStack;
        //     traversalStack.push({0, hittables.size(), -1, true, NodeState::Unprocessed, -1});

        //     while (!traversalStack.empty()) {
        //         StackNode current = traversalStack.top();
        //         traversalStack.pop();

        //         size_t object_span = current.end - current.start;

        //         if (current.state == NodeState::Unprocessed) {
        //             if (object_span == 1 || object_span == 2) {
        //                 // For leaf nodes or small spans, process immediately
        //                 int node_index = -1;

        //                 if (object_span == 1) {
        //                     // Create a leaf node
        //                     BVHNode node;
        //                     node.is_leaf = true;
        //                     node.object_index = current.start;
        //                     node.left_child_index = -1;
        //                     node.right_child_index = -1;
        //                     node.bbox = hittables[node.object_index]->sphere.bounding_box();
        //                     node_stack.push_back(std::make_shared<BVHNode>(node));
        //                     node_index = int(node_stack.size() - 1);

        //                 } else { // object_span == 2
        //                     // Create leaf nodes for both objects
        //                     // Left leaf
        //                     BVHNode left_node;
        //                     left_node.is_leaf = true;
        //                     left_node.object_index = current.start;
        //                     left_node.left_child_index = -1;
        //                     left_node.right_child_index = -1;
        //                     left_node.bbox = hittables[left_node.object_index]->sphere.bounding_box();
        //                     node_stack.push_back(std::make_shared<BVHNode>(left_node));
        //                     int left_node_index = int(node_stack.size() - 1);

        //                     // Right leaf
        //                     BVHNode right_node;
        //                     right_node.is_leaf = true;
        //                     right_node.object_index = current.start + 1;
        //                     right_node.left_child_index = -1;
        //                     right_node.right_child_index = -1;
        //                     right_node.bbox = hittables[right_node.object_index]->sphere.bounding_box();
        //                     node_stack.push_back(std::make_shared<BVHNode>(right_node));
        //                     int right_node_index = int(node_stack.size() - 1);

        //                     // Create parent node
        //                     BVHNode parent_node;
        //                     parent_node.is_leaf = false;
        //                     parent_node.left_child_index = left_node_index;
        //                     parent_node.right_child_index = right_node_index;
        //                     parent_node.bbox = aabb(
        //                         node_stack[left_node_index]->bbox,
        //                         node_stack[right_node_index]->bbox
        //                     );
        //                     node_stack.push_back(std::make_shared<BVHNode>(parent_node));
        //                     node_index = int(node_stack.size() - 1);
        //                 }

        //                 // Update parent node's child index and bounding box
        //                 if (current.parentIndex != -1) {
        //                     auto& parent_node = node_stack[current.parentIndex];
        //                     if (current.isLeftChild) {
        //                         parent_node->left_child_index = node_index;
        //                     } else {
        //                         parent_node->right_child_index = node_index;
        //                     }

        //                     // If both child indices are set, compute the parent's bounding box
        //                     if (parent_node->left_child_index != -1 && parent_node->right_child_index != -1) {
        //                         parent_node->bbox = aabb(
        //                             node_stack[parent_node->left_child_index]->bbox,
        //                             node_stack[parent_node->right_child_index]->bbox
        //                         );
        //                     }
        //                 }

        //             } else {
        //                 // object_span > 2
        //                 // Sort objects along a randomly chosen axis
        //                 int axis = random_int(0, 2);
        //                 auto comparator = (axis == 0) ? box_x_compare
        //                                 : (axis == 1) ? box_y_compare
        //                                             : box_z_compare;
        //                 std::sort(std::begin(hittables) + current.start, std::begin(hittables) + current.end, comparator);

        //                 auto mid = current.start + object_span / 2;

        //                 // Create an internal node but defer bbox computation
        //                 BVHNode node;
        //                 node.is_leaf = false;
        //                 node.left_child_index = -1;
        //                 node.right_child_index = -1;
        //                 node_stack.push_back(std::make_shared<BVHNode>(node));
        //                 int node_index = int(node_stack.size() - 1);

        //                 // Push the current node back onto the stack with state Processing
        //                 current.state = NodeState::Processing;
        //                 current.nodeIndex = node_index;
        //                 traversalStack.push(current);

        //                 // Push right and left children onto the stack
        //                 traversalStack.push({mid, current.end, node_index, false, NodeState::Unprocessed, -1});
        //                 traversalStack.push({current.start, mid, node_index, true, NodeState::Unprocessed, -1});
        //             }

        //         } else if (current.state == NodeState::Processing) {
        //             // Both children have been processed; compute bbox
        //             int node_index = current.nodeIndex;
        //             auto& node = node_stack[node_index];

        //             auto& left_child = node_stack[node->left_child_index];
        //             auto& right_child = node_stack[node->right_child_index];
        //             node->bbox = aabb(left_child->bbox, right_child->bbox);

        //             // Update parent node's child index
        //             if (current.parentIndex != -1) {
        //                 auto& parent_node = node_stack[current.parentIndex];
        //                 if (current.isLeftChild) {
        //                     parent_node->left_child_index = node_index;
        //                 } else {
        //                     parent_node->right_child_index = node_index;
        //                 }

        //                 // If both child indices are set, compute the parent's bounding box
        //                 if (parent_node->left_child_index != -1 && parent_node->right_child_index != -1) {
        //                     parent_node->bbox = aabb(
        //                         node_stack[parent_node->left_child_index]->bbox,
        //                         node_stack[parent_node->right_child_index]->bbox
        //                     );
        //                 }
        //             }
        //         }
        //     }

        //     // After traversal, the root node is the last node processed
        //     if (!node_stack.empty()) {
        //         nodes = node_stack;
        //     }
        // }

        // LARGEST AXIS "OPTIMIZED"

        // void build_bvh_NR() {
        //     nodes.clear();

        //     // Enum to track the processing state of a node
        //     enum class NodeState { Unprocessed, Processing };

        //     struct StackNode {
        //         size_t start, end;
        //         int parentIndex;
        //         bool isLeftChild;
        //         NodeState state;
        //         int nodeIndex;
        //     };

        //     std::vector<std::shared_ptr<BVHNode>> node_stack;
        //     std::stack<StackNode> traversalStack;
        //     traversalStack.push({0, hittables.size(), -1, true, NodeState::Unprocessed, -1});

        //     while (!traversalStack.empty()) {
        //         StackNode current = traversalStack.top();
        //         traversalStack.pop();

        //         size_t object_span = current.end - current.start;

        //         if (current.state == NodeState::Unprocessed) {
        //             if (object_span == 1 || object_span == 2) {
        //                 // For leaf nodes or small spans, process immediately
        //                 int node_index = -1;

        //                 if (object_span == 1) {
        //                     // Create a leaf node
        //                     BVHNode node;
        //                     node.is_leaf = true;
        //                     node.object_index = current.start;
        //                     node.left_child_index = -1;
        //                     node.right_child_index = -1;
        //                     node.bbox = hittables[node.object_index]->sphere.bounding_box();
        //                     node_stack.push_back(std::make_shared<BVHNode>(node));
        //                     node_index = int(node_stack.size() - 1);

        //                 } else { // object_span == 2
        //                     // Create leaf nodes for both objects
        //                     // Left leaf
        //                     BVHNode left_node;
        //                     left_node.is_leaf = true;
        //                     left_node.object_index = current.start;
        //                     left_node.left_child_index = -1;
        //                     left_node.right_child_index = -1;
        //                     left_node.bbox = hittables[left_node.object_index]->sphere.bounding_box();
        //                     node_stack.push_back(std::make_shared<BVHNode>(left_node));
        //                     int left_node_index = int(node_stack.size() - 1);

        //                     // Right leaf
        //                     BVHNode right_node;
        //                     right_node.is_leaf = true;
        //                     right_node.object_index = current.start + 1;
        //                     right_node.left_child_index = -1;
        //                     right_node.right_child_index = -1;
        //                     right_node.bbox = hittables[right_node.object_index]->sphere.bounding_box();
        //                     node_stack.push_back(std::make_shared<BVHNode>(right_node));
        //                     int right_node_index = int(node_stack.size() - 1);

        //                     // Create parent node
        //                     BVHNode parent_node;
        //                     parent_node.is_leaf = false;
        //                     parent_node.left_child_index = left_node_index;
        //                     parent_node.right_child_index = right_node_index;
        //                     parent_node.bbox = aabb(
        //                         node_stack[left_node_index]->bbox,
        //                         node_stack[right_node_index]->bbox
        //                     );
        //                     node_stack.push_back(std::make_shared<BVHNode>(parent_node));
        //                     node_index = int(node_stack.size() - 1);
        //                 }

        //                 // Update parent node's child index and bounding box
        //                 if (current.parentIndex != -1) {
        //                     auto& parent_node = node_stack[current.parentIndex];
        //                     if (current.isLeftChild) {
        //                         parent_node->left_child_index = node_index;
        //                     } else {
        //                         parent_node->right_child_index = node_index;
        //                     }

        //                     // If both child indices are set, compute the parent's bounding box
        //                     if (parent_node->left_child_index != -1 && parent_node->right_child_index != -1) {
        //                         parent_node->bbox = aabb(
        //                             node_stack[parent_node->left_child_index]->bbox,
        //                             node_stack[parent_node->right_child_index]->bbox
        //                         );
        //                     }
        //                 }

        //             } else {
        //                 // Build the bounding box of the span of source objects.
        //                 bbox = aabb::empty;
        //                 for (size_t object_index = current.start; object_index < current.end; object_index++)
        //                     bbox = aabb(bbox, hittables[object_index]->sphere.bounding_box());
        //                 // Split the longest axis of the enclosing bounding box to get the most subdivision
        //                 int axis = bbox.longest_axis();
                        
        //                 auto comparator = (axis == 0) ? box_x_compare
        //                                 : (axis == 1) ? box_y_compare
        //                                             : box_z_compare;
        //                 std::sort(std::begin(hittables) + current.start, std::begin(hittables) + current.end, comparator);

        //                 auto mid = current.start + object_span / 2;

        //                 // Create an internal node but defer bbox computation
        //                 BVHNode node;
        //                 node.is_leaf = false;
        //                 node.left_child_index = -1;
        //                 node.right_child_index = -1;
        //                 node_stack.push_back(std::make_shared<BVHNode>(node));
        //                 int node_index = int(node_stack.size() - 1);

        //                 // Push the current node back onto the stack with state Processing
        //                 current.state = NodeState::Processing;
        //                 current.nodeIndex = node_index;
        //                 traversalStack.push(current);

        //                 // Push right and left children onto the stack
        //                 traversalStack.push({mid, current.end, node_index, false, NodeState::Unprocessed, -1});
        //                 traversalStack.push({current.start, mid, node_index, true, NodeState::Unprocessed, -1});
        //             }

        //         } else if (current.state == NodeState::Processing) {
        //             // Both children have been processed; compute bbox
        //             int node_index = current.nodeIndex;
        //             auto& node = node_stack[node_index];

        //             auto& left_child = node_stack[node->left_child_index];
        //             auto& right_child = node_stack[node->right_child_index];
        //             node->bbox = aabb(left_child->bbox, right_child->bbox);

        //             // Update parent node's child index
        //             if (current.parentIndex != -1) {
        //                 auto& parent_node = node_stack[current.parentIndex];
        //                 if (current.isLeftChild) {
        //                     parent_node->left_child_index = node_index;
        //                 } else {
        //                     parent_node->right_child_index = node_index;
        //                 }

        //                 // If both child indices are set, compute the parent's bounding box
        //                 if (parent_node->left_child_index != -1 && parent_node->right_child_index != -1) {
        //                     parent_node->bbox = aabb(
        //                         node_stack[parent_node->left_child_index]->bbox,
        //                         node_stack[parent_node->right_child_index]->bbox
        //                     );
        //                 }
        //             }
        //         }
        //     }

        //     // After traversal, the root node is the last node processed
        //     if (!node_stack.empty()) {
        //         nodes = node_stack;
        //     }
        // }

        



};
