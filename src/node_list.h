#pragma once

#include "bvh_node.h"



struct node_list {
    
    BVH* objects;
    int size_of_objects = 1;
    AaBb bbox;
    node_list() {}
    

    void add(BVH* object) {
        
        objects = object; 
        // bbox = AaBb(bbox, object->bounding_box());
    }
     
    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;
            
            if(objects->hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
            return hit_anything;
        }
    AaBb bounding_box() {return bbox;}

};



struct flat_node_list {
    
    
    BVH2* tree;
    BVHNode** nodes;

    BVHNode* cudaNodes;
    hittable* hittables;
    
    flat_node_list(){}
    void add(BVH2* tree){
        this->tree = tree;
        nodes = tree->nodes.data();
        
        
        // for (auto& n : tree->nodes){
        //     printf("Left child:  %d\n", n->left_child_index);
        //     printf("rigth Child:  %d\n", n->right_child_index);
        //     printf("object index: %d\n", n->object_index);
        //     if (n->is_leaf)
        //         printf("LEAF\n");
        //     else
        //         printf("NODE\n");
        // }
    }

    void addCudaNode(BVHNode* nodes, hittable* hittables){
        cudaNodes = nodes;
        this->hittables = hittables;
    }

    


    __device__ __host__
    bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;
        
        // if(tree->hit(r, interval(ray_t.min, closest_so_far), temp_rec, nodes, tree->hittables)){
        if(BVH2::hit2(r, interval(ray_t.min, closest_so_far), temp_rec, cudaNodes, hittables)){
            
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
        

        return hit_anything;
    }


    AaBb bbox;
    AaBb bounding_box() {return bbox;}

};
