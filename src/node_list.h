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