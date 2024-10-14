#pragma once


#include "hittable.h"




__device__
static bool object_hit(const ray& r, interval ray_t, hit_record& rec, const hittable& obj);
__device__
static bool object2_hit(const ray& r, interval ray_t, hit_record& rec, const hittable& obj);


struct hittable_list {
    public:
        
        hittable* list;
        int objects_size;
        AaBb bbox;
        
        hittable_list() {};
        
        hittable_list(hittable* objects, int size, AaBb& bbox){
            list = objects;
            objects_size = size;
            bbox = bbox;
        }
        
        hittable_list(std::vector<hittable>& objects) {
            list = objects.data();
            objects_size = objects.size();
            // for(auto &obj: objects){
            //     bbox = AaBb(bbox, obj.sphere.bbox);
            // }
        }
        
        __device__
        bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;
            
            for (int i = 0; i < objects_size; i++) {
            // for (const auto& object : objects ){
                
                if(object2_hit(r, interval(ray_t.min, closest_so_far), temp_rec, list[i])){
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }

        AaBb bounding_box() const { return bbox;}
};
