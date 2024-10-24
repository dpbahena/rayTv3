#pragma once


#include "hittable.h"
#include "stdio.h"


struct hittable_list {
    public:
        hittable* hittables;
        

        // hittable* list;
        size_t objects_size;
        AaBb *bbox;
        // void clear(){objects.clear(); }

        hittable_list(){}         
        



        void add_bbox() {
            // bbox = hittables[0].sphere.bbox;
            // for(size_t i = 1; i < objects_size; i++)
            //     bbox = AaBb(bbox, hittables[i].sphere.bounding_box());
        }

        


        
        
        __device__
        bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;
            
            for (int i = 0; i < objects_size; i++){
                // printf("min: %f", hittables[i].sphere.bbox->axis_interval(1).min);
                if (hittables[i].sphere.hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
                    
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                    
                }
            }
            
            return hit_anything;
            }


        AaBb* bounding_box() const { return bbox; }    
};

        
