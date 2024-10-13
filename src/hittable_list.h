#pragma once


#include "hittable.h"




__device__
static bool object_hit(const ray& r, interval ray_t, const hittable& obj, hit_record& rec);

struct hittable_list {
    public:
        // std::vector<hittable> objects;
        hittable* list;
        int objects_size;
        AaBb bbox;
        // void clear(){objects.clear(); }

        // void add(const hittable& object) {
        //     objects.push_back(object);
        // }
        hittable_list() {};
        
        hittable_list(hittable* objects, int size){
            list = objects;
            objects_size = size;
            bbox = bbox;
            // bbox = list[0].sphere.bbox;
            // bbox = sphere_bounding_box(list[0].sphere); // get the initial bbox
            // for (int i = 1; i < size; i++) {
            //     bbox = AaBb(bbox, sphere_bounding_box(list[i].sphere) );
            // }
        }
        __device__
        bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;

            for (int i = 0; i < objects_size; i++) {
            // for (const auto& object : objects ){
                if(object_hit(r, interval(ray_t.min, closest_so_far), list[i], temp_rec)){
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }

        AaBb bounding_box() const { return bbox;}
};
