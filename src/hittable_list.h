#pragma once


#include "hittable.h"




__device__
static bool object_hit(const ray& r, interval ray_t, const hittable& obj, hit_record& rec);

struct hittable_list {
    public:
        // std::vector<hittable> objects;
        hittable* objects;
        int objects_size;
        // void clear(){objects.clear(); }

        // void add(const hittable& object) {
        //     objects.push_back(object);
        // }
        __device__
        bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;

            for (int i = 0; i < objects_size; i++) {
            // for (const auto& object : objects ){
                if(object_hit(r, interval(ray_t.min, closest_so_far), objects[i], temp_rec)){
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};
