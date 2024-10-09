#pragma once

// #include "ray.h"
// #include "interval.h"
#include "hittable.h"





static bool object_hit(const ray& r, interval ray_t, const hittable& obj, hit_record& rec);

struct hittable_list {
    public:
        std::vector<hittable> objects;

        void clear(){objects.clear(); }

        void add(const hittable& object) {
            objects.push_back(object);
        }

        bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;


            for (const auto& object : objects ){
                if(object_hit(r, interval(ray_t.min, closest_so_far), object, temp_rec)){
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};
