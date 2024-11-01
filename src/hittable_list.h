#pragma once


#include "hittable.h"
#include "stdio.h"



struct hittable_list {
    public:
        hittable* hittables;
        
        // hittable* list;
        size_t objects_size;
        AaBb bbox;
        // void clear(){objects.clear(); }

        hittable_list(){}         
        
        __device__ __host__
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

        AaBb bounding_box() const { return bbox; }    
};








// struct d_hittable_list {
//     public:
        
//         hittable* device_objets;    // reserve only for device objects
//         size_t device_objects_size;
//         AaBb bbox;

//         d_hittable_list() {}
                       
//         __device__
//         bool hit(const ray& r, interval ray_t, hit_record& rec) const {
//             // printf("size : %d\n", (int)device_objects_size) ;
//             hit_record temp_rec;
//             bool hit_anything = false;
//             auto closest_so_far = ray_t.max;

            
//             // for (const auto& object : objects ){
//             for (size_t i = 0; i < device_objects_size; i++) {
//                 // switch(device_objets[i].type) {
//                 //     case Type::SPHERE:
//                         // printf("center: %f, %f, %f\n", device_objets[i].sphere.center.orig.x, device_objets[i].sphere.center.orig.y, device_objets[i].sphere.center.orig.z );
//                         if(device_objets[i].sphere.hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
//                             hit_anything = true;
//                             closest_so_far = temp_rec.t;
//                             rec = temp_rec;
                            
//                         }
//                         // break;
//                     // case Type::BOX:
//                     //     if(object->.hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
//                     //         hit_anything = true;
//                     //         closest_so_far = temp_rec.t;
//                     //         rec = temp_rec;
//                     //     }
//                     //     break;
//                     // default:
//                     //     hit_anything = false;
//                 // }

                
//             }

//             return hit_anything;
//         }

//         AaBb bounding_box() const { return bbox; }

//     // private:
        
// };        


        
