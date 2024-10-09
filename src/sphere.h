
// #include "aabb.h"
#pragma once

#include "hittable.h"




bool hit_sphere(const ray& r, interval ray_t, const sphere_data& sphere, hit_record& rec) {
    glm::vec3 current_center = sphere.center.at(r.time());
    glm::vec3 oc = r.origin - current_center;
    auto a = glm::dot(r.direction, r.direction);
    auto h = glm::dot(r.direction, oc);
    auto c = glm::dot(oc, oc) - sphere.radius * sphere.radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0) return false;   // no Real solution
    auto sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {  // if outside limits  then try other root
        root = (h + sqrtd) / a;
        if (!ray_t.surrounds(root))  return false;  // if still outside limits then not a hit .. return false
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    glm::vec3 outward_normal = (rec.p - current_center) / sphere.radius;
    rec.set_face_normal(r, outward_normal);
    rec.type = Type::SPHERE;
    rec.mat = sphere.mat;
   

    return true;
    
}

/* Global or static hit function that processes hits based on the type of object */       
static bool object_hit(const ray& r, interval ray_t, const hittable& obj, hit_record& rec){
    
    switch(obj.type) {
        case Type::SPHERE:
            return hit_sphere(r, ray_t, obj.sphere, rec);
        
        // handle other types ....
        default:
            return false;
    }
}




// class sphere {
//         public:
//             __host__ __device__
//             sphere() {}
//             // stationary sphere
//             __host__ __device__
//             sphere(const glm::glm::vec3& static_center, float radius, material* mat) : center(static_center, glm::glm::vec3(0.0,0.0,0.0)), radius(radius), mat(mat){
//                 auto rvec = glm::glm::vec3(radius, radius, radius);
//                 bbox = AaBb(static_center - rvec, static_center + rvec);
//             }
            
//             // moving sphere
//             __host__ __device__
//             sphere(const glm::glm::vec3& center1, const glm::glm::vec3& center2, float radius, material* mat) : center(center1, center2 - center1), radius(radius), mat(mat){
//                 auto rvec = glm::glm::vec3(radius, radius, radius);
//                 AaBb box1(center.at(0) - rvec, center.at(0) + rvec);
//                 AaBb box2(center.at(1) - rvec, center.at(1) + rvec);
//                 bbox = AaBb(box1, box2);
//             }

//             __host__ __device__
//             bool hit(const ray& r, interval ray_t, hitRecord &rec) const  {
//                 glm::glm::vec3 current_center = center.at(r.time()); 
//                 glm::glm::vec3 oc =  r.origin - current_center;
//                 auto a = glm::glm::dot(r.direction, r.direction);
//                 auto h = glm::glm::dot(r.direction, oc);
//                 auto c = glm::glm::dot(oc, oc) - radius * radius;

//                 auto discriminant = h * h - a * c;
//                 if (discriminant < 0)
//                     return false;

//                 auto sqrtd = std::sqrt(discriminant);

//                 // find the nearest root that lies in the acceptable range
//                 auto root = (-h - sqrtd) / a;
//                 if (!ray_t.surrounds(root)) {
//                     root = (-h + sqrtd) / a;
//                     if (!ray_t.surrounds(root))
//                         return false;
//                 }

//                 rec.t = root;
//                 rec.p = r.at(rec.t);
//                 glm::glm::vec3 outward_normal = (rec.p - current_center) / radius;
//                 rec.set_face_normal(r, outward_normal);
//                 rec.mat_ptr = mat;
                
                
//                 return true;
//             }

//             AaBb bounding_box() const { return bbox; }

//         private:
//             // glm::glm::vec3 center1;
//             ray center;
//             float radius;
//             material* mat;
//             AaBb bbox;
//             // bool is_moving;
//             // glm::glm::vec3 center_vec;

//             // __device__ __host__
//             // glm::glm::vec3 sphere_center(float t) const {
//             //     /* Linearly interpolate from center1 to center2 according to time, where t = 0 yields center1 + time * center_vec */
//             //     return center1 + t * center_vec;
//             // }
//     };