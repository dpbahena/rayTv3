
// #include "aabb.h"
#pragma once

#include "hittable.h"




__device__ __host__
bool sphere_data::hit(const ray& r, interval ray_t, hit_record& rec)  const {

    glm::vec3 current_center = center.at(r.time());
    glm::vec3 oc = current_center - r.origin;
    auto a = glm::dot(r.direction, r.direction);
    auto h = glm::dot(r.direction, oc);
    auto c = glm::dot(oc, oc) - radius * radius;

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
    glm::vec3 outward_normal = (rec.p - current_center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.type = Type::SPHERE;
    rec.mat = mat;
   

    return true;
    
}

