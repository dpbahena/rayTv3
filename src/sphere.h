
// #include "aabb.h"
#pragma once

#include "hittable.h"



__device__
bool hit_sphere(const ray& r, interval ray_t, const sphere_data& sphere, hit_record& rec) {
    glm::vec3 current_center = sphere.center.at(r.time());
    glm::vec3 oc = current_center - r.origin;
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

static AaBb sphere_bounding_box(const sphere_data& sphere) { return *sphere.bbox;}


/* Global or static hit function that processes hits based on the type of object */       
__device__
static bool object_hit(const ray& r, interval ray_t, const hittable& obj, hit_record& rec){
    
    switch(obj.type) {
        case Type::SPHERE:
            return hit_sphere(r, ray_t, obj.sphere, rec);
        
        // handle other types ....
        default:
            return false;
    }
}
