
// #include "aabb.h"
#pragma once

#include "hittable.h"
#include "texture.h"




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
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.type = Type::SPHERE;
    rec.mat = mat;
   

    return true;
    
}

__device__ __host__
void sphere_data::get_sphere_uv(const glm::vec3& p, float& u, float& v) const {
    /**
     * @p: a given point on the sphere of radius one, centered at the origin
     * @u: returned value [0,1] of angle around the Y axis from X = -1
     * @v: returned value [0,1] of angle form Y = -1 to Y= +1
     * @<1 0 0> yields <0.50 0.50>  <-1 0 0> yields <0.00 0.50> 
     * @<0 1 0> yields <0.50 1.00>  <0 -1 0> yields <0.50 0.00> 
     * @<0 0 1> yields <0.25 0.50>  <0 0 -1> yields <0.75 0.50> 
     * */ 

    float theta = acosf(-p.y);
    float phi = atan2f(-p.z, p.x + M_PI);
    u = phi / (2 * M_PI);
    v = theta / M_PI;
}

__device__ __host__
glm::vec3 checkerTexture_data::value(float u, float v, const glm::vec3& p) const {
    auto xInteger = int(floor(inv_scale * p.x));
    auto yInteger = int(floor(inv_scale * p.y));
    auto zInteger = int(floor(inv_scale * p.z));

    bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;
    // return isEven ? even->checkerTexture.value(u, v, p) : odd->checkerTexture.value(u, v, p);
    return isEven ? even->value(u, v, p) : odd->value(u, v, p);
}

/**
 * @return solid cyan if there is no texture 
 */
__device__ __host__
glm::vec3 imageTexture_data::value(float u, float v, const glm::vec3& p) const {
    if (image->height() <= 0 ) return glm::vec3(0.0f, 1.0f, 1.0f);
    //* Clamp input texture coordinates to [0,1] x [1,0]
    u = interval(0,1).clamp(u);
    v = 1.0 - interval(0,1).clamp(v);  //* Flip V to image coordinates

    auto i = int(u * image->width());
    auto j = int(v * image->height());
    auto pixel = image->pixel_data(i, j);
    auto color_scale = 1.0f / 255.0;
    return glm::vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);

}


