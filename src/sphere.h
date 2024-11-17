
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
    float phi = atan2f(-p.z, p.x) + M_PI;
    u = phi / (2 * M_PI);
    v = theta / M_PI;
}

__device__ __host__
bool quad_data::hit(const ray& r, interval ray_t, hit_record& rec)  const {
    auto denom = glm::dot(normal, r.direction);

    //* No hit if the ray is parallel to the plane
    if (fabsf(denom) < 1e-8 ) return false;
    //* Return false if th ehit point parameter t is outside the ray interval
    auto t = (D - glm::dot(normal, r.origin)) / denom;
    if (!ray_t.contains(t)) return false;

    //* Determine if the hit point lies within the planar shape using its plane coordinates
    auto intersection = r.at(t);
    glm::vec3 planar_hitpt_vector = intersection - Q;
    auto alpha = glm::dot(w, glm::cross(planar_hitpt_vector, v));
    auto beta = glm::dot(w, glm::cross(u, planar_hitpt_vector));
    if (!is_interior(alpha, beta, rec)) return false;
  
    //* Ray hits the 2D shape, set the rest of the hit record and return true
    rec.t = t;
    rec.p = intersection;
    rec.mat = mat;
    rec.set_face_normal(r, normal);
    rec.type = Type::QUAD;

    return true;

}

/**
 * * Given the hit point in plane coordinates,
 * @return false if it is outside the primitive.
 * @return true and set the hit record UV coordinates
 */
__device__ __host__
bool quad_data::is_interior(float a, float b, hit_record& rec) const {
    interval unit_interval = interval(0, 1);
    if (!unit_interval.contains(a) || !unit_interval.contains(b)) return false;
    rec.u = a;
    rec.v = b;
    return true;
}

//* Compute the bounding box of all four vertices
__device__ __host__
void quad_data::set_boundig_box() {
    auto bbox_diagonal1 = AaBb(Q, Q + u + v);
    auto bbox_diagonal2 = AaBb(Q + u, Q + v);
    bbox = AaBb(bbox_diagonal1, bbox_diagonal2);
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
    
    if (image_height <= 0 ) return glm::vec3(0.0f, 1.0f, 1.0f);
    //* Clamp input texture coordinates to [0,1] x [1,0]
    u = interval(0,1).clamp(u);
    v = 1.0 - interval(0,1).clamp(v);  //* Flip V to image coordinates

    auto i = int(u * image_width);
    auto j = int(v * image_height);
    auto pixel = pixel_data(i, j);
    auto color_scale = 1.0f / 255.0;
    return glm::vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);

}

__device__ __host__
    glm::vec3 noiseTexture_data::value(float u, float v, const glm::vec3& p) {
       
        // return glm::vec3(1.0f, 1.0f, 1.0f) * noisy.noise(scale * p);
        // return glm::vec3(1.0f, 1.0f, 1.0f) * noisy.trilinear_noise_smoothing(p);
        // return glm::vec3(1.0f, 1.0f, 1.0f) * noisy.hermitian_noise_smoothing(scale * p);
        //* 5.5 Random vectors Lattice points         
        // return glm::vec3(1.0f, 1.0f, 1.0f) * 0.5f * (1.0f + noisy.perlin_noise_smoothing(scale * p) );
        //* 5.6 Turbolence introduction
        // return glm::vec3(1.0f, 1.0f, 1.0f) * noisy.turbolence(p, 7);
        //* 5.7 Marble texture
        return glm::vec3(0.5f, .5f, 0.5f) * (1.0f + sinf(scale * p.z + 10 * noisy.turbolence(p, 7)));
        // return glm::vec3(0.7f, .7, 0.7f) * (1.0f + sinf(scale * p.z + 10 * noisy.turbolence(p, 7)));


        
    }

/**
 * @return the address of the three RGB bytes of the pixel at x, y.
    * @return magenta if there is no image data
    */
__device__ __host__
const unsigned char* imageTexture_data::pixel_data(int x, int y) const {
    static unsigned char magenta[] = {255, 0, 255};
    if (bdata == nullptr) {
        return magenta;
    }
    x = clamp(x, 0, image_width);
    y = clamp(y, 0, image_height);

    return bdata + y * bytes_per_scanline + x * bytes_per_pixel;
}


/**
 * @return the value clamped to the range [low, high]
    */
__device__ __host__ 
int imageTexture_data::clamp(int x, int low, int high) const {
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
    
}

/**
 * @brief Creates a vector of a 3D box (six sides) that contains the two opposites vertices a & b
 * 
 * @param sides 
 * @param a 
 * @param b 
 * @param mat 
 */
inline void box(std::vector<hittable>& sides, const glm::vec3& a, const glm::vec3& b, material* mat) {
    
    
    // construct the two opposite vertices with the minimum and maximum coordinates
    auto min = glm::vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    auto max = glm::vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));

    auto dx = glm::vec3(max.x - min.x, 0.0f, 0.0f);
    auto dy = glm::vec3(0, max.y - min.y, 0.0f);
    auto dz = glm::vec3(0, 0, max.z - min.z);

    auto side1 = hittable(hittable::make_quad(glm::vec3(min.x, min.y, max.z),  dx,  dy, mat)); // front
    auto side2 = hittable(hittable::make_quad(glm::vec3(max.x, min.y, max.z), -dz,  dy, mat)); // right
    auto side3 = hittable(hittable::make_quad(glm::vec3(max.x, min.y, min.z), -dx,  dy, mat)); // back
    auto side4 = hittable(hittable::make_quad(glm::vec3(min.x, min.y, min.z),  dz,  dy, mat)); // left
    auto side5 = hittable(hittable::make_quad(glm::vec3(min.x, max.y, max.z),  dx, -dz, mat)); // top
    auto side6 = hittable(hittable::make_quad(glm::vec3(min.x, min.y, min.z),  dx,  dz, mat)); // bottom

    sides.push_back(side1);
    sides.push_back(side2);
    sides.push_back(side3);
    sides.push_back(side4);
    sides.push_back(side5);
    sides.push_back(side6);
    
}