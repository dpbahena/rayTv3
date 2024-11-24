
// #include "aabb.h"
#pragma once

#include "hittable.h"
#include "texture.h"
#include <memory>


__device__ inline float     random_float(curandState_t* state);


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
bool hittableList_data::hit(const ray& r, interval ray_t, hit_record& rec, float randNumber)  const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
    
    
        for (int i = 0; i < objects_size; i++){
        if (objects[i].type == Type::SPHERE) {
            if (objects[i].sphere.hit(r, interval(ray_t.min, closest_so_far), temp_rec)){
                
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                
            }
        }
        if (objects[i].type == Type::QUAD) {
            if (objects[i].quad.hit(r, interval(ray_t.min, closest_so_far), temp_rec)){

                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;

            }
        }
        if (objects[i].type == Type::ROTATE_Y) {
            if (objects[i].rotateY.hit(r, interval(ray_t.min, closest_so_far), temp_rec, randNumber)){

                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;

            }
        }
        if (objects[i].type == Type::TRANSLATE) {
            if (objects[i].translate.hit(r, interval(ray_t.min, closest_so_far), temp_rec, randNumber)){

                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;

            }
        }
        if (objects[i].type == Type::MEDIUM) {
            if (objects[i].constantMedium.hit(r, interval(ray_t.min, closest_so_far), temp_rec, randNumber)){

                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;

            }
        }

        if (objects[i].type == Type::LIST) {
            if (objects[i].hittableList.hit(r, interval(ray_t.min, closest_so_far), temp_rec, randNumber)){

                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;

            }
        }


        


    }
    
    return hit_anything;
}

__device__ __host__
void hittableList_data::setList(hittable* hittables, size_t list_size) {
    objects = hittables;
    objects_size = list_size;
    
    // for (int i = 0; i < objects_size; i++){
    //     if (objects[i].type == Type::SPHERE) {
    //         bbox = AaBb(bbox, objects[i].sphere.bounding_box());
    //     }
    //     if (objects[i].type == Type::QUAD) {
    //         bbox = AaBb(bbox, objects[i].quad.bounding_box());
    //     }
    //     if (objects[i].type == Type::ROTATE_Y) {
    //         bbox = AaBb(bbox, objects[i].rotateY.bounding_box());
    //     }
    //     if (objects[i].type == Type::TRANSLATE) {
    //         bbox = AaBb(bbox, objects[i].translate.bounding_box());
    //     }
    //     if (objects[i].type == Type::MEDIUM) {
    //         bbox = AaBb(bbox, objects[i].constantMedium.bounding_box());
    //     }
    //     if (objects[i].type == Type::LIST) {
    //         bbox = AaBb(bbox, objects[i].hittableList.bounding_box());
    //     }
    // }
    
}


void hittableList_data::setNodes(BVHNode* nodes, hittable* hittables)
{
    nodeObjects = nodes;
    objects = hittables;
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

__device__ __host__
bool translate_data::hit(const ray& r, interval ray_t, hit_record& rec, float randNumber)  const {
    //* Move the ray backwards by the offset
    ray offset_r(r.origin - offset, r.direction, r.time());

    bool hit_anything = false;

    //* Determine whether an intersection exist along the offset ray (and if so, where)
    if (object->type == Type::SPHERE) {
        hit_anything = object->sphere.hit(offset_r, ray_t, rec);
    } else if (object->type == Type::QUAD) {
        hit_anything = object->quad.hit(offset_r, ray_t, rec);
    } else if (object->type == Type::ROTATE_Y) {
        hit_anything = object->rotateY.hit(offset_r, ray_t, rec, randNumber);
    } else if (object->type == Type::LIST) {
        hit_anything = object->hittableList.hit(offset_r, ray_t, rec, randNumber);
    } else if (object->type == Type::TRANSLATE) {
        hit_anything = object->translate.hit(offset_r, ray_t, rec, randNumber);
    } 
    
    if (hit_anything) {
    
        //* Move the intersection point forward by the offset
        rec.p += offset;
        return true;
    
    } else {
        
        return false;
    }
    
}

__device__ __host__
bool rotateY_data::hit(const ray& r, interval ray_t, hit_record& rec, float randNumber) const {
    //* Transform the ray from world space to object space
    auto origin     = glm::vec3(cos_theta * r.origin.x - sin_theta * r.origin.z, r.origin.y, sin_theta * r.origin.x + cos_theta * r.origin.z);
    auto direction  = glm::vec3(cos_theta * r.direction.x - sin_theta * r.direction.z, r.direction.y, sin_theta * r.direction.x + cos_theta * r.direction.z);

    ray rotated_r(origin, direction, r.time());

    bool hit_anything = false;

    //* Determine whether an intersection exists in object space (and if so, where)
    if (object->type == Type::QUAD) {
        hit_anything = object->quad.hit(rotated_r, ray_t, rec);
    } else if (object->type == Type::SPHERE) {
        hit_anything = object->sphere.hit(rotated_r, ray_t, rec);
    } else if (object->type == Type::TRANSLATE) {
        hit_anything = object->translate.hit(rotated_r, ray_t, rec, randNumber);
    } else if (object->type == Type::LIST) {
        hit_anything = object->hittableList.hit(rotated_r, ray_t, rec, randNumber);
    } else if (object->type == Type::ROTATE_Y) {
        hit_anything = object->rotateY.hit(rotated_r, ray_t, rec, randNumber);
    }
    if (hit_anything){
        //* Transform the intersection from object space back to world space
        rec.p       = glm::vec3(cos_theta * rec.p.x + sin_theta * rec.p.z, rec.p.y, -sin_theta * rec.p.x + cos_theta * rec.p.z);
        rec.normal  = glm::vec3(cos_theta * rec.normal.x + sin_theta * rec.normal.z, rec.normal.y, -sin_theta * rec.normal.x + cos_theta * rec.normal.z);
        return true;

    } else {

        return false;
    }
}

void rotateY_data::calculateBbox() {

    glm::vec3 min(MAXFLOAT, MAXFLOAT, MAXFLOAT);
    glm::vec3 max(-MAXFLOAT, -MAXFLOAT, -MAXFLOAT);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i * bbox.x.max + (1 - i) * bbox.x.min;
                auto y = j * bbox.y.max + (1 - j) * bbox.y.min;
                auto z = k * bbox.z.max + (1 - k) * bbox.z.min;

                auto newx =  cos_theta * x + sin_theta * z;
                auto newz = -sin_theta * x + cos_theta * z;

                glm::vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fminf(min[c], tester[c]);
                    max[c] = fmaxf(max[c], tester[c]);
                }

            }
        }
    }
    bbox = AaBb(min, max);

}


__device__ __host__
bool constantMedium_data::hit(const ray& r, interval ray_t, hit_record& rec, float randNumber) const {
    hit_record rec1, rec2;
    bool hit_anything = false;

    if (boundary->type == Type::QUAD) {
        hit_anything = boundary->quad.hit(r, interval::universe(), rec1);
        if (!hit_anything) return false;
        hit_anything = boundary->quad.hit(r, interval(rec1.t + 0.0001f, MAXFLOAT), rec2);
        if (!hit_anything) return false;
    } 
    if (boundary->type == Type::SPHERE) {
        hit_anything = boundary->sphere.hit(r, interval::universe(), rec1);
        if (!hit_anything) return false;
        hit_anything = boundary->sphere.hit(r, interval(rec1.t + 0.0001f, MAXFLOAT), rec2);
        if (!hit_anything) return false;
    } 
    if (boundary->type == Type::TRANSLATE) {
        hit_anything = boundary->translate.hit(r, interval::universe(), rec1, randNumber);
        if (!hit_anything) return false;
        hit_anything = boundary->translate.hit(r, interval(rec1.t + 0.0001f, MAXFLOAT), rec2, randNumber);
        if (!hit_anything) return false;

    } 
    if (boundary->type == Type::ROTATE_Y) {
        hit_anything = boundary->rotateY.hit(r, interval::universe(), rec1, randNumber);
        if (!hit_anything) return false;
        hit_anything = boundary->rotateY.hit(r, interval(rec1.t + 0.0001f, MAXFLOAT), rec2, randNumber);
        if (!hit_anything) return false;
        
    }
   
    if (boundary->type == Type::LIST) {
        hit_anything = boundary->hittableList.hit(r, interval::universe(), rec1, randNumber);
        if (!hit_anything) return false;
        hit_anything = boundary->hittableList.hit(r, interval(rec1.t + 0.0001f, MAXFLOAT), rec2, randNumber);
        if (!hit_anything) return false;
        
    }

    if (rec1.t < ray_t.min) rec1.t = ray_t.min;
    if (rec2.t > ray_t.max) rec2.t = ray_t.max;

    if (rec1.t >= rec2.t) return false;
    if (rec1.t < 0) return rec1.t = 0;

    auto ray_length = glm::length(r.direction);
    
    auto distance_inside_boundary = (rec2.t -rec1.t) * ray_length;
    
    auto hit_distance = neg_inv_density * logf(randNumber);
    
    
    if (hit_distance > distance_inside_boundary) return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);  

    rec.normal = glm::vec3(1.0f, 0.0f, 0.0f);    // arbitrary
    rec.front_face = true;
    rec.mat = phase_function;

    return true;
        
   
}

__device__ __host__
AaBb constantMedium_data::bounding_box() const {
    AaBb bbox;
    
    if (boundary->type == Type::QUAD) {
        bbox = boundary->quad.bounding_box();
    } else if (boundary->type == Type::SPHERE) {
        bbox = boundary->sphere.bounding_box();
    } else if (boundary->type == Type::TRANSLATE) {
        bbox = boundary->translate.bounding_box();
    } else if (boundary->type == Type::ROTATE_Y) {
        bbox = boundary->rotateY.bounding_box(); 
    } else if (boundary->type == Type::LIST) {
        bbox = boundary->hittableList.bounding_box();
    }
    return bbox;
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

/**
 * @brief Creates a vector of a 3D box (six sides) that contains the two opposites vertices a & b
 * 
 * @param sides 
 * @param a 
 * @param b 
 * @param mat 
 */
hittable* box( const glm::vec3& a, const glm::vec3& b, material* mat) {
    
    auto sides = new hittable[6];
    
    // construct the two opposite vertices with the minimum and maximum coordinates
    auto min = glm::vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    auto max = glm::vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));

    auto dx = glm::vec3(max.x - min.x, 0.0f, 0.0f);
    auto dy = glm::vec3(0, max.y - min.y, 0.0f);
    auto dz = glm::vec3(0, 0, max.z - min.z);

    auto side0 = hittable(hittable::make_quad(glm::vec3(min.x, min.y, max.z),  dx,  dy, mat)); // front
    auto side1 = hittable(hittable::make_quad(glm::vec3(max.x, min.y, max.z), -dz,  dy, mat)); // right
    auto side2 = hittable(hittable::make_quad(glm::vec3(max.x, min.y, min.z), -dx,  dy, mat)); // back
    auto side3 = hittable(hittable::make_quad(glm::vec3(min.x, min.y, min.z),  dz,  dy, mat)); // left
    auto side4 = hittable(hittable::make_quad(glm::vec3(min.x, max.y, max.z),  dx, -dz, mat)); // top
    auto side5 = hittable(hittable::make_quad(glm::vec3(min.x, min.y, min.z),  dx,  dz, mat)); // bottom

    sides[0] = side0;
    sides[1] = side1;
    sides[2] = side2;
    sides[3] = side3;
    sides[4] = side4;
    sides[5] = side5;

    return sides;
}