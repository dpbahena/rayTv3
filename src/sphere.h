
// #include <glm/glm.hpp>
#include "aabb.h"


struct material;



// struct hitRecord {
//     glm::vec3 p;
//     glm::vec3 normal;
//     material* mat_ptr;
//     float t;
//     bool front_face;

//     __host__ __device__
//     void set_face_normal(const ray& r, const glm::vec3& outward_normal) {
//         front_face = glm::dot(r.direction, outward_normal) < 0;
//         normal = front_face ? outward_normal : -outward_normal;
//     }
// };


// class sphere {
//         public:
//             __host__ __device__
//             sphere() {}
//             // stationary sphere
//             __host__ __device__
//             sphere(const glm::vec3& center, float radius, material* mat) : center1(center), radius(radius), mat(mat), is_moving(false) {}
            
//             // moving sphere
//             __host__ __device__
//             sphere(const glm::vec3& center1, const glm::vec3& center2, float radius, material* mat) : center1(center1), radius(radius), mat(mat), is_moving(true) {

//                 center_vec = center2 - center1;
//             }


//             __host__ __device__
//             bool hit(const ray& r, interval ray_t, hitRecord &rec) const  {
//                 auto current_center = is_moving ? sphere_center(r.time()) : center1; 
//                 glm::vec3 oc = r.origin - current_center;
//                 auto a = glm::dot(r.direction, r.direction);
//                 auto h = glm::dot(oc, r.direction);
//                 auto c = glm::dot(oc, oc) - radius * radius;

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
//                 glm::vec3 outward_normal = (rec.p - current_center) / radius;
//                 rec.set_face_normal(r, outward_normal);
//                 rec.mat_ptr = mat;
                
//                 return true;
//             }

//         private:
//             glm::vec3 center1;
//             float radius;
//             material* mat;
//             bool is_moving;
//             glm::vec3 center_vec;

//             __device__ __host__
//             glm::vec3 sphere_center(float t) const {
//                 /* Linearly interpolate from center1 to center2 according to time, where t = 0 yields center1 + time * center_vec */
//                 return center1 + t * center_vec;
//             }
//     };

class sphere {
        public:
            __host__ __device__
            sphere() {}
            // stationary sphere
            __host__ __device__
            sphere(const glm::vec3& static_center, float radius, material* mat) : center(static_center, glm::vec3(0.0,0.0,0.0)), radius(radius), mat(mat){
                auto rvec = glm::vec3(radius, radius, radius);
                bbox = AaBb(static_center - rvec, static_center + rvec);
            }
            
            // moving sphere
            __host__ __device__
            sphere(const glm::vec3& center1, const glm::vec3& center2, float radius, material* mat) : center(center1, center2 - center1), radius(radius), mat(mat){
                auto rvec = glm::vec3(radius, radius, radius);
                AaBb box1(center.at(0) - rvec, center.at(0) + rvec);
                AaBb box2(center.at(1) - rvec, center.at(1) + rvec);
                bbox = AaBb(box1, box2);
            }

            __host__ __device__
            bool hit(const ray& r, interval ray_t, hitRecord &rec) const  {
                glm::vec3 current_center = center.at(r.time()); 
                glm::vec3 oc =  r.origin - current_center;
                auto a = glm::dot(r.direction, r.direction);
                auto h = glm::dot(r.direction, oc);
                auto c = glm::dot(oc, oc) - radius * radius;

                auto discriminant = h * h - a * c;
                if (discriminant < 0)
                    return false;

                auto sqrtd = std::sqrt(discriminant);

                // find the nearest root that lies in the acceptable range
                auto root = (-h - sqrtd) / a;
                if (!ray_t.surrounds(root)) {
                    root = (-h + sqrtd) / a;
                    if (!ray_t.surrounds(root))
                        return false;
                }

                rec.t = root;
                rec.p = r.at(rec.t);
                glm::vec3 outward_normal = (rec.p - current_center) / radius;
                rec.set_face_normal(r, outward_normal);
                rec.mat_ptr = mat;
                
                return true;
            }

            AaBb bounding_box() const { return bbox; }

        private:
            // glm::vec3 center1;
            ray center;
            float radius;
            material* mat;
            AaBb bbox;
            // bool is_moving;
            // glm::vec3 center_vec;

            // __device__ __host__
            // glm::vec3 sphere_center(float t) const {
            //     /* Linearly interpolate from center1 to center2 according to time, where t = 0 yields center1 + time * center_vec */
            //     return center1 + t * center_vec;
            // }
    };