
#include <glm/glm.hpp>

struct material;


struct hitRecord {
    glm::vec3 p;
    glm::vec3 normal;
    material* mat_ptr;
    float t;
    bool front_face;

    __host__ __device__
    void set_face_normal(const ray& r, const glm::vec3& outward_normal) {
        front_face = glm::dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class sphere {
        public:
            __host__ __device__
            sphere() {}
            __host__ __device__
            sphere(const glm::vec3& center, float radius, material* mat) : center(center), radius(radius), mat(mat){}
            __host__ __device__
            bool hit(const ray& r, interval ray_t, hitRecord &rec) const  {
                glm::vec3 oc = r.origin - center;
                auto a = glm::dot(r.direction, r.direction);
                auto h = glm::dot(oc, r.direction);
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
                glm::vec3 outward_normal = (rec.p - center) / radius;
                rec.set_face_normal(r, outward_normal);
                rec.mat_ptr = mat;
                
                return true;
            }

        private:
            glm::vec3 center;
            float radius;
            material* mat;
    };

