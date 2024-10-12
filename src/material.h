#pragma once


// #include <curand_kernel.h>



enum class Type {NONE, SPHERE, BBOX, LAMBERTIAN, METAL, DIELECTRIC};

// __device__ inline glm::vec3 random_unit_vector(curandState_t* states, int i, int j);
// __device__ inline bool near_zero(const glm::vec3 v);
// __device__ inline float reflectance(float cosine, float refraction_index);
// __device__ inline float random_float(curandState_t* state);



// Define a struct for lambertian material
struct lambertian_data {
    glm::vec3 albedo;
};

struct metal_data {
    glm::vec3 albedo;
    float fuzz;
};

struct dielectric_data {
    /*  Refractive index (ri) in vacuum or air, or the ratio of the material's ri over the ri of the enclosing media */
    double refraction_index;
};



struct material {
    Type type;

    union {
        lambertian_data lambertian;
        metal_data metal;
        dielectric_data dielectric;
    };


    /* Default constructor */
    material(): type(Type::NONE) {}

    // constructor for type lambertian
    static material lambertian_material(const glm::vec3& albedo) {
        material obj;
        obj.type = Type::LAMBERTIAN;
        obj.lambertian.albedo = albedo;
        return obj;
    }
    // constructor for type metal
    static material metal_material(const glm::vec3& albedo, double fuzz) {
        material obj;
        obj.type = Type::METAL;
        obj.metal.albedo = albedo;
        obj.metal.fuzz = fuzz < 1 ? fuzz : 1.0;
        return obj;
    }
    // constructor for type dielectric
    static material dielectric_material(double refraction_index) {
        material obj;
        obj.type = Type::DIELECTRIC;
        obj.dielectric.refraction_index = refraction_index;
        return obj;
    }



};

// __device__
// static bool lambertian_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scatter, lambertian_data& lambertian, curandState_t* states,  int i, int j) {
//     auto scatter_direction = rec.normal + random_unit_vector(states,  i, j);

//     // Catch degenerate scatter direction
//     if (near_zero(scatter_direction))
//         scatter_direction = rec.normal;

//     scatter = ray(rec.p, scatter_direction);
//     attenuation = lambertian.albedo;
    
//     return true;
// }

// __device__
// static bool metal_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scatter, metal_data& metal, curandState_t* states,  int i, int j) {
//     glm::vec3 reflected = reflect(r_in.direction, rec.normal);
//     reflected = glm::normalize(reflected) + (metal.fuzz * random_unit_vector(states,  i, j));
//     scatter = ray(rec.p, reflected);
//     attenuation = metal.albedo;
    
//     return (glm::dot(scatter.direction, rec.normal) > 0);
// }


// __device__
// static bool dielectric_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scatter, dielectric_data& dielectric, curandState_t* states,  int i, int j) {
//     attenuation = glm::vec3(1.0, 1.0, 1.0);
//     double ri = rec.front_face ? (1.0/dielectric.refraction_index) : dielectric.refraction_index;

//     glm::vec3 unit_direction = glm::normalize(r_in.direction);
//     double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
//     double sin_theta = std::sqrt(1.0 - cos_theta*cos_theta);

//     bool cannot_refract = ri * sin_theta > 1.0;
//     glm::vec3 direction;

//     curandState_t x = states[i];  // for random data

//     if (cannot_refract || reflectance(cos_theta, ri) > random_float(&x) )
//         direction = reflect(unit_direction, rec.normal);
//     else   
//         direction = refract(unit_direction, rec.normal, ri);
    
//     states[i] = x; // save back value

//     scatter = ray(rec.p, direction);
//     return true;
// }




// struct alignas(16) material {
//     public:
//         bool (*scatter)(const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j);
//         MaterialType type;
    
    
// };


// struct lambertian {
//     public:
//         material base;
//         glm::vec3 albedo;

//         __device__ 
//         static bool scatter(const lambertian* self, const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j){
//             auto scatter_direction = rec.normal + random_unit_vector(states,  i, j);
            
//             // Catch degenerate scatter direction
//             if (near_zero(scatter_direction)) scatter_direction = rec.normal;
//             scattered = ray(rec.p, scatter_direction, r_in.time());
//             attenuation = self->albedo;  // Access the albedo using itself
           

//             return true;
//         }

//         __device__ __host__
//         lambertian(const glm::vec3& a) : albedo(a) {
//             base.scatter = (bool (*)(const ray&, const hitRecord&, glm::vec3&, ray&, curandState_t*, int, int))scatter;
//             base.type = LAMBERTIAN;
            
//         }
        
// };

// struct metal {
//    public:
//       material base;
//       glm::vec3 albedo;
//       float fuzz;

//       __device__ 
//       static bool scatter(const metal* self, const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j) {
//             glm::vec3 reflected = reflect(r_in.direction, rec.normal);
//             reflected = glm::normalize(reflected) + (static_cast<float>(self->fuzz) * random_unit_vector(states, i, j));
//             scattered = ray(rec.p, reflected, r_in.time());
//             attenuation = self->albedo;
            
//             return (glm::dot(scattered.direction, rec.normal) > 0.0f);
//       }

//       __device__ __host__
//       metal(const glm::vec3& a, float fuzz) : albedo(a), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {
//             base.scatter = (bool (*)(const ray&, const hitRecord&, glm::vec3&, ray&, curandState_t*, int, int))scatter;
//             base.type = METAL;
//       }
// };

// struct dielectric {
//    public:
//       material base;
//       float refraction_index;
//       /* Refractive index in vacumm or air, or the ratio of the material's refractive index over the refractive index of the enclosing media */

//       __device__ 
//       static bool scatter(const dielectric* self, const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j) {
//             attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
//             float ri = rec.front_face ? (1.0f / self->refraction_index) : self->refraction_index;
//             glm::vec3 unit_direction = glm::normalize(r_in.direction);
//             float cos_theta = fmin(glm::dot(-unit_direction, rec.normal), 1.0f);
//             float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

//             bool cannot_refract = ri * sin_theta > 1.0f;
//             glm::vec3 direction;
//             curandState_t x = states[i];  // for random data

//             if (cannot_refract || reflectance(cos_theta, ri) > random_float(&x) )
//                direction = reflect(unit_direction, rec.normal);
//             else   
//                direction = refract(unit_direction, rec.normal, ri);
            
//             states[i] = x; // save back value
//             scattered  = ray(rec.p, direction, r_in.time());
//             return true;
//       }

//       __device__ __host__
//       dielectric(float r_i) : refraction_index(r_i) {
//             base.scatter = (bool (*)(const ray&, const hitRecord&, glm::vec3&, ray&, curandState_t*, int, int))scatter;
//             base.type = DIELECTRIC;
//       }
// };