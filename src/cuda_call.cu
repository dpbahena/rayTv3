#include "cuda_call.h"
#include "ray.h"
#include "interval.h"

// #include "bvh_node.h"
#include "cuda_bvh_node.h"
#include "mem_manager.h"
#include "texture.h"

#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
// #include <curand_kernel.h>

#define MAX_STACK_SIZE 20


__device__ inline glm::vec3 random_on_hemisphere(curandState_t* states,  int i, int j,const glm::vec3& normal);
__device__ inline glm::vec3 random_in_unit_sphere(curandState_t* states,  int i, int j);
__device__ inline glm::vec3 random_vector_in_range(curandState_t* states,  int i, int j, float min, float max);
__device__ inline glm::vec3 random_vector(curandState_t* states,  int i, int j);
__device__ inline float     random_float_in_range(curandState_t* state, float a, float b);
__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n);
__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat);
__device__ inline glm::vec3 random_in_unit_disk(curandState_t* states,  int i, int j);
__device__ inline glm::vec3 defocus_disk_sample(curandState_t* states,  int i, int j, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v);
// __device__ inline glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const ray &r, const hittable_list& world);
__device__ inline glm::vec3 random_unit_vector(curandState_t* states, int i, int j);
__device__ inline bool      near_zero(const glm::vec3 v);
__device__ inline float     reflectance(float cosine, float refraction_index);
__device__  inline float     random_float(curandState_t* state);







inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    // static std::mt19937 generator;   // uncomment for same results
    static std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));  // comment for same results
    return distribution(generator);
}

inline double random_double(float min, float max) {
    static std::uniform_real_distribution<double> distribution;
    // static std::mt19937 generator;   // uncomment for same results
    static std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));   // comment for same results
    return distribution(generator, std::uniform_real_distribution<double>::param_type(min, max));
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return int(random_double(min, max+1));
}

inline glm::vec3 unit_vector(const glm::vec3& v){
    // auto a = glm::length(v);
    // return glm::vec3(v/a);
    return glm::normalize(v);
}



#define checkCuda(result) { gpuAssert((result), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(code == cudaSuccess);
   }
}


__device__
static bool lambertian_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, lambertian_data& lambertian, curandState_t* states,  int i, int j) {
    auto scatter_direction = rec.normal + random_unit_vector(states,  i, j);

    // Catch degenerate scatter direction
    if (near_zero(scatter_direction))
        scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction, r_in.time());
    // attenuation = lambertian.albedo;
    attenuation = lambertian.tex->value(rec.u, rec.v, rec.p);
    
    return true;
}

__device__
static bool metal_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, metal_data& metal, curandState_t* states,  int i, int j) {
    glm::vec3 reflected = reflect(r_in.direction, rec.normal);
    reflected = glm::normalize(reflected) + (metal.fuzz * random_unit_vector(states,  i, j));
    scattered = ray(rec.p, reflected, r_in.time());
    attenuation = metal.albedo;
    
    return (glm::dot(scattered.direction, rec.normal) > 0);
}


__device__
static bool dielectric_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, dielectric_data& dielectric, curandState_t* states,  int i, int j) {
    attenuation = glm::vec3(1.0, 1.0, 1.0);
    float ri = rec.front_face ? (1.0f / dielectric.refraction_index) : dielectric.refraction_index;

    glm::vec3 unit_direction = glm::normalize(r_in.direction);
    float cos_theta = min(glm::dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0f;
    glm::vec3 direction;

    curandState_t x = states[i];  // for random data

    if (cannot_refract || reflectance(cos_theta, ri) > random_float(&x) )
        direction = reflect(unit_direction, rec.normal);
    else   
        direction = refract(unit_direction, rec.normal, ri);
    
    states[i] = x; // save back value

    scattered = ray(rec.p, direction, r_in.time());
    return true;
}

__device__ __host__
static glm::vec3 emitted(float u, float v, const glm::vec3& p, diffuseLight_data& diffuse){
    return diffuse.tex->value(u, v, p);
}

__device__ 
static bool isotropic_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, isotropic_data& isotropic, curandState_t* states,  int i, int j){
    scattered = ray(rec.p, random_unit_vector(states,  i, j), r_in.time());
    if(isotropic.tex->type == Type::CHECKER) {
     attenuation = isotropic.tex->checkerTexture.value(rec.u, rec.v, rec.p);
    } else if (isotropic.tex->type == Type::IMAGE) {
     attenuation = isotropic.tex->imageTexture.value(rec.u, rec.v, rec.p);
    } else if (isotropic.tex->type == Type::NOISE) {
     attenuation = isotropic.tex->noiseTexture.value(rec.u, rec.v, rec.p);
    } else if (isotropic.tex->type == Type::SOLID) {
     attenuation = isotropic.tex->solidColor.value(rec.u, rec.v, rec.p);
    }
    return true;

}




__device__ __host__
static glm::vec3 checkeredTexture_value(float u, float v, const glm::vec3& p, checkerTexture_data& checkered) {
    auto xInteger = int(floor(checkered.inv_scale * p.x));
    auto yInteger = int(floor(checkered.inv_scale * p.y));
    auto zInteger = int(floor(checkered.inv_scale * p.z));

    bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;
    // return isEven ? even->checkerTexture.value(u, v, p) : odd->checkerTexture.value(u, v, p);
    return isEven ? checkered.even->value(u, v, p) : checkered.odd->value(u, v, p);
}



__device__
inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n){
    return v - 2 * glm::dot(v, n) * n;
}

__device__ 
inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat){
    auto cos_theta = fmin(glm::dot(-uv, n), 1.0f);
    glm::vec3 r_out_perp = static_cast<float>(etai_over_etat) * (uv + cos_theta * n);
    glm::vec3 r_out_parallel = static_cast<float>(-sqrt(fabs(1.0 - glm::dot(r_out_perp, r_out_perp))) ) * n;
    return r_out_perp + r_out_parallel;
}

__device__ 
inline float reflectance(float cosine, float refraction_index){
    // Use Schilick's approximation for reflectance
    auto r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow( (1.0f - cosine), 5.0f);
}

__device__ 
inline float random_float(curandState_t* state){
    return curand_uniform_double(state);
}




__device__ 
glm::vec3 random_unit_vector(curandState_t* states, int i, int j){
    auto p = random_in_unit_sphere(states, i, j);
    return glm::normalize(p);
}

__device__
bool near_zero(const glm::vec3 v) {
    auto s = 1e-8f;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

__device__
glm::vec3 sample_square(curandState_t* states, int &i, int &j) {
    curandState_t x = states[i];
    curandState_t y = states[j];
    auto a = random_float(&x) - 0.5f;
    auto b = random_float(&y) - 0.5f;
    states[i] = x; // save back the value
    states[j] = y;
    return glm::vec3(a, b, 0.0f);
}

__device__
float linear_to_gamma(float linear_component)
{
    if (linear_component > 0.0f)
        return std::sqrt(linear_component);

    return 0;
}

__device__
uint32_t colorToUint32_t(glm::vec3& c)
{
    /* Ensure that the input values within the range [0.0, 1.0] */
    c.x = (c.x < 0.0f) ? 0.0f : ((c.x > 1.0f) ? 1.0f : c.x);  // red
    c.y = (c.y < 0.0f) ? 0.0f : ((c.y > 1.0f) ? 1.0f : c.y);  // green
    c.z = (c.z < 0.0f) ? 0.0f : ((c.z > 1.0f) ? 1.0f : c.z);  // blue

    // Apply a linear to gamma transform for gamma 2
    c.x = linear_to_gamma(c.x);
    c.y = linear_to_gamma(c.y);
    c.z = linear_to_gamma(c.z);

    // convert to integers
    uint32_t ri = static_cast<uint32_t>(c.x * 255.0);
    uint32_t gi = static_cast<uint32_t>(c.y * 255.0);
    uint32_t bi = static_cast<uint32_t>(c.z * 255.0);

    // Combine into a single uint32_t with FF for alpha (opacity)
    uint32_t color = (0x00 << 24) | (ri << 16) | (gi << 8) | bi;

    return color;
}

__device__
glm::vec3 random_on_hemisphere(curandState_t* states,  int i, int j,const glm::vec3& normal) {
    glm::vec3 on_unit_sphere = random_unit_vector(states, i, j);
    if (glm::dot(on_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__
glm::vec3 random_in_unit_sphere(curandState_t* states,  int i, int j) {
    while (true) {
        glm::vec3 p = random_vector_in_range(states, i, j, -1.0f ,1.0f);
        if (glm::dot(p,p) < 1.0f){
            return p;
        }
    }
}

__device__
glm::vec3 random_in_unit_disk(curandState_t* states,  int i, int j){
    curandState_t x = states[i];
    curandState_t y = states[j];
    while (true) {
        auto p = glm::vec3(random_float_in_range(&x, -1, 1), random_float_in_range(&y, -1, 1), 0);
        if (glm::dot(p,p) < 1.0f)
            return p;
    }
}

 __device__
glm::vec3 defocus_disk_sample(curandState_t* states,  int i, int j, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v) {
    // returns a random point in the camera defocus disk
    glm::vec3 p = random_in_unit_disk(states, i, j);
    return center + p.x * defocusDisk_u + p.y * defocusDisk_v;
}

__device__
glm::vec3 random_vector_in_range(curandState_t* states,  int i, int j, float min, float max){
    curandState_t x = states[i];
    curandState_t y = states[j];
    float a = random_float_in_range(&x, min, max);
    float b = random_float_in_range(&y, min, max);
    float c = random_float_in_range(&x, min, max);

    // float c = a * b;
    states[i] = x; // save value back
    states[j] = y;
    return glm::vec3(a, b, c);
}
__device__
glm::vec3 random_vector(curandState_t* states,  int i, int j){
    curandState_t x = states[i];
    curandState_t y = states[j];
    float a = random_float(&x);
    float b = random_float(&y);
    float c = random_float(&x); //a * b;
    states[i] = x; // save value back
    states[j] = y;
    return glm::vec3(a, b, c);

}








__device__ float random_float_in_range(curandState_t* state, float a, float b) {
    // return a + (b - a) * curand_uniform_float(state);  // this does not include b  e.g -1 to 1.0  it does not include 1.0
    return a + (b - a) * (curand_uniform_double(state) - 0.5) * 2.0;  // this approach includes the upper limit   -1 to 1.0  it includes 1.0
}

/**
 * @return a random integer in [min, max] including the upper limit
 */
__device__ int random_int(curandState_t* state, int a, int b) {
    return static_cast<int>(a + (b - a) * (curand_uniform_double(state) - 0.5) * 2.0);  // this approach includes the upper limit   -1 to 1.0  it includes 1.0
}


// YES BVH
__device__
glm::vec3 ray_color(curandState_t* state, int i, int j, int depth, const glm::vec3& background, const ray& r, const hittable& world, const BVHNode* __restrict__ nodes, const hittable* __restrict__ hittables, int* stack) {
    ray cur_ray = r;
    glm::vec3 cur_attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 final_color = glm::vec3(0.0f, 0.0f, 0.0f);

    // Loop through the ray bounces up to the specified depth
    for (int k = 0; k < depth; k++) {
        hit_record rec;

        // Check if the ray hits anything; if not, add the background color and return
        if (!hit(cur_ray, interval(0.001f, FLT_MAX), rec, nodes, hittables, stack)) {
            final_color += cur_attenuation * background;
            return final_color;
        }

        // Ensure that the material pointer is valid
        // if (!rec.mat) return final_color;

        // Handle emission from the material
        glm::vec3 color_from_emission = glm::vec3(0.0f, 0.0f, 0.0f);
        if (rec.mat->type == Type::DIFFUSE) {
            color_from_emission = emitted(rec.u, rec.v, rec.p, rec.mat->diffuseLight);
        }

        // Add the emitted light to the final color
        // final_color += cur_attenuation * (color_from_emission * 3.5f);  // 3.5 creates intensity ..(my own idea)
        final_color += cur_attenuation * color_from_emission;

        // Prepare to handle scattering
        ray scattered;
        glm::vec3 attenuation;
        bool did_scatter = false;

        // Scatter based on the material type
        if (rec.mat->type == Type::METAL) {
            did_scatter = metal_scatter(cur_ray, rec, attenuation, scattered, rec.mat->metal, state, i, j);
        } else if (rec.mat->type == Type::LAMBERTIAN) {
            did_scatter = lambertian_scatter(cur_ray, rec, attenuation, scattered, rec.mat->lambertian, state, i, j);
        } else if (rec.mat->type == Type::DIELECTRIC) {
            did_scatter = dielectric_scatter(cur_ray, rec, attenuation, scattered, rec.mat->dielectric, state, i, j);
        }

        // If scattering did not occur, return the accumulated color
        if (!did_scatter) {
            return final_color;
        }

        // Update the current ray and attenuation for the next bounce
        cur_ray = scattered;
        cur_attenuation *= attenuation;
    }

    // Return the accumulated color after all bounces
    return final_color;
}

// YES BVH  hittable world list
__device__
glm::vec3 ray_color(curandState_t* state, int i, int j, int depth, const glm::vec3& background, const ray& r, hittable* world, int* stack) {
    ray cur_ray = r;
    glm::vec3 cur_attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 final_color = glm::vec3(0.0f, 0.0f, 0.0f);

    // Loop through the ray bounces up to the specified depth
    for (int k = 0; k < depth; k++) {
        hit_record rec;

        // Check if the ray hits anything; if not, add the background color and return
        if (!hit(cur_ray, interval(0.001f, FLT_MAX), rec, world->hittableList.nodeObjects, world->hittableList.objects, stack)) {
            final_color += cur_attenuation * background;
            return final_color;
        }

        // Ensure that the material pointer is valid
        // if (!rec.mat) return final_color;

        // Handle emission from the material
        glm::vec3 color_from_emission = glm::vec3(0.0f, 0.0f, 0.0f);
        if (rec.mat->type == Type::DIFFUSE) {
            color_from_emission = emitted(rec.u, rec.v, rec.p, rec.mat->diffuseLight);
        }

        // Add the emitted light to the final color
        // final_color += cur_attenuation * (color_from_emission * 3.5f);  // 3.5 creates intensity ..(my own idea)
        final_color += cur_attenuation * color_from_emission;

        // Prepare to handle scattering
        ray scattered;
        glm::vec3 attenuation;
        bool did_scatter = false;

        // Scatter based on the material type
        if (rec.mat->type == Type::METAL) {
            did_scatter = metal_scatter(cur_ray, rec, attenuation, scattered, rec.mat->metal, state, i, j);
        } else if (rec.mat->type == Type::LAMBERTIAN) {
            did_scatter = lambertian_scatter(cur_ray, rec, attenuation, scattered, rec.mat->lambertian, state, i, j);
        } else if (rec.mat->type == Type::DIELECTRIC) {
            did_scatter = dielectric_scatter(cur_ray, rec, attenuation, scattered, rec.mat->dielectric, state, i, j);
        }

        // If scattering did not occur, return the accumulated color
        if (!did_scatter) {
            return final_color;
        }

        // Update the current ray and attenuation for the next bounce
        cur_ray = scattered;
        cur_attenuation *= attenuation;
    }

    // Return the accumulated color after all bounces
    return final_color;
}


// NO BVH
// __device__
// glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const glm::vec3& background, const ray &r, const hittable_list& world) {
//     ray cur_ray = r;
//     glm::vec3 cur_attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
//     glm::vec3 final_color     = glm::vec3(0.0f, 0.0f, 0.0f);
    
//     // Loop through the ray bounces up to the specified depth
//     for (int k = 0; k < depth; k++){
//         hit_record rec;
        
//         //* Check if the ray hits anything; if not, add the background color and return;
//         if(!world.hit(cur_ray, interval(0.001f, FLT_MAX), rec)){
//             final_color += cur_attenuation * background;
//             return final_color;
//         }
//         //* Handle emitted light fromt he material
//         glm::vec3 color_from_emission = glm::vec3(0.0f, 0.0f, 0.0f);
//         if (rec.mat->type == Type::DIFFUSE){
//             color_from_emission = emitted(rec.u, rec.v, rec.p, rec.mat->diffuseLight);
//         }
//         //* Add the emitted light to the final color
//         final_color += cur_attenuation * color_from_emission;

//         //* Prepare to handle scattering
//             // auto dir = rec.normal + random_unit_vector(state, i, j); // first approach using Lambertian  reflection
//         ray scattered;
//         glm::vec3 attenuation;
//         bool did_scatter = false;
//         //* Scatter based on material type
//         if (rec.mat->type == Type::METAL){
//             did_scatter = metal_scatter(cur_ray, rec, attenuation, scattered, rec.mat->metal, state, i, j);
//         } else if (rec.mat->type == Type::LAMBERTIAN){
//             did_scatter = lambertian_scatter(cur_ray, rec, attenuation, scattered, rec.mat->lambertian, state, i, j);
//         } else if (rec.mat->type == Type::DIELECTRIC){
//             did_scatter = dielectric_scatter(cur_ray, rec, attenuation, scattered, rec.mat->dielectric, state, i, j);    
//         }
//         //* If scattering did not occur, return the accumulated color
//         if(!did_scatter) {
//             return final_color;
//         }
        
//         //* Update the current ray and attenuation for the next bounce
//         cur_ray = scattered;
//         cur_attenuation *= attenuation;
//     }
    
//     // Return the accumulated color after all bounces
//     return final_color;
// }

// NO BVH HITTABLE
__device__
glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const glm::vec3& background, const ray &r, const hittable* world) {
    ray cur_ray = r;
    glm::vec3 cur_attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 final_color     = glm::vec3(0.0f, 0.0f, 0.0f);
    
    // Loop through the ray bounces up to the specified depth
    for (int k = 0; k < depth; k++){
        hit_record rec;
        curandState_t x = state[i];
        float randNumber = random_float(&x);
        state[i] = x;  // saves the random back
        //* Check if the ray hits anything; if not, add the background color and return;
        // if(!world->hit(cur_ray, interval(0.001f, FLT_MAX), rec)){
        if(!world->hittableList.hit(cur_ray, interval(0.001f, FLT_MAX), rec, randNumber )){
            final_color += cur_attenuation * background;
            return final_color;
        }
        //* Handle emitted light fromt he material
        glm::vec3 color_from_emission = glm::vec3(0.0f, 0.0f, 0.0f);
        if (rec.mat->type == Type::DIFFUSE){
            color_from_emission = emitted(rec.u, rec.v, rec.p, rec.mat->diffuseLight);
        }
        //* Add the emitted light to the final color
        final_color += cur_attenuation * color_from_emission;

        //* Prepare to handle scattering
            // auto dir = rec.normal + random_unit_vector(state, i, j); // first approach using Lambertian  reflection
        ray scattered;
        glm::vec3 attenuation;
        bool did_scatter = false;
        //* Scatter based on material type
        if (rec.mat->type == Type::METAL){
            did_scatter = metal_scatter(cur_ray, rec, attenuation, scattered, rec.mat->metal, state, i, j);
        } else if (rec.mat->type == Type::LAMBERTIAN){
            did_scatter = lambertian_scatter(cur_ray, rec, attenuation, scattered, rec.mat->lambertian, state, i, j);
        } else if (rec.mat->type == Type::DIELECTRIC){
            did_scatter = dielectric_scatter(cur_ray, rec, attenuation, scattered, rec.mat->dielectric, state, i, j);    
        } else if (rec.mat->type == Type::ISOTROPIC){
            did_scatter = isotropic_scatter(cur_ray, rec, attenuation, scattered, rec.mat->isotropic, state, i, j);    
        }
        //* If scattering did not occur, return the accumulated color
        if(!did_scatter) {
            return final_color;
        }
        
        //* Update the current ray and attenuation for the next bounce
        cur_ray = scattered;
        cur_attenuation *= attenuation;
    }
    
    // Return the accumulated color after all bounces
    return final_color;
}


__device__
ray get_ray(curandState_t* states, int &i, int &j, glm::vec3& pixel00_loc, glm::vec3& cameraCenter, glm::vec3& delta_u, glm::vec3& delta_v, float& defocusAngle, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v) {
    /* Construct a camara ray originating from the defocus disk and directed at a randdomly sampled point around the pixel locations i, j */
    auto offset = sample_square(states, i, j);
    auto pixel_sample = pixel00_loc 
                        + ((i + offset.x) * delta_u)
                        + ((j + offset.y) * delta_v);
    
    auto ray_origin =  (defocusAngle <= 0) ? cameraCenter : defocus_disk_sample(states, i, j, cameraCenter, defocusDisk_u, defocusDisk_v);
    auto ray_direction = pixel_sample - ray_origin;
    auto x = states[i];
    auto ray_time = random_float(&x);
    states[i] = x; // put value back after using it

    return ray(ray_origin, ray_direction, ray_time);
}

// Use high-resolution clock to generate a seed
unsigned int seed = static_cast<unsigned int>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()
);

__global__ void init_random(unsigned int seed, curandState_t* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}


// YES BVH
__global__ void rayTracer_kernel(curandState_t* states, Camera* cam, uint32_t* image, hittable* world, const  BVHNode* __restrict__ nodes, const hittable* __restrict__ hittables) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam->image_width || j >= cam->image_height) return;
    // compute a unique thread ID within the block
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;   // like column calculations

    extern __shared__ int shared_memory[];
    int* stack = &shared_memory[thread_id * MAX_STACK_SIZE];  

    glm::vec3 color = {0.0f, 0.0f, 0.0f};
    for (int sample = 0; sample < cam->samples_per_pixel; sample++){
        ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center, cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle, cam->defocus_disk_u, cam->defocus_disk_v);
        color  += ray_color(states, i, j, cam->max_depth, cam->background, r, *world, nodes, hittables, stack);
        // color  += ray_color(states, i, j, depth, r, *world);
    }
    // float pixel_sample_scale = 1.0f / static_cast<float>(cam->samples_per_pixel); // color scale factor for a sume of pixel samples
    color *= cam->pixel_sample_scale;
    image[cam->image_width * j + i] = colorToUint32_t(color);  
}

// YES BVH hittable   list
__global__ void rayTracer_nodes_kernel(curandState_t* states, Camera* cam, uint32_t* image, hittable* world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam->image_width || j >= cam->image_height) return;
    // compute a unique thread ID within the block
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;   // like column calculations

    extern __shared__ int shared_memory[];
    int* stack = &shared_memory[thread_id * MAX_STACK_SIZE];  

    glm::vec3 color = {0.0f, 0.0f, 0.0f};
    for (int sample = 0; sample < cam->samples_per_pixel; sample++){
        ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center, cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle, cam->defocus_disk_u, cam->defocus_disk_v);
        color  += ray_color(states, i, j, cam->max_depth, cam->background, r, world, stack);
        // color  += ray_color(states, i, j, depth, r, *world);
    }
    // float pixel_sample_scale = 1.0f / static_cast<float>(cam->samples_per_pixel); // color scale factor for a sume of pixel samples
    color *= cam->pixel_sample_scale;
    image[cam->image_width * j + i] = colorToUint32_t(color);  
}

// NO BVH 
// __global__ void rayTracer_kernel(curandState_t* states, Camera* cam, uint32_t* image, hittable_list* world) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= cam->image_width || j >= cam->image_height) return;
    
//     glm::vec3 color = {0.0f, 0.0f, 0.0f};
//     for (int sample = 0; sample < cam->samples_per_pixel; sample++){
//         ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center, cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle, cam->defocus_disk_u, cam->defocus_disk_v);
//         color  += ray_color(states, i, j, cam->max_depth, cam->background, r, *world);
//     }
    
//     color *= cam->pixel_sample_scale;
//     image[cam->image_width * j + i] = colorToUint32_t(color);  
// }

// NO BVH HITTABLE*
__global__ void rayTracer_kernel(curandState_t* states, Camera* cam, uint32_t* image, hittable* world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam->image_width || j >= cam->image_height) return;
    
    glm::vec3 color = {0.0f, 0.0f, 0.0f};
    for (int sample = 0; sample < cam->samples_per_pixel; sample++){
        ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center, cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle, cam->defocus_disk_u, cam->defocus_disk_v);
        color  += ray_color(states, i, j, cam->max_depth, cam->background, r, world);
    }
    
    color *= cam->pixel_sample_scale;
    image[cam->image_width * j + i] = colorToUint32_t(color);  
}




void bouncing_spheres(Camera& cam, std::vector<material*> device_materials, std::vector<texture*> device_textures,  std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(13.0f, 2.0f,  3.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.6f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();

    
    std::vector<hittable> h_hittables_list;
    material* d_ground;
    texture* d_ground_tex;
    
    // ground 
    /* material */
    texture h_ground_tex = texture::checker_texture(0.32f, glm::vec3(0.2f, 0.3f, 0.1f), glm::vec3(0.9f, 0.9f, 0.9f));
    checkCuda(cudaMalloc((void**)&d_ground_tex, sizeof(texture)) );
    checkCuda(cudaMemcpy(d_ground_tex, &h_ground_tex, sizeof(texture), cudaMemcpyHostToDevice) );
    device_textures.push_back(d_ground_tex);
    material h_ground = material::lambertian_material(d_ground_tex);
    checkCuda(cudaMalloc((void**)&d_ground, sizeof(material)) );
    checkCuda(cudaMemcpy(d_ground, &h_ground, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_ground);
    
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,-1000.0, 0.0), 1000, d_ground);
    h_hittables_list.push_back(hittable_obj);

    // Create random spheres 
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_material = random_double();
            glm::vec3 center(a + 0.9f * random_double(), 0.2f, b + 0.9f * random_double());
            
            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                if(choose_material < 0.8f) {
                    // difuse
                    glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
                    auto a_material = material::lambertian_material(albedo);
                    material* d_mat;
                    checkCuda(cudaMalloc((void**)&d_mat, sizeof(material)) );
                    checkCuda(cudaMemcpy(d_mat, &a_material, sizeof(material), cudaMemcpyHostToDevice) );
                    device_materials.push_back(d_mat);
                    glm::vec3 center2 = center + glm::vec3(0,random_double(0, 0.5), 0);
                    auto sphere = hittable::make_sphere(center, center2, 0.2f, d_mat);
                    h_hittables_list.push_back(sphere);

                }else if(choose_material < 0.95f) {
                    // metal
                    glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
                    float fuzz = random_double(0.0f, 0.5f);
                    auto a_material = material::metal_material(albedo, fuzz);
                    material* d_mat;
                    checkCuda(cudaMalloc((void**)&d_mat, sizeof(material)) );
                    checkCuda(cudaMemcpy(d_mat, &a_material, sizeof(material), cudaMemcpyHostToDevice) );
                    device_materials.push_back(d_mat);
                    auto sphere = hittable::make_sphere(center, 0.2f, d_mat);
                    h_hittables_list.push_back(sphere);
                }
                else  {
                    // dielectric
                    glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
                    auto a_material = material::dielectric_material(1.5);
                    material* d_mat;
                    checkCuda(cudaMalloc((void**)&d_mat, sizeof(material)) );
                    checkCuda(cudaMemcpy(d_mat, &a_material, sizeof(material), cudaMemcpyHostToDevice) );
                    device_materials.push_back(d_mat);
                    auto sphere = hittable::make_sphere(center, 0.2f, d_mat);
                    h_hittables_list.push_back(sphere);
                }
            }
        }
    }


    // Three secundary spheres

    /* material */
    material h_mat1 = material::dielectric_material(1.5f);
    material* d_mat1;
    checkCuda(cudaMalloc((void**)&d_mat1, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat1, &h_mat1, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat1);
    hittable_obj = hittable::make_sphere(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, d_mat1);
    h_hittables_list.push_back(hittable_obj);
    
    // /* Material */
    material h_mat2 = material::lambertian_material(glm::vec3(0.4f, 0.2f, 0.1f));
    material* d_mat2;
    checkCuda(cudaMalloc((void**)&d_mat2, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat2, &h_mat2, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat2);
    hittable_obj = hittable::make_sphere(glm::vec3(-4.0f, 1.0f, 0.0f), 1.0f, d_mat2);
    h_hittables_list.push_back(hittable_obj);

    // // /* Material */
    material h_mat3 = material::metal_material(glm::vec3(0.7f, 0.6f, 0.5f), 0.0);
    material* d_mat3;
    checkCuda(cudaMalloc((void**)&d_mat3, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat3, &h_mat3, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat3);
    hittable_obj = hittable::make_sphere(glm::vec3(4.0f, 1.0f, 0.0f), 1.0f, d_mat3);
    h_hittables_list.push_back(hittable_obj);
 
    
    size_t number_of_hittables = h_hittables_list.size();
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );
     

     

    // hittable_list h_world;
    auto h_world = hittable::make_hittableList();

    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
    
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }
    

}

void checkered_spheres(Camera& cam, std::vector<material*> device_materials, std::vector<texture*> device_textures,  std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(13.0f, 2.0f,  3.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();

    
    std::vector<hittable> h_hittables_list;
    material* d_ground;
    texture* d_ground_tex;
    
    // ground 
    //* Texture
    texture h_ground_tex = texture::checker_texture(0.32f, glm::vec3(0.2f, 0.3f, 0.1f), glm::vec3(0.9f, 0.9f, 0.9f));
    checkCuda(cudaMalloc((void**)&d_ground_tex, sizeof(texture)) );
    checkCuda(cudaMemcpy(d_ground_tex, &h_ground_tex, sizeof(texture), cudaMemcpyHostToDevice) );
    device_textures.push_back(d_ground_tex);
    //* material
    material h_ground = material::lambertian_material(d_ground_tex);
    checkCuda(cudaMalloc((void**)&d_ground, sizeof(material)) );
    checkCuda(cudaMemcpy(d_ground, &h_ground, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_ground);
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,-10.0, 0.0), 10, d_ground);
    h_hittables_list.push_back(hittable_obj);

    hittable_obj = hittable::make_sphere(glm::vec3(0.0, 10.0, 0.0), 10, d_ground);
    h_hittables_list.push_back(hittable_obj);

    
    
    size_t number_of_hittables = h_hittables_list.size();
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );
     

    // hittable_list h_world;
    hittable h_world = hittable::make_hittableList();
   
    
    if (!cam.isBvh){  //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }
   

}

void earth(Camera& cam, rtw_image* &d_rtw_image, std::vector<material*> device_materials, std::vector<texture*> device_textures,  std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(0.0f, 0.0f,  12.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
    
    

    
    std::vector<hittable> h_hittables_list;
    material* d_ground;
    texture* d_image_tex;
    
    // ground 
    //* Texture

    auto image = rtw_image("images/earth_map.jpg");
    unsigned char* d_bdata;
    // printf(" texture width: %d, height: %d\n", image.width(), image.height());
    
    checkCuda(cudaMalloc((void**)&d_bdata, image.width() * image.height() * image.pixelSize() * sizeof(unsigned char)) );

    checkCuda(cudaMemcpy(d_bdata, image.imageData(), image.width() * image.height() * image.pixelSize() * sizeof(unsigned char), cudaMemcpyHostToDevice) );

    
    texture h_image_tex = texture::image_texture(d_bdata, image.width(), image.height(), image.scanLineSize(), image.pixelSize());

    checkCuda(cudaMalloc((void**)&d_image_tex, sizeof(texture)) );
    checkCuda(cudaMemcpy(d_image_tex, &h_image_tex, sizeof(texture), cudaMemcpyHostToDevice) );
    device_textures.push_back(d_image_tex);

    material h_ground = material::lambertian_material(d_image_tex);
    checkCuda(cudaMalloc((void**)&d_ground, sizeof(material)) );
    checkCuda(cudaMemcpy(d_ground, &h_ground, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_ground);
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,0.0, 0.0), 2, d_ground);
    h_hittables_list.push_back(hittable_obj);

    

    
    
    size_t number_of_hittables = h_hittables_list.size();
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );
     

    auto h_world = hittable::make_hittableList();
    
    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }
   

}

void perlin_spheres(Camera& cam, rtw_image* &d_rtw_image, std::vector<material*> device_materials, std::vector<texture*> device_textures,  std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(13.0f, 2.0f,  3.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
    
    
    
    
    std::vector<hittable> h_hittables_list;
    material* d_noise_mat;
    texture* d_noise_tex;
    
    //* Texture


    Perlin noise;
    float scramble_frequency = 4.0f;  // default is 1.0f;
    texture h_noise_tex = texture::noise_texture(noise, scramble_frequency);

    checkCuda(cudaMalloc((void**)&d_noise_tex, sizeof(texture)) );
    checkCuda(cudaMemcpy(d_noise_tex, &h_noise_tex, sizeof(texture), cudaMemcpyHostToDevice) );
    device_textures.push_back(d_noise_tex);

    //* material
    material h_ground = material::lambertian_material(d_noise_tex);
    checkCuda(cudaMalloc((void**)&d_noise_mat, sizeof(material)) );
    checkCuda(cudaMemcpy(d_noise_mat, &h_ground, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_noise_mat);
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0, -1000.0, 0.0), 1000, d_noise_mat);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_sphere(glm::vec3(0.0,2.0, 0.0), 2, d_noise_mat);
    h_hittables_list.push_back(hittable_obj);

    
    size_t number_of_hittables = h_hittables_list.size();
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );
     

    hittable h_world = hittable::make_hittableList();
    
    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }
   

}

void quads(Camera& cam, rtw_image* &d_rtw_image, std::vector<material*> device_materials, std::vector<texture*> device_textures,  std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 80.0f;
    cam.lookfrom = glm::vec3( 0.0f, 0.0f,  9.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;
    
    /* material */
    material h_left_red = material::lambertian_material(glm::vec3(1.0, 0.2, 0.2));
    material* d_left_red;
    checkCuda(cudaMalloc((void**)&d_left_red, sizeof(material)) );
    checkCuda(cudaMemcpy(d_left_red, &h_left_red, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_left_red);

    material h_back_green = material::lambertian_material(glm::vec3(0.2, 1.0, 0.2));
    material* d_back_green;
    checkCuda(cudaMalloc((void**)&d_back_green, sizeof(material)) );
    checkCuda(cudaMemcpy(d_back_green, &h_back_green, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_back_green);

    material h_right_blue = material::lambertian_material(glm::vec3(0.2, 0.2, 1.0));
    material* d_right_blue;
    checkCuda(cudaMalloc((void**)&d_right_blue, sizeof(material)) );
    checkCuda(cudaMemcpy(d_right_blue, &h_right_blue, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_right_blue);

    material h_upper_orange = material::lambertian_material(glm::vec3(1.0, 0.5, 0.0));
    material* d_upper_orange;
    checkCuda(cudaMalloc((void**)&d_upper_orange, sizeof(material)) );
    checkCuda(cudaMemcpy(d_upper_orange, &h_upper_orange, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_upper_orange);

    material h_lower_teal = material::lambertian_material(glm::vec3(0.2, 0.8, 0.8));
    material* d_lower_teal;
    checkCuda(cudaMalloc((void**)&d_lower_teal, sizeof(material)) );
    checkCuda(cudaMemcpy(d_lower_teal, &h_lower_teal, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_lower_teal);

    /* Quads */
    hittable hittable_obj;
    hittable_obj = hittable::make_quad(glm::vec3(-3.0f, -2.0f, 5.0f), glm::vec3(0.0f, 0.0f, -4.0f), glm::vec3(0.0f, 4.0f,  0.0f), d_left_red);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3(-2.0f, -2.0f, 0.0f), glm::vec3(4.0f, 0.0f, -0.0f), glm::vec3(0.0f, 4.0f,  0.0f), d_back_green);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3( 3.0f, -2.0f, 1.0f), glm::vec3(0.0f, 0.0f,  4.0f), glm::vec3(0.0f, 4.0f,  0.0f), d_right_blue);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3(-2.0f,  3.0f, 1.0f), glm::vec3(4.0f, 0.0f, -0.0f), glm::vec3(0.0f, 0.0f,  4.0f), d_upper_orange);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3(-2.0f, -3.0f, 5.0f), glm::vec3(4.0f, 0.0f, -0.0f), glm::vec3(0.0f, 0.0f, -4.0f), d_lower_teal);
    h_hittables_list.push_back(hittable_obj);

    
    size_t number_of_hittables = h_hittables_list.size();
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );
     

    hittable h_world = hittable::make_hittableList();
    
    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }
    

}

void simple_light(Camera& cam, rtw_image* &d_rtw_image, std::vector<material*> device_materials, std::vector<texture*> device_textures,  std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3( 26.0f, 3.0f,  6.0f);
    cam.lookat   = glm::vec3( 0.0f, 2.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
   
    
    
    std::vector<hittable> h_hittables_list;

    hittable hittable_obj;  // holds any hittable temporarily

    /* texture */
    texture* d_noise_tex;
    Perlin noise;
    float scramble_frequency = 4.0f;  // default is 1.0f;
    texture h_noise_tex = texture::noise_texture(noise, scramble_frequency);
    checkCuda(cudaMalloc((void**)&d_noise_tex, sizeof(texture)) );
    checkCuda(cudaMemcpy(d_noise_tex, &h_noise_tex, sizeof(texture), cudaMemcpyHostToDevice) );
    device_textures.push_back(d_noise_tex);

    /* materials */
    material h_mat = material::lambertian_material(d_noise_tex);
    material* d_mat;
    checkCuda(cudaMalloc((void**)&d_mat, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat, &h_mat, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat);

    


    /* Spheres */
    auto hittable_obj2 = hittable::make_sphere(glm::vec3(0.0, -1000.0, 0.0), 1000, d_mat);
    h_hittables_list.push_back(hittable_obj2);

    auto hittable_obj3 = hittable::make_sphere(glm::vec3(0.0, 2.0, 0.0), 2, d_mat);
    h_hittables_list.push_back(hittable_obj3);

    /* Quads */
    material* d_mat1;
    auto h_mat1 = material::diffuseLight_material(glm::vec3(4.0f, 4.0f, 4.0f));
    checkCuda(cudaMalloc((void**)&d_mat1, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat1, &h_mat1, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat1);

    auto hittable_obj4 = hittable::make_quad(glm::vec3( 3.0f,  1.0f, -2.0f), glm::vec3(2.0f, 0.0f, -0.0f), glm::vec3(0.0f, 2.0f,  0.0f), d_mat1);
    h_hittables_list.push_back(hittable_obj4);

    material* d_mat2;
    auto h_mat2 = material::diffuseLight_material(glm::vec3(6.0f, 0.0f, 0.2f));
    checkCuda(cudaMalloc((void**)&d_mat2, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat2, &h_mat2, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat2);

    hittable_obj4 = hittable::make_sphere(glm::vec3(-5.0f, 3.0f, 3.0f), .5, d_mat2);
    h_hittables_list.push_back(hittable_obj4);

    
    size_t number_of_hittables = h_hittables_list.size();
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );
     

    hittable h_world = hittable::make_hittableList();
    
    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }
    

}


void cornell_box(Camera& cam, rtw_image* &d_rtw_image, std::vector<material*> device_materials, std::vector<texture*> device_textures,  std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 278.0f, 278.0f, -800.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
    
    std::vector<hittable> h_hittables_list;

    hittable hittable_obj;  // holds any hittable temporarily

    

    auto red   = material::lambertian_material(glm::vec3(.65, .05, .05));
    auto white = material::lambertian_material(glm::vec3(.73, .73, .73));
    auto green = material::lambertian_material(glm::vec3(.12, .45, .15));
    auto light = material::diffuseLight_material(glm::vec3(15, 15, 15));

    material* d_red;  
    material* d_white;
    material* d_green;
    material* d_light;

    checkCuda(cudaMalloc((void**)&d_red, sizeof(material)) );
    checkCuda(cudaMemcpy(d_red, &red, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_red);

    checkCuda(cudaMalloc((void**)&d_white, sizeof(material)) );
    checkCuda(cudaMemcpy(d_white, &white, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_white);

    checkCuda(cudaMalloc((void**)&d_green, sizeof(material)) );
    checkCuda(cudaMemcpy(d_green, &green, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_green);

    checkCuda(cudaMalloc((void**)&d_light, sizeof(material)) );
    checkCuda(cudaMemcpy(d_light, &light, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_light);



    auto obj1 = hittable(hittable::make_quad(glm::vec3(555,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), d_green));
    auto obj2 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), d_red));
    auto obj3 = hittable(hittable::make_quad(glm::vec3(343, 554, 332), glm::vec3(-130,0,0), glm::vec3(0,0,-105), d_light));
    auto obj4 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(555,0,0), glm::vec3(0,0,555), d_white));
    auto obj5 = hittable(hittable::make_quad(glm::vec3(555,555,555), glm::vec3(-555,0,0), glm::vec3(0,0,-555), d_white));
    auto obj6 = hittable(hittable::make_quad(glm::vec3(0,0,555), glm::vec3(555,0,0), glm::vec3(0,555,0), d_white));
   
    h_hittables_list.push_back(obj1);
    h_hittables_list.push_back(obj2);
    h_hittables_list.push_back(obj3);
    h_hittables_list.push_back(obj4);
    h_hittables_list.push_back(obj5);
    h_hittables_list.push_back(obj6);
    
    //* Create two boxes
    box(h_hittables_list, glm::vec3(130.0f, 0.0f, 65.0f),  glm::vec3(295.0f, 165.0f, 230.0f), d_white);
    box(h_hittables_list, glm::vec3(265.0f, 0.0f, 295.0f), glm::vec3(430.0f, 330.0f, 460.0f), d_white);



    
    size_t number_of_hittables = h_hittables_list.size();
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );
     

    hittable h_world = hittable::make_hittableList();
    
    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }
    

}

void cornell_box_instances(Camera& cam, rtw_image* &d_rtw_image, std::vector<material*> device_materials, std::vector<texture*> device_textures, std::vector<hittable*>  allocated_hittables, std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 278.0f, 278.0f, -800.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;
    

    hittable hittable_obj;  // holds any hittable temporarily

    

    auto red   = material::lambertian_material(glm::vec3(.65, .05, .05));
    auto white = material::lambertian_material(glm::vec3(.73, .73, .73));
    auto green = material::lambertian_material(glm::vec3(.12, .45, .15));
    auto light = material::diffuseLight_material(glm::vec3(15, 15, 15));

    material* d_red;  
    material* d_white;
    material* d_green;
    material* d_light;

    checkCuda(cudaMalloc((void**)&d_red, sizeof(material)) );
    checkCuda(cudaMemcpy(d_red, &red, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_red);

    checkCuda(cudaMalloc((void**)&d_white, sizeof(material)) );
    checkCuda(cudaMemcpy(d_white, &white, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_white);

    checkCuda(cudaMalloc((void**)&d_green, sizeof(material)) );
    checkCuda(cudaMemcpy(d_green, &green, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_green);

    checkCuda(cudaMalloc((void**)&d_light, sizeof(material)) );
    checkCuda(cudaMemcpy(d_light, &light, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_light);



    auto obj1 = hittable(hittable::make_quad(glm::vec3(555,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), d_green));
    auto obj2 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), d_red));
    auto obj3 = hittable(hittable::make_quad(glm::vec3(343, 554, 332), glm::vec3(-130,0,0), glm::vec3(0,0,-105), d_light));
    auto obj4 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(555,0,0), glm::vec3(0,0,555), d_white));
    auto obj5 = hittable(hittable::make_quad(glm::vec3(555,555,555), glm::vec3(-555,0,0), glm::vec3(0,0,-555), d_white));
    auto obj6 = hittable(hittable::make_quad(glm::vec3(0,0,555), glm::vec3(555,0,0), glm::vec3(0,555,0), d_white));
   
    h_hittables_list.push_back(obj1);
    h_hittables_list.push_back(obj2);
    h_hittables_list.push_back(obj3);
    h_hittables_list.push_back(obj4);
    h_hittables_list.push_back(obj5);
    h_hittables_list.push_back(obj6);

    
    std::vector<hittable> box1, box2, transform_box1, transform_box2;
    

    //* Create two boxes
    box(box1, glm::vec3(.0f, 0.0f, 0.0f),  glm::vec3(165.0f, 330.0f, 165.0f), d_white);
    box(box2, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(165.0f, 165.0f, 165.0f), d_white);

    // // transform_box.clear();
    // for (auto& side : box1){
    //     auto s = hittable(hittable::make_rotateY(&side, 15));
    //     transform_box1.push_back(s);
    //     // h_hittables_list.push_back(s);
    // }
    // for (auto& side : transform_box1){
    //     auto s = hittable(hittable::make_translate(&side, glm::vec3(265.0f, 0.0f, 295.0f)));
    //     h_hittables_list.push_back(s);
    // }
    // // transform_box.clear();
    // for (auto& side : box2){
    //     auto s = hittable(hittable::make_rotateY(&side, -18));
    //     transform_box2.push_back(s);
    //     // h_hittables_list.push_back(s);
    // }
    // for (auto& side : transform_box2){
    //     auto s = hittable(hittable::make_translate(&side, glm::vec3(130.0f, 0.0f, 65.0f)));
    //     h_hittables_list.push_back(s);
    // }
    

    // transform_box.clear();
    for (auto& side : box1){
        auto rotated = new hittable(hittable::make_rotateY(&side, 15));
        auto translated = new hittable(hittable::make_translate(rotated, glm::vec3(265.0f, 0.0f, 295.0f)));
        h_hittables_list.push_back(*translated);  
        allocated_hittables.push_back(rotated);
        allocated_hittables.push_back(translated);
    }
    
    // transform_box.clear();
    for (auto& side : box2){
        auto rotated = new hittable(hittable(hittable::make_rotateY(&side, -18)));
        auto translated = new hittable(hittable(hittable::make_translate(rotated, glm::vec3(130.0f, 0.0f, 65.0f))));
        h_hittables_list.push_back(*translated);
        allocated_hittables.push_back(rotated);
        allocated_hittables.push_back(translated);
    }
   
    size_t number_of_hittables = h_hittables_list.size();
    printf("size: %d\n", (int)number_of_hittables);
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );

    hittable h_world = hittable::make_hittableList();
    
    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }

    
    

}

void cornell_smoke(Camera& cam, rtw_image* &d_rtw_image, std::vector<material*> device_materials, std::vector<texture*> device_textures, std::vector<hittable*>  allocated_hittables, std::vector<BVHNode*> allocated_flat_nodes, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 278.0f, 278.0f, -800.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;
    

    hittable hittable_obj;  // holds any hittable temporarily

    auto atex = texture::solid_texture(glm::vec3(0.0f, 0.0f, 0.0f));
    texture* d_atex;
    checkCuda(cudaMalloc(&d_atex, sizeof(texture)));
    checkCuda(cudaMemcpy(d_atex, &atex, sizeof(texture), cudaMemcpyHostToDevice));
    auto negro = material::isotropic_material(d_atex);
    device_textures.push_back(d_atex);
    material* d_negro;
    checkCuda(cudaMalloc((void**)&d_negro, sizeof(material)) );
    checkCuda(cudaMemcpy(d_negro, &negro, sizeof(material), cudaMemcpyHostToDevice) );



    auto red   = material::lambertian_material(glm::vec3(.65, .05, .05));
    auto white = material::lambertian_material(glm::vec3(.73, .73, .73));
    auto green = material::lambertian_material(glm::vec3(.12, .45, .15));
    auto light = material::diffuseLight_material(glm::vec3(7, 7, 7));

    material* d_red;  
    material* d_white;
    material* d_green;
    material* d_light;

    checkCuda(cudaMalloc((void**)&d_red, sizeof(material)) );
    checkCuda(cudaMemcpy(d_red, &red, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_red);

    checkCuda(cudaMalloc((void**)&d_white, sizeof(material)) );
    checkCuda(cudaMemcpy(d_white, &white, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_white);

    checkCuda(cudaMalloc((void**)&d_green, sizeof(material)) );
    checkCuda(cudaMemcpy(d_green, &green, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_green);

    checkCuda(cudaMalloc((void**)&d_light, sizeof(material)) );
    checkCuda(cudaMemcpy(d_light, &light, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_light);



    auto obj1 = hittable(hittable::make_quad(glm::vec3(555,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), d_green));
    auto obj2 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), d_red));
    auto obj3 = hittable(hittable::make_quad(glm::vec3(113, 554, 127), glm::vec3(330,0,0), glm::vec3(0,0, 305), d_light));
    auto obj4 = hittable(hittable::make_quad(glm::vec3(0,555,0), glm::vec3(555,0,0), glm::vec3(0,0,555), d_white));
    auto obj5 = hittable(hittable::make_quad(glm::vec3(0 ,0 , 0), glm::vec3(555,0,0), glm::vec3(0,0,555), d_white));
    auto obj6 = hittable(hittable::make_quad(glm::vec3(0,0,555), glm::vec3(555,0,0), glm::vec3(0,555,0), d_white));
   
    h_hittables_list.push_back(obj1);
    h_hittables_list.push_back(obj2);
    h_hittables_list.push_back(obj3);
    h_hittables_list.push_back(obj4);
    h_hittables_list.push_back(obj5);
    h_hittables_list.push_back(obj6);

    
    std::vector<hittable> box1, box2;

    

    //* Create two boxes
    box(box1, glm::vec3(.0f, 0.0f, 0.0f),  glm::vec3(165.0f, 330.0f, 165.0f), d_white);
    // box(box2, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(165.0f, 165.0f, 165.0f), d_white);

  
   
    for (auto& side : box1){
        // auto rotated    = new hittable(hittable::make_rotateY(&side, 15));
        // auto translated = new hittable(hittable::make_translate(rotated, glm::vec3(265.0f, 0.0f, 295.0f)));
        // auto smoked     = new hittable(hittable::make_constantMedium(translated, 0.01f, glm::vec3(0.0f, 0.0f, 0.0f)));
        auto smoked     = new hittable(hittable::make_constantMedium(&side, 0.01f, d_negro));
        box2.push_back(*smoked);  
        // allocated_hittables.push_back(rotated);
        // allocated_hittables.push_back(translated);
        allocated_hittables.push_back(smoked);
    }

    h_hittables_list.insert(h_hittables_list.begin(), box2.begin(), box2.end());



    
   
    // for (auto& side : box2){
    //     auto rotated = new hittable(hittable(hittable::make_rotateY(&side, -18)));
    //     auto translated = new hittable(hittable(hittable::make_translate(rotated, glm::vec3(130.0f, 0.0f, 65.0f))));
    //     auto smoked     = new hittable(hittable::make_constantMedium(translated, 0.01f, glm::vec3(1.0f, 1.0f, 1.0f)));
    //     h_hittables_list.push_back(*smoked);
    //     allocated_hittables.push_back(rotated);
    //     allocated_hittables.push_back(translated);
    // }




   
    size_t number_of_hittables = h_hittables_list.size();
    printf("size: %d\n", (int)number_of_hittables);
  
    checkCuda(cudaMalloc((void**)&d_hittable_list, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_hittable_list, h_hittables_list.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );

    hittable h_world = hittable::make_hittableList();
    
    if (!cam.isBvh){ //* No bboxes
        h_world.hittableList.setList(d_hittable_list, number_of_hittables);
        /* Allocate memory for hittable list on the device */
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );
    } else {
        /** Implementing ROPE based BHV nodes ind cuda */
        int number_of_nodes = (2 * number_of_hittables -1);
        checkCuda(cudaMalloc((void**)&bvh_nodes, number_of_nodes * sizeof(BVHNode)) );
        build_bvh_NR_ROPE8<<<1, 1>>>(bvh_nodes, d_hittable_list, number_of_hittables);
        h_world.hittableList.setNodes(bvh_nodes, d_hittable_list);
        checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable)) );
        checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable), cudaMemcpyHostToDevice) );  // copy host world to device world
    }

    
    

}

void RayTracer::cudaCall(Camera &cam, uint32_t *colorBuffer)
{


    cudaSetDevice(0);
    // Get the current stack size limit
    size_t currentSize;
    checkCuda(cudaDeviceGetLimit(&currentSize, cudaLimitStackSize));
    printf("Current Stack Size: %d bytes\n", (int)currentSize);

    size_t newSize;
    if (currentSize <= 1024) { 
        newSize = currentSize + 1024;
        checkCuda(cudaDeviceSetLimit(cudaLimitStackSize, newSize));
         printf("New Stack Size: %d bytes\n", (int)newSize);
    }


    


    std::vector<material*>  device_materials;  
    std::vector<texture*>   device_textures;
    std::vector<BVHNode*>   allocated_flat_nodes; 
    std::vector<hittable*>  allocated_hittables;
    hittable*               d_hittables_list;

    Camera* d_cam;
    rtw_image* d_rtw_image;

    /* device variables */
    uint32_t*   d_image;    // for display buffer
    curandState_t* d_states;  // random calculations in GPU
    
    // hittable_list* d_world;
    hittable* d_world;
    // hittable* d_world;
    BVHNode* bvh_nodes;
    
      
    switch (cam.scene)  
    {
    case 1:
        
        bouncing_spheres(cam, device_materials, device_textures, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 2:
        checkered_spheres(cam, device_materials, device_textures, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 3:
        earth(cam, d_rtw_image, device_materials, device_textures, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 4:
        perlin_spheres(cam, d_rtw_image, device_materials, device_textures, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 5:
        quads(cam, d_rtw_image, device_materials, device_textures, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 6:
        simple_light(cam, d_rtw_image, device_materials, device_textures, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 7:
        cornell_box(cam, d_rtw_image, device_materials, device_textures, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 8:
        cornell_box_instances(cam, d_rtw_image, device_materials, device_textures, allocated_hittables, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    case 9:
        cornell_smoke(cam, d_rtw_image, device_materials, device_textures, allocated_hittables, allocated_flat_nodes, bvh_nodes, d_hittables_list, d_world);
        break;
    default:
        break;
    }
    
    
    checkCuda(cudaMalloc((void**)&d_image, cam.image_width * cam.image_height * sizeof(uint32_t)));
    checkCuda(cudaMalloc((void**)&d_cam,  sizeof(Camera)));
    checkCuda(cudaMemcpy(d_cam, &cam, sizeof(Camera), cudaMemcpyHostToDevice));
    
    
    clock_t start, stop;
    start = clock();

    int threads = 8;
    dim3 blockSize(threads, threads);
    int blocks_x = (cam.image_width  + blockSize.x - 1) / blockSize.x;
    int blocks_y = (cam.image_height  + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(blocks_x, blocks_y);



    //generate random seed to be used in rayTracer kernel
    int num_threads = threads * threads * blocks_x * blocks_y;
    
    checkCuda(cudaMalloc(&d_states, num_threads * sizeof(curandState_t)));
    init_random<<<gridSize, blockSize>>>(seed, d_states);
    // checkCuda(cudaDeviceSynchronize() );
 
    
    size_t shared_memory_size =  blockSize.x * blockSize.y * MAX_STACK_SIZE * sizeof(int);
    if(cam.isBvh)
        rayTracer_nodes_kernel<<<gridSize, blockSize, shared_memory_size>>>(d_states, d_cam, d_image, d_world);
        // rayTracer_kernel<<<gridSize, blockSize, shared_memory_size>>>(d_states, d_cam, d_image, d_world, bvh_nodes, d_hittables_list);
    else
        rayTracer_kernel<<<gridSize, blockSize>>>(d_states, d_cam, d_image, d_world);
    
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Took %f seconds with %d samples per pixel and %d max depth\n", timer_seconds, cam.samples_per_pixel, cam.max_depth);

    checkCuda(cudaMemcpy(colorBuffer, d_image, cam.image_width * cam.image_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // delete all device pointers of the materials and AaBb boxes
    
    for (auto nodes : allocated_flat_nodes)
        delete nodes;

    for (auto alloc : allocated_hittables) 
        delete alloc;
    
    for(auto& device : device_materials) 
        checkCuda(cudaFree(device) );
    for(auto& device : device_textures) 
        checkCuda(cudaFree(device) );
    
    checkCuda(cudaFree(d_hittables_list) );
    checkCuda(cudaFree(d_image) );
    
    checkCuda(cudaFree(d_cam) );
    if(cam.isBvh) {
        checkCuda(cudaFree(bvh_nodes) );
    }else {
        // checkCuda(cudaFree(d_world) );
        checkCuda(cudaFree(d_world))
    }
    checkCuda(cudaFree(d_states) );
    
    
}
