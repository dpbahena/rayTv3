#include "cuda_call.h"
#include "ray.h"
#include "interval.h"
#include <cstdio>
#include <curand_kernel.h>
#include <unistd.h>







#define checkCuda(result) { gpuAssert((result), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(code == cudaSuccess);
   }
}


enum MaterialType {LAMBERTIAN, METAL, DIELECTRIC};
enum HittableType {SPHERE, LIST, BBOX};
struct material;
struct hittable;
struct hittable_list;

__device__ inline glm::vec3 random_on_hemisphere(curandState_t* states,  int i, int j,const glm::vec3& normal);
__device__ inline glm::vec3 random_in_unit_sphere(curandState_t* states,  int i, int j);
__device__ inline glm::vec3 random_vector_in_range(curandState_t* states,  int i, int j, float min, float max);
__device__ inline glm::vec3 random_vector(curandState_t* states,  int i, int j);
__device__ inline float random_float_in_range(curandState_t* state, float a, float b);
__device__ inline glm::vec3 random_unit_vector(curandState_t* states, int i, int j);
__device__ inline bool near_zero(const glm::vec3 v);
__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n);
__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat);
__device__ inline float reflectance(float cosine, float refraction_index);
__device__ inline float random_float(curandState_t* state);
__device__ glm::vec3 random_in_unit_disk(curandState_t* states,  int i, int j);
__device__ glm::vec3 defocus_disk_sample(curandState_t* states,  int i, int j, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v);
__device__ ray get_ray(curandState_t* states, int &i, int &j, glm::vec3& pixel00_loc, glm::vec3& cameraCenter, glm::vec3& delta_u, glm::vec3& delta_v);
__device__ glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const ray &r, const hittable_list& world);







struct hitRecord {
    glm::vec3 p;
    glm::vec3 normal;
    sphere* sphere_ptr;
    material* mat_ptr;
    float t;
    bool front_face;

    __host__ __device__
    void set_face_normal(const ray& r, const glm::vec3& outward_normal) {
        front_face = glm::dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct hittable {
    // bool (*hit_funct)(const void*, const ray&, interval, hitRecord&);
    // void* obj;   // generic pointer to the derived class instance
    // // const sphere* self;

    // __device__
    // bool hit(const ray& r, interval ray_t, hitRecord& rec) const {
    //     printf("Ari\n");
    //     return hit_funct(obj, r, ray_t, rec);  //self  is just a name..you can change is for obj or ana..or whatever
    // }

     bool (*hit)(const ray&, interval, hitRecord&);
     HittableType type;

   
};






class sphere {
    public:
        
        __host__ __device__
        sphere() {}

        __host__ __device__
        sphere(const glm::vec3& center, float radius, material* mat) : center(center), radius(radius), mat(mat){
            
            base.hit = (bool (*)(const ray& r, interval ray_t, hitRecord& rec))hit;
            base.type = SPHERE;
            // base.obj = this; // set this object pointer to the instance of the sphere
            // base.hit_funct = &sphere::hit;  // assign the member function
        }

        __host__ __device__
        static bool hit(sphere* self, const ray& r, interval ray_t, hitRecord &rec)  {
            // auto s = static_cast<const sphere*>(self);
            auto s = reinterpret_cast<sphere*>(self);
            printf("Ariana\n");
            glm::vec3 oc = r.origin - self->center;
            auto a = glm::dot(r.direction, r.direction);
            auto h = glm::dot(oc, r.direction);
            auto c = glm::dot(oc, oc) - self->radius * self->radius;

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
            glm::vec3 outward_normal = (rec.p - self->center) / self->radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = self->mat;
            rec.sphere_ptr = self;
            
            
            
            
            return true;
        }
        
    // private:
        hittable base;  // base structure
        
         
        glm::vec3 center;
        float radius;
        material* mat;
};





// class sphere {
//         public:
//             __host__ __device__
//             sphere() {}
//             __host__ __device__
//             sphere(const glm::vec3& center, float radius, material* mat) : center(center), radius(radius), mat(mat){}
//             __host__ __device__
//             bool hit(const ray& r, interval ray_t, hitRecord &rec) const  {
//                 glm::vec3 oc = r.origin - center;
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
//                 glm::vec3 outward_normal = (rec.p - center) / radius;
//                 rec.set_face_normal(r, outward_normal);
//                 rec.mat_ptr = mat;
                
//                 return true;
//             }

//         private:
//             glm::vec3 center;
//             float radius;
//             material* mat;
//     };

struct alignas(16) material {
    public:
        bool (*scatter)(const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j);
        MaterialType type;
    
    
};



struct hittable_list {
    sphere* list;
    int list_size;
};



struct lambertian {
    public:
        material base;
        glm::vec3 albedo;

        __device__ 
        static bool scatter(const lambertian* self, const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j){
            auto scatter_direction = rec.normal + random_unit_vector(states,  i, j);
            
            // Catch degenerate scatter direction
            if (near_zero(scatter_direction)) scatter_direction = rec.normal;
            scattered = ray(rec.p, scatter_direction, r_in.time());
            attenuation = self->albedo;  // Access the albedo using itself
           

            return true;
        }

        __device__ __host__
        lambertian(const glm::vec3& a) : albedo(a) {
            base.scatter = (bool (*)(const ray&, const hitRecord&, glm::vec3&, ray&, curandState_t*, int, int))scatter;
            base.type = LAMBERTIAN;
            
        }
        
};  

struct metal {
   public:
      material base;
      glm::vec3 albedo;
      float fuzz;

      __device__ 
      static bool scatter(const metal* self, const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j) {
            glm::vec3 reflected = reflect(r_in.direction, rec.normal);
            reflected = glm::normalize(reflected) + (static_cast<float>(self->fuzz) * random_unit_vector(states, i, j));
            scattered = ray(rec.p, reflected, r_in.time());
            attenuation = self->albedo;
            
            return (glm::dot(scattered.direction, rec.normal) > 0.0f);
      }

      __device__ __host__
      metal(const glm::vec3& a, float fuzz) : albedo(a), fuzz(fuzz < 1.0f ? fuzz : 1.0f) {
            base.scatter = (bool (*)(const ray&, const hitRecord&, glm::vec3&, ray&, curandState_t*, int, int))scatter;
            base.type = METAL;
      }
};

struct dielectric {
   public:
      material base;
      float refraction_index;
      /* Refractive index in vacumm or air, or the ratio of the material's refractive index over the refractive index of the enclosing media */

      __device__ 
      static bool scatter(const dielectric* self, const ray& r_in, const hitRecord& rec, glm::vec3& attenuation, ray& scattered, curandState_t* states,  int i, int j) {
            attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
            float ri = rec.front_face ? (1.0f / self->refraction_index) : self->refraction_index;
            glm::vec3 unit_direction = glm::normalize(r_in.direction);
            float cos_theta = fmin(glm::dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0f;
            glm::vec3 direction;
            curandState_t x = states[i];  // for random data

            if (cannot_refract || reflectance(cos_theta, ri) > random_float(&x) )
               direction = reflect(unit_direction, rec.normal);
            else   
               direction = refract(unit_direction, rec.normal, ri);
            
            states[i] = x; // save back value
            scattered  = ray(rec.p, direction, r_in.time());
            return true;
      }

      __device__ __host__
      dielectric(float r_i) : refraction_index(r_i) {
            base.scatter = (bool (*)(const ray&, const hitRecord&, glm::vec3&, ray&, curandState_t*, int, int))scatter;
            base.type = DIELECTRIC;
      }
};


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
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow( (1 - cosine), 5);
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


__device__ bool hit(const hittable_list& world, const ray& r, interval ray_t, hitRecord& rec) {
    hitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
    // printf("hi, %d\n", world.list_size);
    for (int i = 0; i < world.list_size; i++) {
        // auto sphere_ptr = reinterpret_cast<const sphere*>( world.list[i].base.self);
        // printf("hola\n");
        // auto sphere_ptr = reinterpret_cast<const sphere*>(rec.sphere_ptr);
        // printf("hola\n");
        auto sphere_ptr = reinterpret_cast <const sphere*>(&world.list[i]);

        // if (world.list[i].base.hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
        // if (sphere::hit(sphere_ptr, interval(ray_t.min, closest_so_far), temp_rec)) {
        if (sphere::hit(sphere_ptr, r, ray_t, rec))
            printf("senor\n");
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}


__device__
glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const ray &r, const hittable_list& world) {
    ray cur_ray = r;
    glm::vec3 cur_attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 final_color     = glm::vec3(0.0f, 0.0f, 0.0f);
    
    
    for (int k = 0; k < depth; k++){
        hitRecord rec;
        if(hit(world, cur_ray, interval(0.001f, FLT_MAX), rec)){
            auto dir = rec.normal + random_unit_vector(state, i, j); // first approach using Lambertian  reflection
            ray scattered;
            glm::vec3 attenuation;

            bool did_scattter = false;
        
            if (rec.mat_ptr->type == METAL){
                auto metal_ptr = reinterpret_cast<const metal*>(rec.mat_ptr);
                did_scattter =  metal::scatter(metal_ptr, cur_ray, rec, attenuation, scattered, state, i, j);

            } else if (rec.mat_ptr->type == LAMBERTIAN){
                auto lamberian_ptr = reinterpret_cast<const lambertian*>(rec.mat_ptr);
                did_scattter = lambertian::scatter(lamberian_ptr, cur_ray, rec, attenuation, scattered, state, i, j);

            } else if (rec.mat_ptr->type == DIELECTRIC){
                auto dielectric_ptr = reinterpret_cast<const dielectric*>(rec.mat_ptr);
                did_scattter = dielectric::scatter(dielectric_ptr, cur_ray, rec, attenuation, scattered, state, i, j);
            }

            if (did_scattter){
                cur_ray = scattered;
                cur_attenuation *= attenuation;
            } else {
                final_color = glm::vec3(0.0f, 0.0f, 0.0f);  // if no scattering, no contribution
            }
        } else {  // color background
            glm::vec3 unitDirection = glm::normalize(cur_ray.direction );
            float a = 0.5f * (unitDirection.y + 1.0f);
            glm::vec3 background =  glm::vec3(1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + glm::vec3(a) * glm::vec3(0.5f, 0.7f, 1.0f);
            final_color =  cur_attenuation * background;
            break;
        }
    }
    
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
    return ray(ray_origin, ray_direction);
}


__global__ void init_random(unsigned int seed, curandState_t* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// __global__ void rayTracer_kernel(curandState_t* states, int depth, int width, int height, glm::vec3 cameraCenter, float defocusAngle, vec3 defocusDisk_u, vec3 defocusDisk_v, glm::vec3 pixel00, glm::vec3 delta_u, glm::vec3 delta_v, int samples_per_pixel, uint32_t* image, hittable_list* world) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= width || j >= height) return;
    
//     glm::vec3 color = {0.0, 0.0, 0.0};
//     for (int sample = 0; sample < samples_per_pixel; sample++){
//         ray r = get_ray(states, i, j, pixel00, cameraCenter, delta_u, delta_v, defocusAngle, defocusDisk_u, defocusDisk_v);
//         color  += ray_color(states, i, j, depth, r, *world);
//     }
//     float pixel_sample_scale = 1.0f / static_cast<float>(samples_per_pixel); // color scale factor for a sume of pixel samples
//     color *= pixel_sample_scale;
//     image[width * j + i] = colorToUint32_t(color);  
// }

__global__ void rayTracer_kernel(curandState_t* states, int depth, int width, int height, glm::vec3 cameraCenter, glm::vec3 pixel00, glm::vec3 delta_u, glm::vec3 delta_v, int samples_per_pixel, float defocusAngle, glm::vec3 defocusDisk_u, glm::vec3 defocusDisk_v, uint32_t* image, hittable_list* world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    
    glm::vec3 color = {0.0f, 0.0f, 0.0f};
    for (int sample = 0; sample < samples_per_pixel; sample++){
        ray r = get_ray(states, i, j, pixel00, cameraCenter, delta_u, delta_v, defocusAngle, defocusDisk_u, defocusDisk_v);
        color  += ray_color(states, i, j, depth, r, *world);
    }
    float pixel_sample_scale = 1.0f / static_cast<float>(samples_per_pixel); // color scale factor for a sume of pixel samples
    color *= pixel_sample_scale;
    image[width * j + i] = colorToUint32_t(color);  
}

void init_objects(lambertian* &ground, lambertian* &center, dielectric* &left, dielectric* &bubble, metal* &right, sphere* &spheres, hittable_list* &world){

    // allocate memory on the host
    lambertian  h_material_ground(glm::vec3(0.8f, 0.8f, 0.0f) );
    lambertian  h_material_center(glm::vec3(0.1f, 0.2f, 0.5f) );
    dielectric  h_material_left(1.50f);  //glass
    dielectric  h_material_bubble(1.00f / 1.50f);
    metal       h_material_right(glm::vec3(0.8f, 0.6f, 0.2f), 1.0f ); // fuzz 

    // ALLOCATE
    checkCuda(cudaMalloc((void**)&ground, sizeof(lambertian)) );
    checkCuda(cudaMalloc((void**)&center, sizeof(lambertian)) );
    checkCuda(cudaMalloc((void**)&left, sizeof(dielectric)) );
    checkCuda(cudaMalloc((void**)&bubble, sizeof(dielectric)) );
    checkCuda(cudaMalloc((void**)&right, sizeof(metal)) );

    checkCuda(cudaMemcpy(ground, &h_material_ground, sizeof(lambertian), cudaMemcpyHostToDevice) );
    checkCuda(cudaMemcpy(center, &h_material_center, sizeof(lambertian), cudaMemcpyHostToDevice) );
    checkCuda(cudaMemcpy(left, &h_material_left, sizeof(dielectric), cudaMemcpyHostToDevice) );
    checkCuda(cudaMemcpy(bubble, &h_material_bubble, sizeof(dielectric), cudaMemcpyHostToDevice) );
    checkCuda(cudaMemcpy(right, &h_material_right, sizeof(metal), cudaMemcpyHostToDevice) );

    // INCLUDE MATERIAL IN HITTABLE OBJECTS
    sphere h_spheres[] = {
            
            sphere(glm::vec3( 0.0f, -100.5f, -1.0f), 100.0f, &ground->base),
            sphere(glm::vec3( 0.0f,  0.0f,   -1.2f), 0.5f,   &center->base),
            sphere(glm::vec3(-1.0f,  0.0f,   -1.0f), 0.5f,   &left->base),
            sphere(glm::vec3(-1.0f,  0.0f,   -1.0f), 0.4f,   &bubble->base),
            sphere(glm::vec3( 1.0f,  0.0f,   -1.0f), 0.5f,   &right->base)
    };

    

    int number_hittables = 5;

    
    checkCuda(cudaMalloc((void**)&spheres, number_hittables * sizeof(sphere)));
    checkCuda(cudaMemcpy(spheres, h_spheres, number_hittables * sizeof(sphere), cudaMemcpyHostToDevice) );

    // create a hittable list
    hittable_list h_world;
    h_world.list = spheres;
    h_world.list_size = number_hittables;

    // Allocate memory for hittable list on the device
    
    checkCuda(cudaMalloc((void**)&world, sizeof(hittable_list)));
    checkCuda(cudaMemcpy(world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice) );
}


void RayTracer::cudaCall(int image_width, int image_height, int max_depth,  glm::vec3 center, glm::vec3 pixel00_loc, glm::vec3 pixel_delta_u, glm::vec3 pixel_delta_v, int samples_per_pixel, float& defocusAngle, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v, uint32_t* colorBuffer)
{  

//    // allocate memory on the host
//    lambertian  h_material_ground(glm::vec3(0.8f, 0.8f, 0.0f) );
//    lambertian  h_material_center(glm::vec3(0.1f, 0.2f, 0.5f) );
//    dielectric  h_material_left(1.50f);  //glass
//    dielectric  h_material_bubble(1.00f / 1.50f);
//    metal       h_material_right(glm::vec3(0.8f, 0.6f, 0.2f), 1.0f ); // fuzz 

   // device variables
   uint32_t*   d_image;
   lambertian* d_material_ground;
   lambertian* d_material_center;
   dielectric* d_material_left;
   dielectric* d_material_bubble;
   metal*      d_material_right;

   curandState_t* d_states;
   sphere* d_spheres;
   hittable_list* d_world;
   

    // // ALLOCATE
    // checkCuda(cudaMalloc((void**)&d_material_ground, sizeof(lambertian)) );
    // checkCuda(cudaMalloc((void**)&d_material_center, sizeof(lambertian)) );
    // checkCuda(cudaMalloc((void**)&d_material_left, sizeof(dielectric)) );
    // checkCuda(cudaMalloc((void**)&d_material_bubble, sizeof(dielectric)) );
    // checkCuda(cudaMalloc((void**)&d_material_right, sizeof(metal)) );

    // checkCuda(cudaMemcpy(d_material_ground, &h_material_ground, sizeof(lambertian), cudaMemcpyHostToDevice) );
    // checkCuda(cudaMemcpy(d_material_center, &h_material_center, sizeof(lambertian), cudaMemcpyHostToDevice) );
    // checkCuda(cudaMemcpy(d_material_left, &h_material_left, sizeof(dielectric), cudaMemcpyHostToDevice) );
    // checkCuda(cudaMemcpy(d_material_bubble, &h_material_bubble, sizeof(dielectric), cudaMemcpyHostToDevice) );
    // checkCuda(cudaMemcpy(d_material_right, &h_material_right, sizeof(metal), cudaMemcpyHostToDevice) );

    // // INCLUDE MATERIAL IN HITTABLE OBJECTS
    // sphere h_spheres[] = {
            
    //         sphere(glm::vec3( 0.0f, -100.5f, -1.0f), 100.0f, &d_material_ground->base),
    //         sphere(glm::vec3( 0.0f,  0.0f,   -1.2f), 0.5f, &d_material_center->base),
    //         sphere(glm::vec3(-1.0f,  0.0f,   -1.0f), 0.5f, &d_material_left->base),
    //         sphere(glm::vec3(-1.0f,  0.0f,   -1.0f), 0.4f, &d_material_bubble->base),
    //         sphere(glm::vec3( 1.0f,  0.0f,   -1.0f), 0.5f, &d_material_right->base)
    // };

    

    // int number_hittables = 5;

    
    // checkCuda(cudaMalloc((void**)&d_spheres, number_hittables * sizeof(sphere)));
    // checkCuda(cudaMemcpy(d_spheres, h_spheres, number_hittables * sizeof(sphere), cudaMemcpyHostToDevice) );

    // // create a hittable list
    // hittable_list h_world;
    // h_world.list = d_spheres;
    // h_world.list_size = number_hittables;

    // // Allocate memory for hittable list on the device
    
    // checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable_list)));
    // checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice) );

    init_objects(d_material_ground, d_material_center, d_material_left, d_material_bubble, d_material_right, d_spheres, d_world);

    checkCuda(cudaMalloc((void**)&d_image, image_width * image_height * sizeof(uint32_t)));
    
    clock_t start, stop;
    start = clock();
    int threads = 16;
    dim3 blockSize(threads, threads);
    int blocks_x = (image_width + blockSize.x - 1) / blockSize.x;
    int blocks_y = (image_height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(blocks_x, blocks_y);

    //generate random seed to be used in rayTracer kernel
    int num_threads = threads * threads * blocks_x * blocks_y;
    
    checkCuda(cudaMalloc(&d_states, num_threads * sizeof(curandState_t)));
    init_random<<<gridSize, blockSize>>>(time(0) ^ getpid(), d_states);
    checkCuda(cudaDeviceSynchronize() );

    rayTracer_kernel<<<gridSize, blockSize>>>(d_states, max_depth, image_width, image_height, center, pixel00_loc, pixel_delta_u, pixel_delta_v, samples_per_pixel, defocusAngle, defocusDisk_u, defocusDisk_v, d_image, d_world);
    // checkCuda(cudaPeekAtLastError() );
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    // std::cerr << "took " << timer_seconds << " seconds with " << cam.samples_per_pixel << " samples per pixel and depth of " << cam.max_depth << "\n";
    printf("It took %f seconds with %d samples per pixel and %d max depth\n", timer_seconds, samples_per_pixel, max_depth );
    
    checkCuda(cudaMemcpy(colorBuffer, d_image, image_width * image_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_image);
    cudaFree(d_world);
    cudaFree(d_spheres);
    cudaFree(d_states);
    cudaFree(d_material_ground);
    cudaFree(d_material_left);
    cudaFree(d_material_right);
    cudaFree(d_material_center);
    cudaFree(d_material_bubble);
    
}