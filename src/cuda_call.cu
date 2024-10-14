#include "cuda_call.h"
#include "ray.h"
#include "interval.h"

#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"
// #include "aabb.h"
#include "bvh_node.h"

#include <cstdio>
#include <vector>
#include <random>
#include <chrono>


// struct hittable_list;

__device__ inline glm::vec3 random_on_hemisphere(curandState_t* states,  int i, int j,const glm::vec3& normal);
__device__ inline glm::vec3 random_in_unit_sphere(curandState_t* states,  int i, int j);
__device__ inline glm::vec3 random_vector_in_range(curandState_t* states,  int i, int j, float min, float max);
__device__ inline glm::vec3 random_vector(curandState_t* states,  int i, int j);
__device__ inline float random_float_in_range(curandState_t* state, float a, float b);
__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n);
__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat);
__device__ glm::vec3 random_in_unit_disk(curandState_t* states,  int i, int j);
__device__ glm::vec3 defocus_disk_sample(curandState_t* states,  int i, int j, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v);
__device__ ray get_ray(curandState_t* states, int &i, int &j, glm::vec3& pixel00_loc, glm::vec3& cameraCenter, glm::vec3& delta_u, glm::vec3& delta_v);
__device__ glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const ray &r, const hittable_list& world);



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



#define checkCuda(result) { gpuAssert((result), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(code == cudaSuccess);
   }
}


__device__
static bool lambertian_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scatter, lambertian_data& lambertian, curandState_t* states,  int i, int j) {
    auto scatter_direction = rec.normal + random_unit_vector(states,  i, j);

    // Catch degenerate scatter direction
    if (near_zero(scatter_direction))
        scatter_direction = rec.normal;

    scatter = ray(rec.p, scatter_direction, r_in.time());
    attenuation = lambertian.albedo;
    
    return true;
}

__device__
static bool metal_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scatter, metal_data& metal, curandState_t* states,  int i, int j) {
    glm::vec3 reflected = reflect(r_in.direction, rec.normal);
    reflected = glm::normalize(reflected) + (metal.fuzz * random_unit_vector(states,  i, j));
    scatter = ray(rec.p, reflected, r_in.time());
    attenuation = metal.albedo;
    
    return (glm::dot(scatter.direction, rec.normal) > 0);
}


__device__
static bool dielectric_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scatter, dielectric_data& dielectric, curandState_t* states,  int i, int j) {
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

    scatter = ray(rec.p, direction, r_in.time());
    return true;
}

// struct hittable_list {
//     sphere* list;
//     int list_size;
//     AaBb bbox;
//     hittable_list() {};
//     hittable_list(sphere* objects, int size) : list(objects), list_size(size) {
//         /* Update the bounding box incrementally as each new chidl is added */
        
//         if (list_size > 0) {
//             printf("before\n");
//             bbox = list[0].bounding_box();  // initialize first box
//             printf("after\n");
            
//         }
//         for (int i = 1; i < list_size; i++){    // then continue initializing the rest
//             bbox = AaBb(bbox, list[i].bounding_box());
//             // printf("ana\n");
//         }
        
//     }
    
// };

// struct hittable_list {
//     sphere* list;
//     int list_size;
//     AaBb bbox;

//     hittable_list() {}

//     hittable_list(sphere* objects, int size) : list(objects), list_size(size) {
//         // Ensure the list is not empty
//         if (list_size > 0) {
//             printf("Initializing bounding box for the first sphere\n");
//             AaBb first_bbox = list[0].bounding_box();  // Get the bounding box of the first object
            
//             // Validate the first bounding box
//             if (first_bbox.axis_interval(0).min > first_bbox.axis_interval(0).max ||
//                 first_bbox.axis_interval(1).min > first_bbox.axis_interval(1).max ||
//                 first_bbox.axis_interval(2).min > first_bbox.axis_interval(2).max) {
//                 printf("Error: Invalid bounding box for the first sphere\n");
//                 return;
//             }
            
//             bbox = first_bbox; // Initialize with the first bounding box
//             printf("First bounding box initialized\n");
//         }

//         for (int i = 1; i < list_size; i++) {
//             printf("Processing sphere %d\n", i);
//             AaBb current_bbox = list[i].bounding_box();

//             // Validate the current bounding box before using it
//             if (current_bbox.axis_interval(0).min > current_bbox.axis_interval(0).max ||
//                 current_bbox.axis_interval(1).min > current_bbox.axis_interval(1).max ||
//                 current_bbox.axis_interval(2).min > current_bbox.axis_interval(2).max) {
//                 printf("Error: Invalid bounding box for sphere %d\n", i);
//                 continue; // Skip this sphere if its bounding box is invalid
//             }

//             bbox = AaBb(bbox, current_bbox);
//         }
//     }
// };



 


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


/* __device__ bool hit(const hittable_list& world, const ray& r, interval ray_t, hitRecord& rec) {
    hitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_t.max;
    
    for (int i = 0; i < world.list_size; i++) {
        if (world.list[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
} */


__device__
glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const ray &r, const hittable_list& world) {
    ray cur_ray = r;
    glm::vec3 cur_attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 final_color     = glm::vec3(0.0f, 0.0f, 0.0f);
    
    
    for (int k = 0; k < depth; k++){
        hit_record rec;
        if(world.hit(cur_ray, interval(0.001f, FLT_MAX), rec)){
        
            auto dir = rec.normal + random_unit_vector(state, i, j); // first approach using Lambertian  reflection
            ray scattered;
            glm::vec3 attenuation;

            bool did_scatter = false;

            if (rec.mat->type == Type::METAL){
                did_scatter = metal_scatter(cur_ray, rec, attenuation, scattered, rec.mat->metal, state, i, j);

            } else if (rec.mat->type == Type::LAMBERTIAN){
                did_scatter = lambertian_scatter(cur_ray, rec, attenuation, scattered, rec.mat->lambertian, state, i, j);

            } else if (rec.mat->type == Type::DIELECTRIC){
                did_scatter = dielectric_scatter(cur_ray, rec, attenuation, scattered, rec.mat->dielectric, state, i, j);    
            }

            if (did_scatter){
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


__global__ void rayTracer_kernel(curandState_t* states, int depth, int width, int height, glm::vec3 cameraCenter, glm::vec3 pixel00, glm::vec3 delta_u, glm::vec3 delta_v, int samples_per_pixel, float defocusAngle, glm::vec3 defocusDisk_u, glm::vec3 defocusDisk_v, uint32_t* image, hittable_list* world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;
    // if (i == 0 and j == 0) {
    //     printf("min: %f, max: %f\n", world->list[0].sphere.bbox.axis_interval(1).min, world->list[0].sphere.bbox.axis_interval(1).max);
    // }

    glm::vec3 color = {0.0f, 0.0f, 0.0f};
    for (int sample = 0; sample < samples_per_pixel; sample++){
        ray r = get_ray(states, i, j, pixel00, cameraCenter, delta_u, delta_v, defocusAngle, defocusDisk_u, defocusDisk_v);
        color  += ray_color(states, i, j, depth, r, *world);
    }
    float pixel_sample_scale = 1.0f / static_cast<float>(samples_per_pixel); // color scale factor for a sume of pixel samples
    color *= pixel_sample_scale;
    image[width * j + i] = colorToUint32_t(color);  
}

void init_objects(std::vector<material*> device_materials, hittable* &d_spheres, BVHNode* &d_nodes, hittable_list* &d_world){

    std::vector<hittable> h_spheres;
    

    material h_ground = material::lambertian_material((glm::vec3(0.5, 0.5, 0.5)));
    material* d_ground;
    checkCuda(cudaMalloc((void**)&d_ground, sizeof(material)) );
    checkCuda(cudaMemcpy(d_ground, &h_ground, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_ground);
    h_spheres.push_back(hittable::make_sphere(glm::vec3(0.0,-1000.0, 0.0), 1000, d_ground));

    // // Create random spheres 
    // for (
    //     int a = -11; a < 11; a++) {
    //     for (int b = -11; b < 11; b++) {
    //         auto choose_material = random_double();
    //         glm::vec3 center(a + 0.9f * random_double(), 0.2f, b + 0.9f * random_double());
            
    //         if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
    //             if(choose_material < 0.8f) {
    //                 // difuse
    //                 glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
    //                 auto a_material = material::lambertian_material(albedo);
    //                 material* d_mat;
    //                 checkCuda(cudaMalloc((void**)&d_mat, sizeof(material)) );
    //                 checkCuda(cudaMemcpy(d_mat, &a_material, sizeof(material), cudaMemcpyHostToDevice) );
    //                 device_materials.push_back(d_mat);
    //                 glm::vec3 center2 = center + glm::vec3(0,random_double(0, 0.5), 0);
    //                 h_spheres.push_back(hittable::make_sphere(center, center2, 0.2f, d_mat));
    //                 // h_spheres.push_back(hittable::make_sphere(center, 0.2f, d_mat));

    //             }
    //             if(choose_material < 0.95f) {
    //                 // metal
    //                 glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
    //                 float fuzz = random_double(0.0f, 0.5f);
    //                 auto a_material = material::metal_material(albedo, fuzz);
    //                 material* d_mat;
    //                 checkCuda(cudaMalloc((void**)&d_mat, sizeof(material)) );
    //                 checkCuda(cudaMemcpy(d_mat, &a_material, sizeof(material), cudaMemcpyHostToDevice) );
    //                 device_materials.push_back(d_mat);
    //                 h_spheres.push_back(hittable::make_sphere(center, 0.2f, d_mat));
    //             }
    //             else  {
    //                 // dielectric
    //                 glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
    //                 auto a_material = material::dielectric_material(1.5);
    //                 material* d_mat;
    //                 checkCuda(cudaMalloc((void**)&d_mat, sizeof(material)) );
    //                 checkCuda(cudaMemcpy(d_mat, &a_material, sizeof(material), cudaMemcpyHostToDevice) );
    //                 device_materials.push_back(d_mat);
    //                 h_spheres.push_back(hittable::make_sphere(center, 0.2f, d_mat));
    //             }
    //         }
    //     }
    // }


    // Three secundary spheres

    material h_mat1 = material::dielectric_material(1.5f);
    material* d_mat1;
    checkCuda(cudaMalloc((void**)&d_mat1, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat1, &h_mat1, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat1);
    h_spheres.push_back(hittable::make_sphere(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, d_mat1));

    material h_mat2 = material::lambertian_material(glm::vec3(0.4f, 0.2f, 0.1f));
    material* d_mat2;
    checkCuda(cudaMalloc((void**)&d_mat2, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat2, &h_mat2, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat2);
    h_spheres.push_back(hittable::make_sphere(glm::vec3(-4.0f, 1.0f, 0.0f), 1.0f, d_mat2));

    material h_mat3 = material::metal_material(glm::vec3(0.7f, 0.6f, 0.5f), 0.0);
    material* d_mat3;
    checkCuda(cudaMalloc((void**)&d_mat3, sizeof(material)) );
    checkCuda(cudaMemcpy(d_mat3, &h_mat3, sizeof(material), cudaMemcpyHostToDevice) );
    device_materials.push_back(d_mat3);
    h_spheres.push_back(hittable::make_sphere(glm::vec3(4.0f, 1.0f, 0.0f), 1.0f, d_mat3));
    AaBb bbox;
    for(auto &obj: h_spheres){
        bbox = AaBb(bbox, obj.sphere.bbox);
    }    
        
    
    // hittable_list hh_world(h_spheres);

    // BVH tree = BVH(hh_world);
    // auto bvh_nodes = tree.nodes;

    // BVH bvh = BVH(h_spheres);
    std::vector<BVHNode> h_nodes = BVH(h_spheres).nodes;
    for(auto &node : h_nodes){
        printf("Index: %d, is leaf? %d, leftIndex: %d, rightIndex: %d \n", node.object_index, node.is_leaf, node.left_child_index, node.right_child_index);
    }
    




    int number_of_hittables = h_spheres.size();
    checkCuda(cudaMalloc((void**)&d_spheres, number_of_hittables * sizeof(hittable)) );
    checkCuda(cudaMemcpy(d_spheres, h_spheres.data(), number_of_hittables * sizeof(hittable), cudaMemcpyHostToDevice) );

    int number_of_nodes = h_nodes.size();
    checkCuda(cudaMalloc((void**)&d_nodes, number_of_nodes * sizeof(BVHNode)) );
    checkCuda(cudaMemcpy(d_nodes, h_nodes.data(), number_of_nodes * sizeof(BVHNode), cudaMemcpyHostToDevice) );



    // hittable_list h_world(d_spheres, number_of_hittables);

    auto t_world(hittable::make_bvhTree(d_nodes, d_spheres));
    hittable_list h_world(&t_world, number_of_nodes, bbox);

    /* Allocate memory for hittable list on the device */
    checkCuda(cudaMalloc((void**)&d_world, sizeof(hittable_list)) );
    checkCuda(cudaMemcpy(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice) );




    
}


void RayTracer::cudaCall(int image_width, int image_height, int max_depth,  glm::vec3 center, glm::vec3 pixel00_loc, glm::vec3 pixel_delta_u, glm::vec3 pixel_delta_v, int samples_per_pixel, float& defocusAngle, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v, uint32_t* colorBuffer)
{  

    std::vector<material*>  device_materials;    

    /* device variables */
    uint32_t*   d_image;    // for display buffer
    curandState_t* d_states;  // random calculations in GPU
    
    
    

    hittable* d_spheres;
    hittable_list* d_world;
    BVHNode* d_nodes;
   
      

    init_objects(device_materials, d_spheres, d_nodes, d_world);

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
    init_random<<<gridSize, blockSize>>>(seed, d_states);
    // checkCuda(cudaDeviceSynchronize() );

    rayTracer_kernel<<<gridSize, blockSize>>>(d_states, max_depth, image_width, image_height, center, pixel00_loc, pixel_delta_u, pixel_delta_v, samples_per_pixel, defocusAngle, defocusDisk_u, defocusDisk_v, d_image, d_world);
    // checkCuda(cudaPeekAtLastError() );
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Took %f seconds with %d samples per pixel and %d max depth\n", timer_seconds, samples_per_pixel, max_depth);

    checkCuda(cudaMemcpy(colorBuffer, d_image, image_width * image_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // delete all device pointers of the materials
    for(auto& device : device_materials) 
        cudaFree(device);

    cudaFree(d_image);
    cudaFree(d_world);
    cudaFree(d_spheres);
    cudaFree(d_nodes);
    cudaFree(d_states);
    
    
}