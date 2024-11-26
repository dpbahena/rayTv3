#include "cuda_call.h"
#include "hittable.h"
#include "sphere.h"
#include "mem_manager.h"
#include "texture.h"

#include <cstdio>
#include <vector>
#include <random>
#include <chrono>

#define MAX_STACK_SIZE 20


__device__ inline glm::vec3 random_on_hemisphere(curandState_t* states,  int i, int j,const glm::vec3& normal);
__device__ inline glm::vec3 random_in_unit_sphere(curandState_t* states,  int i, int j);
__device__ inline glm::vec3 random_vector_in_range(curandState_t* states,  int i, int j, float min, float max);
__device__ inline glm::vec3 random_vector(curandState_t* states,  int i, int j);
__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n);
__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat);
__device__ inline glm::vec3 random_in_unit_disk(curandState_t* states,  int i, int j);
__device__ inline glm::vec3 defocus_disk_sample(curandState_t* states,  int i, int j, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v);
__device__ inline glm::vec3 random_unit_vector(curandState_t* states, int i, int j);
__device__ inline bool      near_zero(const glm::vec3 v);
__device__ inline float     reflectance(float cosine, float refraction_index);
__device__ inline float     random_float(curandState_t* state);
__device__ inline float     random_float_in_range(curandState_t* state, float a, float b);

// host function declarations
hittable* createBox(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat);
hittable* createConglomerate(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat, int numSpheres);
inline double random_double();
inline double random_double(float min, float max);
inline int random_int(int min, int max);
inline glm::vec3 unit_vector(const glm::vec3& v);



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
    return glm::normalize(v);
}

/**
 * @SOLID:   createTexture(memoryManager, Type::SOLID, color)
 * @IMAGE:   createTexture(memoryManager, Type::IMAGE, glm::vec3(0.0), glm::vec3(0.0), "/images/filename")
 * @CHECKER: createTexture(memoryManager, Type::CHECKER, glm::vec3(red, green, blue), glm::vec3(red, green, blue), NULL, 0, scale (0.0f - 1.0f))
 * @NOISE:   createTexture(memoryManager, Type::NOISE, glm::vec3(0.0), glm::vec3(0.0), NULL, scramble_frequency (0.0f - 1.0f))
 */
texture* createTexture(HybridMemoryManager& memoryManager, Type type, glm::vec3 color, glm::vec3 color2 = glm::vec3(0.0f), const char* filename = "", float scramble_frequency = 0.0f, float scale = 0.0f){
    texture* d_texture = memoryManager.allocateDevice<texture>();

    if (type == Type::SOLID) {
        texture h_color = texture::solid_texture(color);
        memoryManager.copyToDevice(d_texture, &h_color);
    
    } else if (type == Type::IMAGE) {
        auto image = rtw_image(filename);
        int size = image.width() * image.height() * image.pixelSize();
        unsigned char* d_bdata = memoryManager.allocateDevice<unsigned char>(size);
        memoryManager.copyToDevice(d_bdata, image.imageData(), size);
        auto h_image = texture::image_texture(d_bdata, image.width(), image.height(), image.scanLineSize(), image.pixelSize());
        memoryManager.copyToDevice(d_texture, &h_image);

    } else if (type == Type::NOISE) {
        Perlin noise;
        auto h_noise = texture(texture::noise_texture(noise, color,  scramble_frequency));
        memoryManager.copyToDevice(d_texture, &h_noise);

    } else if (type == Type::CHECKER) {
        auto odd = memoryManager.allocateHost<texture>(texture::solid_texture(color));
        auto even = memoryManager.allocateHost<texture>(texture::solid_texture(color2));
        auto checkerColor = texture::checker_texture(scale, odd, even);
        memoryManager.copyToDevice(d_texture, &checkerColor);
    }

    return d_texture;
}

material* createMaterial(HybridMemoryManager& memoryManager, Type type, texture* tex, glm::vec3 color = glm::vec3(0.0f), float fuzz = 0.0f, float refraction_index = 0.0f){
    material* d_mat = memoryManager.allocateDevice<material>();
    
    if(type == Type::LAMBERTIAN){
        auto h_mat = material::lambertian_material(tex);
        memoryManager.copyToDevice(d_mat, &h_mat);
        
    } else if(type == Type::DIFFUSE) {
        auto h_mat = material::diffuseLight_material(tex);
        memoryManager.copyToDevice(d_mat, &h_mat);
        
    } else if(type == Type::DIELECTRIC) {
        auto h_mat = material::dielectric_material(refraction_index);
        memoryManager.copyToDevice(d_mat, &h_mat);
    
    } else if (type == Type::METAL) {
        
        auto h_mat = material::metal_material(color, fuzz);
        memoryManager.copyToDevice(d_mat, &h_mat);
    }
     
    return d_mat;
}

/**
 * @brief Creates a vector of a 3D box (six sides) that contains the two opposites vertices a & b
 * 
 * @param sides 
 * @param a 
 * @param b 
 * @param mat 
 */
hittable* createBox(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat) {
    // Allocate raw memory for 6 hittable objects
    hittable* sides = static_cast<hittable*>(::operator new[](sizeof(hittable) * 6));
    memoryManager.host_allocations.push_back(sides); // Track allocation for cleanup

    // Calculate dimensions
    auto min = glm::vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    auto max = glm::vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));

    auto dx = glm::vec3(max.x - min.x, 0.0f, 0.0f);
    auto dy = glm::vec3(0, max.y - min.y, 0.0f);
    auto dz = glm::vec3(0, 0, max.z - min.z);
    AaBb bbox;
    // Use placement new to initialize the array elements
    new (&sides[0]) hittable(hittable::make_quad(glm::vec3(min.x, min.y, max.z), dx, dy, mat));  // front
    bbox = AaBb(bbox, sides[0].quad.bounding_box());
    new (&sides[1]) hittable(hittable::make_quad(glm::vec3(max.x, min.y, max.z), -dz, dy, mat)); // right
    bbox = AaBb(bbox, sides[1].quad.bounding_box());
    new (&sides[2]) hittable(hittable::make_quad(glm::vec3(max.x, min.y, min.z), -dx, dy, mat)); // back
    bbox = AaBb(bbox, sides[2].quad.bounding_box());
    new (&sides[3]) hittable(hittable::make_quad(glm::vec3(min.x, min.y, min.z), dz, dy, mat));  // left
    bbox = AaBb(bbox, sides[3].quad.bounding_box());
    new (&sides[4]) hittable(hittable::make_quad(glm::vec3(min.x, max.y, max.z), dx, -dz, mat)); // top
    bbox = AaBb(bbox, sides[4].quad.bounding_box());
    new (&sides[5]) hittable(hittable::make_quad(glm::vec3(min.x, min.y, min.z), dx, dz, mat));  // bottom
    bbox = AaBb(bbox, sides[5].quad.bounding_box());

    auto box = memoryManager.allocateHost<hittable>(hittable::make_hittableList(sides, 6)); 
    box->hittableList.bbox = bbox;
    return box;
}
/**
 * @brief Generates a collection of spheres with randomized positions 
 *        within a bounding box defined by glm::vec3(a) and glm::vec3(b). 
 *        The function uses placement new for efficient memory management 
 *        and integrates with the HybridMemoryManager class.
 * @param memoryManager - manages memory allocations and deletions
 * @param a random position within a bounding box
 * @param b random position within a bounding box
 * @param tex - texture of each sphere in the conglomerate 
 */
hittable* createConglomerate(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat, int numSpheres) {
    // Allocate raw memory for the number of spheres
    hittable* spheres = static_cast<hittable*>(::operator new[](sizeof(hittable) * numSpheres));
    memoryManager.host_allocations.push_back(spheres); // Track allocation for cleanup

    AaBb bbox;
    bool firstSphere = true;

    // Random number generator for sphere positions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(a.x, b.x);
    std::uniform_real_distribution<float> distY(a.y, b.y);
    std::uniform_real_distribution<float> distZ(a.z, b.z);
    std::uniform_real_distribution<float> radiusDist(0.1f, glm::length(b - a) * 0.1f); // Spheres' radii

    for (int i = 0; i < numSpheres; ++i) {
        // Randomized center position
        glm::vec3 center(distX(gen), distY(gen), distZ(gen));
        float radius = radiusDist(gen);

        // Create sphere using placement new
        new (&spheres[i]) hittable(hittable::make_sphere(center, radius, mat));

        // Update bounding box
        if (firstSphere) {
            bbox = spheres[i].sphere.bounding_box();
            firstSphere = false;
        } else {
            bbox = AaBb(bbox, spheres[i].sphere.bounding_box());
        }
    }

    // Allocate a hittable list for the conglomerate
    auto conglomerate = memoryManager.allocateHost<hittable>(hittable::make_hittableList(spheres, numSpheres));
    conglomerate->hittableList.bbox = bbox;

    return conglomerate;
}

/**
 * @brief Convert an array of hittables to a BVH hittable
 * @param group: std::vector<hittable> to be converted to BVH hittable
 * @output:  a BVH hittable
 */
hittable createBVH(HybridMemoryManager& memoryManager, std::vector<hittable> group) {
    auto d_group = memoryManager.allocateDevice<hittable>(group.size());
    memoryManager.copyToDevice(d_group, group.data(), group.size());
    int number_of_nodes = (2 * group.size() - 1);
    auto nodes = memoryManager.allocateDevice<BVHNode>(number_of_nodes);
    return hittable::make_bvhNode(nodes, d_group, group.size());
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
    // attenuation = metal.tex->value(rec.u, rec.v, rec.p);

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
     attenuation = isotropic.tex->solidColor.value(rec.u, rec.v, rec.p);
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

// scenes 1 - 10
void bouncing_spheres(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(13.0f, 2.0f,  3.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.6f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();

    std::vector<hittable> h_hittables_list, small_spheres;
    
    auto tex = createTexture(memoryManager, Type::CHECKER, glm::vec3(0.2f, 0.3f, 0.1f), glm::vec3(0.9f, 0.9f, 0.9f),{}, {}, 0.32f);
    auto ground = createMaterial(memoryManager, Type::LAMBERTIAN, tex);
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,-1000.0, 0.0), 1000, ground);
    h_hittables_list.push_back(hittable_obj);

    // Create random spheres 
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_material = random_double();
            glm::vec3 center(a + 0.9f * random_double(), 0.2f, b + 0.9f * random_double());
            hittable sphere;

            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f) {
                if(choose_material < 0.8f) {
                    // difuse
                    glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
                    glm::vec3 center2 = center + glm::vec3(0,random_double(0, 0.5), 0);
                    auto lmat = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, albedo));
                    
                    sphere = hittable::make_sphere(center, center2, 0.2f, lmat);

                }else if(choose_material < 0.95f) {
                    // metal
                    glm::vec3 albedo = glm::vec3(random_double(), random_double(), random_double()) * glm::vec3(random_double(), random_double(), random_double());
                    float fuzz = random_double(0.0f, 0.5f);
                    auto mat = createMaterial(memoryManager, Type::METAL, {}, albedo, fuzz);
                    sphere = hittable::make_sphere(center, 0.2f, mat);
                }
                else  {
                    // dielectric
                    auto mat = createMaterial(memoryManager, Type::DIELECTRIC, {}, {}, {}, 1.5);
                    sphere = hittable::make_sphere(center, 0.2f, mat);
                }
                small_spheres.push_back(sphere);
            }
        }
    }


    // Three secundary spheres
    auto glass = createMaterial(memoryManager, Type::DIELECTRIC, {}, {}, {}, 1.5f);
    hittable_obj = hittable::make_sphere(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, glass);
    h_hittables_list.push_back(hittable_obj);

    auto a_color = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.4f, 0.2f, 0.1f), {},{},{},{}));
    hittable_obj = hittable::make_sphere(glm::vec3(-4.0f, 1.0f, 0.0f), 1.0f, a_color);
    h_hittables_list.push_back(hittable_obj);

    auto silver = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(0.7f, 0.6f, 0.5f), 0.0, {});
    hittable_obj = hittable::make_sphere(glm::vec3(4.0f, 1.0f, 0.0f), 1.0f, silver);
    h_hittables_list.push_back(hittable_obj);
 

    // convert this group to BVH
    auto bvhItem = createBVH(memoryManager, small_spheres);
    h_hittables_list.push_back(bvhItem);


    size_t number_of_hittables = h_hittables_list.size();

    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
       
    auto h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
}

void checkered_spheres(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(13.0f, 2.0f,  3.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();

    std::vector<hittable> h_hittables_list;
    
    auto tex1 = createTexture(memoryManager, Type::CHECKER, glm::vec3(0.2f, 0.3f, 0.1f), glm::vec3(0.9f, 0.9f, 0.9f), NULL, 0, 0.32f);
    auto topMat = createMaterial(memoryManager, Type::LAMBERTIAN, tex1);
    
    auto tex2 = createTexture(memoryManager, Type::CHECKER, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), NULL, 0, 0.08f);
    auto bottomMat = createMaterial(memoryManager, Type::LAMBERTIAN, tex2);
    
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,-10.0, 0.0), 10, bottomMat);
    h_hittables_list.push_back(hittable_obj);

    hittable_obj = hittable::make_sphere(glm::vec3(0.0, 10.0, 0.0), 10, topMat);
    h_hittables_list.push_back(hittable_obj);

    size_t number_of_hittables = h_hittables_list.size();

    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);

    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);

}

void earth(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(0.0f, 0.0f,  12.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
      
    std::vector<hittable> h_hittables_list;

    auto tex = createTexture(memoryManager, Type::IMAGE, {}, {}, "images/earth_map.jpg");
    auto globe = createMaterial(memoryManager, Type::LAMBERTIAN, tex);

    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,0.0, 0.0), 2, globe);
    h_hittables_list.push_back(hittable_obj);

    size_t number_of_hittables = h_hittables_list.size();
    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
    
    auto h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);

    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
    
}

void perlin_spheres(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(13.0f, 2.0f,  3.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
    
    
    std::vector<hittable> h_hittables_list;
    auto red = glm::vec3(1.0, 0.0, 0.0f);
    auto blue = glm::vec3(0.0, 0.0, 1.0);   
    auto tex = createTexture(memoryManager, Type::NOISE, red, {}, {}, 4.0f);
    auto perlin = createMaterial(memoryManager, Type::LAMBERTIAN, tex);

    auto tex1 = createTexture(memoryManager, Type::NOISE, blue, {}, {}, 1.0f);
    auto perlin1 = createMaterial(memoryManager, Type::LAMBERTIAN, tex1);

    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0, -1000.0, 0.0), 1000, perlin);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_sphere(glm::vec3(0.0,2.0, 0.0), 2, perlin1);
    h_hittables_list.push_back(hittable_obj);

    
    size_t number_of_hittables = h_hittables_list.size();

    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);

    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);

}

void quads(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

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
    // material h_left_red = material::lambertian_material(glm::vec3(1.0, 0.2, 0.2));
    // material* d_left_red = memoryManager.allocateDevice<material>();
    // memoryManager.copyToDevice(d_left_red, &h_left_red);

    auto left_red = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 0.2, 0.2)));

    // checkCuda(cudaMalloc((void**)&d_left_red, sizeof(material)) );
    // checkCuda(cudaMemcpy(d_left_red, &h_left_red, sizeof(material), cudaMemcpyHostToDevice) );
    // device_materials.push_back(d_left_red);

    // material h_back_green = material::lambertian_material(glm::vec3(0.2, 1.0, 0.2));
    // material* d_back_green = memoryManager.allocateDevice<material>();
    // memoryManager.copyToDevice(d_back_green, &h_back_green);

    auto back_green = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 1.0, 0.2)));

    // checkCuda(cudaMalloc((void**)&d_back_green, sizeof(material)) );
    // checkCuda(cudaMemcpy(d_back_green, &h_back_green, sizeof(material), cudaMemcpyHostToDevice) );
    // device_materials.push_back(d_back_green);

    // material h_right_blue = material::lambertian_material(glm::vec3(0.2, 0.2, 1.0));
    // material* d_right_blue = memoryManager.allocateDevice<material>();
    // memoryManager.copyToDevice(d_right_blue, &h_right_blue);


    auto right_blue = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.2, 1.0)));
    // checkCuda(cudaMalloc((void**)&d_right_blue, sizeof(material)) );
    // checkCuda(cudaMemcpy(d_right_blue, &h_right_blue, sizeof(material), cudaMemcpyHostToDevice) );
    // device_materials.push_back(d_right_blue);

    // material h_upper_orange = material::lambertian_material(glm::vec3(1.0, 0.5, 0.0));
    // material* d_upper_orange = memoryManager.allocateDevice<material>();
    // memoryManager.copyToDevice(d_upper_orange, &h_upper_orange);

    auto upper_orange = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 0.5, 0.0)));
    // checkCuda(cudaMalloc((void**)&d_upper_orange, sizeof(material)) );
    // checkCuda(cudaMemcpy(d_upper_orange, &h_upper_orange, sizeof(material), cudaMemcpyHostToDevice) );
    // device_materials.push_back(d_upper_orange);

    // material h_lower_teal = material::lambertian_material(glm::vec3(0.2, 0.8, 0.8));
    // material* d_lower_teal = memoryManager.allocateDevice<material>();
    // memoryManager.copyToDevice(d_lower_teal, &h_lower_teal);

    auto lower_teal = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.8, 0.8)));
    // checkCuda(cudaMalloc((void**)&d_lower_teal, sizeof(material)) );
    // checkCuda(cudaMemcpy(d_lower_teal, &h_lower_teal, sizeof(material), cudaMemcpyHostToDevice) );
    // device_materials.push_back(d_lower_teal);

    /* Quads */
    hittable hittable_obj;
    hittable_obj = hittable::make_quad(glm::vec3(-3.0f, -2.0f, 5.0f), glm::vec3(0.0f, 0.0f, -4.0f), glm::vec3(0.0f, 4.0f,  0.0f), left_red);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3(-2.0f, -2.0f, 0.0f), glm::vec3(4.0f, 0.0f, -0.0f), glm::vec3(0.0f, 4.0f,  0.0f), back_green);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3( 3.0f, -2.0f, 1.0f), glm::vec3(0.0f, 0.0f,  4.0f), glm::vec3(0.0f, 4.0f,  0.0f), right_blue);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3(-2.0f,  3.0f, 1.0f), glm::vec3(4.0f, 0.0f, -0.0f), glm::vec3(0.0f, 0.0f,  4.0f), upper_orange);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_quad(glm::vec3(-2.0f, -3.0f, 5.0f), glm::vec3(4.0f, 0.0f, -0.0f), glm::vec3(0.0f, 0.0f, -4.0f), lower_teal);
    h_hittables_list.push_back(hittable_obj);

    
    size_t number_of_hittables = h_hittables_list.size();

    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
     

    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
    
}

void simple_light(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3( 26.0f, 3.0f,  6.0f);
    cam.lookat   = glm::vec3( 0.0f, 2.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
    
    std::vector<hittable> h_hittables_list;

    auto whiteNoise_tex = createTexture(memoryManager, Type::NOISE, glm::vec3(0.8, 0.8, 0.8), {},{}, 4.0f);
    auto whiteNoise = createMaterial(memoryManager, Type::LAMBERTIAN, whiteNoise_tex);

    /* Spheres */
    auto hittable_obj2 = hittable::make_sphere(glm::vec3(0.0, -1000.0, 0.0), 1000, whiteNoise);
    h_hittables_list.push_back(hittable_obj2);

    auto greenNoise_tex = createTexture(memoryManager, Type::NOISE, glm::vec3(0.1, 0.8, 0.1), {},{}, 2.0f);
    auto greenNoise = createMaterial(memoryManager, Type::LAMBERTIAN, greenNoise_tex);


    auto hittable_obj3 = hittable::make_sphere(glm::vec3(0.0, 2.0, 0.0), 2, greenNoise);
    h_hittables_list.push_back(hittable_obj3);

    auto mat1 = createMaterial(memoryManager, Type::DIFFUSE, createTexture(memoryManager, Type::SOLID, glm::vec3(0.0f, 0.0f, 0.0f)) );

    auto hittable_obj4 = hittable::make_quad(glm::vec3( 3.0f,  1.0f, -2.0f), glm::vec3(2.0f, 0.0f, -0.0f), glm::vec3(0.0f, 2.0f,  0.0f), mat1);
    h_hittables_list.push_back(hittable_obj4);

    auto mat2 = createMaterial(memoryManager, Type::DIFFUSE, createTexture(memoryManager, Type::SOLID, glm::vec3(8.0f, 0.0f, 0.0f)) );

    hittable_obj4 = hittable::make_sphere(glm::vec3(-5.0f, 3.0f, 3.0f), .5, mat2);
    h_hittables_list.push_back(hittable_obj4);

    
    size_t number_of_hittables = h_hittables_list.size();
    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
    
    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);

    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
    
}


void cornell_box(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

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

    

    auto red    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.65, 0.05, 0.05)));
    auto white    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(.73, .73, .73)));
    auto green    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(.12, .45, .15)));
    auto light    = createMaterial(memoryManager, Type::DIFFUSE, createTexture(memoryManager, Type::SOLID, glm::vec3(15, 15, 15)));
    auto mirror    = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(1.0, 1.0, 1.0), 0.0 );

    auto obj1 = hittable(hittable::make_quad(glm::vec3(555,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), green));
    auto obj2 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), red));
    auto obj3 = hittable(hittable::make_quad(glm::vec3(343, 554, 332), glm::vec3(-130,0,0), glm::vec3(0,0,-105), light));
    auto obj4 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(555,0,0), glm::vec3(0,0,555), white));
    auto obj5 = hittable(hittable::make_quad(glm::vec3(555,555,555), glm::vec3(-555,0,0), glm::vec3(0,0,-555), white));
    auto obj6 = hittable(hittable::make_quad(glm::vec3(0,0,555), glm::vec3(555,0,0), glm::vec3(0,555,0), white));
   
    auto obj7 = hittable(hittable::make_quad(glm::vec3(555,0, 75), glm::vec3(0,200,0), glm::vec3(0, 0, 200), mirror));

    h_hittables_list.push_back(obj1);
    h_hittables_list.push_back(obj2);
    h_hittables_list.push_back(obj3);
    h_hittables_list.push_back(obj4);
    h_hittables_list.push_back(obj5);
    h_hittables_list.push_back(obj6);
    h_hittables_list.push_back(obj7);
    
    //* Create two boxes
    auto box1 = createBox(memoryManager, glm::vec3(130.0f, 0.0f, 65.0f),  glm::vec3(295.0f, 165.0f, 230.0f), white);
    auto box2 = createBox(memoryManager, glm::vec3(265.0f, 0.0f, 295.0f), glm::vec3(430.0f, 330.0f, 460.0f), white);
    h_hittables_list.push_back(*box1);
    h_hittables_list.push_back(*box2);
    
    size_t number_of_hittables = h_hittables_list.size();

    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
  
    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
}

void cornell_box_instances(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 278.0f, 278.0f, -800.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
    
    std::vector<hittable> h_hittables_list;  // hold all items in general native and bvh items
    std::vector<hittable> boxes1, boxes2;  // holds bvh items by group
    
    
    
    auto red    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.65, 0.05, 0.05)));
    auto white    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(.73, .73, .73)));
    auto green    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(.12, .45, .15)));
    auto light    = createMaterial(memoryManager, Type::DIFFUSE, createTexture(memoryManager, Type::SOLID, glm::vec3(15, 15, 15)));
    auto mirror    = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(1.0, 1.0, 1.0), 0.0 );


    auto obj1 = hittable(hittable::make_quad(glm::vec3(555,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), green));
    auto obj2 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), red));
    auto obj3 = hittable(hittable::make_quad(glm::vec3(343, 554, 332), glm::vec3(-130,0,0), glm::vec3(0,0,-105), light));
    auto obj4 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(555,0,0), glm::vec3(0,0,555), white));
    auto obj5 = hittable(hittable::make_quad(glm::vec3(555,555,555), glm::vec3(-555,0,0), glm::vec3(0,0,-555), white));
    auto obj6 = hittable(hittable::make_quad(glm::vec3(0,0,555), glm::vec3(555,0,0), glm::vec3(0,555,0), white));
   
    h_hittables_list.push_back(obj1);
    h_hittables_list.push_back(obj2);
    h_hittables_list.push_back(obj3);
    h_hittables_list.push_back(obj4);
    h_hittables_list.push_back(obj5);
    h_hittables_list.push_back(obj6);

    

    //* Create two boxes
    auto box1 = createBox(memoryManager, glm::vec3(.0f, 0.0f, 0.0f),  glm::vec3(165.0f, 330.0f, 165.0f), white);
    auto box2 = createBox(memoryManager, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(165.0f, 165.0f, 165.0f), white);
    

    auto rotated = memoryManager.allocateHost<hittable>(hittable::make_rotateY(box1, 15));
    auto translated = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(265.0, 0.0f, 295.0f)));
    // h_hittables_list.push_back(*translated);
    boxes1.push_back(*translated);

    rotated = memoryManager.allocateHost<hittable>(hittable::make_rotateY(box2, -18));
    translated = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(130.0f, 0.0f, 65.0f)));
    // h_hittables_list.push_back(*translated);
    boxes2.push_back(*translated);

    // transfer boxes to BVH nodes
    auto bvhItem1 = createBVH(memoryManager, boxes1);
    h_hittables_list.push_back(bvhItem1);
    auto bvhItem2 = createBVH(memoryManager, boxes2);
    h_hittables_list.push_back(bvhItem2);
  
    size_t number_of_hittables = h_hittables_list.size();
  
    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);

    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);

    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);

}

void cornell_smoke(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 278.0f, 278.0f, -800.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;  // hold all items in general native and bvh items
    std::vector<hittable> boxes1, boxes2;  // holds bvh items by group
    
    // allocate textures
    auto atex = texture::solid_texture(glm::vec3(0.0f, 0.0f, 0.0f)); 
    texture* d_atex = memoryManager.allocateDevice<texture>();
    memoryManager.copyToDevice(d_atex, &atex);

    auto blueTex = texture::solid_texture(glm::vec3(0.1f, 0.1f, 1.0f)); 
    texture* d_blueTex = memoryManager.allocateDevice<texture>();
    memoryManager.copyToDevice(d_blueTex, &blueTex);

    auto light_tex = texture::solid_texture(glm::vec3(7.0f, 7.0f, 7.0f));
    texture* d_light_tex = memoryManager.allocateDevice<texture>();
    memoryManager.copyToDevice(d_light_tex, &light_tex);
   
    // Allocate Materials
    auto negro = material::isotropic_material(d_atex);
    material* d_negro = memoryManager.allocateDevice<material>();
    memoryManager.copyToDevice(d_negro, &negro);

    auto blueish = material::isotropic_material(d_blueTex);
    material* d_blueish = memoryManager.allocateDevice<material>();
    memoryManager.copyToDevice(d_blueish, &blueish);


    auto red    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.65, 0.05, 0.05)));
    auto white    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(.73, .73, .73)));
    auto green    = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(.12, .45, .15)));
    auto light    = createMaterial(memoryManager, Type::DIFFUSE, createTexture(memoryManager, Type::SOLID, glm::vec3(7.0, 7.0, 7.0)));
    auto mirror    = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(1.0, 1.0, 1.0), 0.0 );

    auto obj1 = hittable(hittable::make_quad(glm::vec3(555,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), green));
    auto obj2 = hittable(hittable::make_quad(glm::vec3(0,0,0), glm::vec3(0,555,0), glm::vec3(0,0,555), red));
    auto obj3 = hittable(hittable::make_quad(glm::vec3(113, 554, 127), glm::vec3(330,0,0), glm::vec3(0,0, 305), light));
    auto obj4 = hittable(hittable::make_quad(glm::vec3(0,555,0), glm::vec3(555,0,0), glm::vec3(0,0,555), white));
    auto obj5 = hittable(hittable::make_quad(glm::vec3(0 ,0 , 0), glm::vec3(555,0,0), glm::vec3(0,0,555), white));
    auto obj6 = hittable(hittable::make_quad(glm::vec3(0,0,555), glm::vec3(555,0,0), glm::vec3(0,555,0), white));

    auto obj7 = hittable(hittable::make_quad(glm::vec3(555,50, 50), glm::vec3(0,400,0), glm::vec3(0, 0, 400), mirror));
    auto obj8 = hittable(hittable::make_quad(glm::vec3(0,50, 50), glm::vec3(0,400,0), glm::vec3(0, 0, 400), mirror));
   
   
    h_hittables_list.push_back(obj1);
    h_hittables_list.push_back(obj2);
    h_hittables_list.push_back(obj3);
    h_hittables_list.push_back(obj4);
    h_hittables_list.push_back(obj5);
    h_hittables_list.push_back(obj6);
    
    h_hittables_list.push_back(obj7);
    h_hittables_list.push_back(obj8);
    

    auto sphere = memoryManager.allocateHost<hittable>(hittable::make_sphere(glm::vec3(150, 350, 275), 60, white));
    auto nube   = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(sphere, 0.01f, d_blueish));
    h_hittables_list.push_back(*nube);

    // Allocate hittable boxes
    auto box1 = createBox(memoryManager, glm::vec3(0.0f, 0.0f, 0.0f),  glm::vec3(165.0f, 330.0f, 165.0f), white);
    auto box2 = createBox(memoryManager, glm::vec3(0.0f, 0.0f, 0.0f),  glm::vec3(165.0f, 165.0f, 165.0f), white);
    
    auto rotated        = memoryManager.allocateHost<hittable>(hittable::make_rotateY(box1, 15));
    auto translated     = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(265, 0, 295)));
    auto smoked    = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(translated, 0.01f, d_negro));
    // h_hittables_list.push_back(*smoked);
    boxes1.push_back(*smoked);
    


    rotated     = memoryManager.allocateHost<hittable>(hittable::make_rotateY(box2, -18));
    translated  = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(130, 0, 65)));
    smoked      = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(translated, 0.01f, white));
    // h_hittables_list.push_back(*smoked);
    boxes2.push_back(*smoked);

    // transfer boxes to BVH nodes
    auto bvhItem1 = createBVH(memoryManager, boxes1);
    h_hittables_list.push_back(bvhItem1);
    auto bvhItem2 = createBVH(memoryManager, boxes2);
    h_hittables_list.push_back(bvhItem2);
   
    size_t number_of_hittables = h_hittables_list.size();
    printf("size: %d\n", (int)number_of_hittables);
    
    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
    
    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);

    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
}



void finalScene(Camera& cam, HybridMemoryManager& memoryManager, BVHNode* &bvh_nodes, hittable* &d_hittable_list, hittable* &d_world){
    
    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 478.0f, 278.0f, -600.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
    
    // hold some hittables
    std::vector<hittable> h_hittables_list;
    
    // create a hittable list of boxes (each box is a hittable)
    auto groundColor = createTexture(memoryManager, Type::SOLID, glm::vec3(0.48, 0.83, 0.53));
    auto ground = createMaterial(memoryManager, Type::LAMBERTIAN, groundColor);


    std::vector<hittable> boxesGroup, conglomerateGroup;
    int boxes_per_side = 20;

    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1,101);
            auto z1 = z0 + w;
            auto box = createBox(memoryManager, glm::vec3(x0, y0, z0), glm::vec3(x1, y1, z1), ground);
            boxesGroup.push_back(*box);
        }
    }
     

    // ceiling light
    auto lightColor = createTexture(memoryManager, Type::SOLID, glm::vec3(7.0, 7.0, 7.0));
    auto light      = createMaterial(memoryManager, Type::DIFFUSE, lightColor);
    auto ceilingLamp = hittable::make_quad(glm::vec3(123, 554, 147), glm::vec3(300, 0, 0), glm::vec3(0, 0, 226), light);
    h_hittables_list.push_back(ceilingLamp);
    
    //moving sphere
    auto center1 = glm::vec3(400, 400, 200);
    auto center2 = center1 + glm::vec3(30, 0, 0);
    auto sphereTexture  = createTexture(memoryManager, Type::SOLID, glm::vec3(0.7, 0.3, 0.1));
    auto sphereMaterial = createMaterial(memoryManager, Type::LAMBERTIAN, sphereTexture);
    h_hittables_list.push_back(hittable::make_sphere(center1, center2, 50, sphereMaterial));
    // glass sphere
    auto glass_material = createMaterial(memoryManager, Type::DIELECTRIC, 0, glm::vec3(0), 0, 1.5f );
    h_hittables_list.push_back(hittable::make_sphere(glm::vec3(260, 150, 45), 50, glass_material));
    // metal sphere
    auto metalMat = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(0.8f, 0.8f, 0.9f), 1.0f);
    h_hittables_list.push_back(hittable::make_sphere(glm::vec3(0, 150, 145), 50, metalMat));

    // blue shiny sphere
    auto boundary = memoryManager.allocateHost<hittable>(hittable::make_sphere(glm::vec3(360, 150, 145), 70, createMaterial(memoryManager, Type::DIELECTRIC, NULL, glm::vec3(0), 0, 1.5)));
    h_hittables_list.push_back(*boundary);
    h_hittables_list.push_back(*memoryManager.allocateHost<hittable>(hittable::make_constantMedium(boundary, 0.2, glm::vec3(0.2, 0.4, 0.9))));
    // ??? what sphere is this one?
    auto boundary1 = memoryManager.allocateHost<hittable>(hittable::make_sphere(glm::vec3(0, 0, 0), 5000, createMaterial(memoryManager, Type::DIELECTRIC, NULL, glm::vec3(0), 0, 1.5) ));
    h_hittables_list.push_back(*memoryManager.allocateHost<hittable>(hittable::make_constantMedium(boundary1, .0001f, createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 1.0, 1.0)))));
    // Globe
    auto mapTex = createTexture(memoryManager, Type::IMAGE, glm::vec3(0), glm::vec3(0), "images/earth_map.jpg");
    auto emat = createMaterial(memoryManager, Type::LAMBERTIAN, mapTex);
    h_hittables_list.push_back(hittable::make_sphere(glm::vec3(400, 200, 400), 100, emat ));
    // Perlin patter sphere
    auto perlinTex = createTexture(memoryManager, Type::NOISE, glm::vec3(0.5,0.5,0.5), glm::vec3(0), NULL , 0.2);
    h_hittables_list.push_back(hittable::make_sphere(glm::vec3(220, 280, 300), 80, createMaterial(memoryManager, Type::LAMBERTIAN, perlinTex)));
    // h_hittables_list.push_back(hittable::make_sphere(glm::vec3(220, 280, 300), 80, andreaMat));
    auto a = glm::vec3(0, 0, 0);
    auto b = glm::vec3(165, 165, 165);

    auto white  = createTexture(memoryManager, Type::SOLID, glm::vec3(0.73, 0.73, 0.73));
    auto whiteMat = createMaterial(memoryManager, Type::LAMBERTIAN, white);


    auto conglomerate = createConglomerate(memoryManager, a, b, whiteMat, 1000 );
    auto rotated    = memoryManager.allocateHost<hittable>(hittable::make_rotateY(conglomerate, 15));
    auto translated = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(-100, 270, 395)));

    conglomerateGroup.push_back(*translated);

    auto bvhItem1 = createBVH(memoryManager, boxesGroup);
    h_hittables_list.push_back(bvhItem1);

    auto bvhItem2 = createBVH(memoryManager, conglomerateGroup);
    h_hittables_list.push_back(bvhItem2);


    // complete the scene
    size_t number_of_hittables = h_hittables_list.size();
    printf("size: %d\n", (int)number_of_hittables);
    
    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);

    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);

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
    HybridMemoryManager memoryManager;
   
    hittable*   d_hittables_list    = memoryManager.deferDeviceAllocation<hittable>();
    hittable*   d_world             = memoryManager.deferDeviceAllocation<hittable>();
    BVHNode*    bvh_nodes           = memoryManager.deferDeviceAllocation<BVHNode>();
    switch (cam.scene)  
    {
    case 1:
        
        bouncing_spheres(cam, memoryManager,  bvh_nodes, d_hittables_list, d_world);
        break;
    case 2:
        checkered_spheres(cam, memoryManager,  bvh_nodes, d_hittables_list, d_world);
        break;
    case 3:
        earth(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    case 4:
        perlin_spheres(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    case 5:
        quads(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    case 6:
        simple_light(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    case 7:
        cornell_box(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    case 8:
        cornell_box_instances(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    case 9:
        cornell_smoke(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    case 10:
        finalScene(cam, memoryManager, bvh_nodes, d_hittables_list, d_world);
        break;
    default:
        break;
    }
    Camera*     d_cam       = memoryManager.allocateDevice<Camera>();
    uint32_t*   d_image     = memoryManager.allocateDevice<uint32_t>(cam.image_width * cam.image_height * sizeof(uint32_t));    // for display buffer
    
    
    memoryManager.copyToDevice(d_cam, &cam);
    
    clock_t start, stop;
    start = clock();

    int threads = 8;
    dim3 blockSize(threads, threads);
    int blocks_x = (cam.image_width  + blockSize.x - 1) / blockSize.x;
    int blocks_y = (cam.image_height  + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(blocks_x, blocks_y);



    //generate random seed to be used in rayTracer kernel
    int num_threads = threads * threads * blocks_x * blocks_y;
    curandState_t* d_states = memoryManager.allocateDevice<curandState_t>(num_threads * sizeof(curandState_t));  // random calculations in GPU
    // checkCuda(cudaMalloc(&d_states, num_threads * sizeof(curandState_t)));
    init_random<<<gridSize, blockSize>>>(seed, d_states);
    // checkCuda(cudaDeviceSynchronize() );
 
    
    // size_t shared_memory_size =  blockSize.x * blockSize.y * MAX_STACK_SIZE * sizeof(int);
    // if(cam.isBvh){
        
    //     rayTracer_nodes_kernel<<<gridSize, blockSize, shared_memory_size>>>(d_states, d_cam, d_image, d_world);
        
    // }
    // else
    rayTracer_kernel<<<gridSize, blockSize>>>(d_states, d_cam, d_image, d_world);
    
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Took %f seconds with %d samples per pixel and %d max depth\n", timer_seconds, cam.samples_per_pixel, cam.max_depth);

    checkCuda(cudaMemcpy(colorBuffer, d_image, cam.image_width * cam.image_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
}
