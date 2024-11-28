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
inline hittable* createBox(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat);
inline hittable* createConglomerate(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat, int numSpheres);
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

    } else if (type == Type::ISOTROPIC) {
        auto h_mat = material::isotropic_material(tex);
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
inline hittable* createBox(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat) {
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
inline hittable* createConglomerate(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat, int numSpheres) {
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
/**
 * @brief Convert a single hittables to a BVH hittable
 * @param  single compound <hittable> to be converted to BVH hittable (conglomerate, box, etc)
 * @output:  a BVH hittable
 */
hittable createBVH(HybridMemoryManager& memoryManager, hittable* object) {
    auto d_object = memoryManager.allocateDevice<hittable>(1);
    memoryManager.copyToDevice(d_object, object, 1);
    int number_of_nodes = (2 * 1 - 1);
    auto nodes = memoryManager.allocateDevice<BVHNode>(number_of_nodes);
    return hittable::make_bvhNode(nodes, d_object, 1);
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
inline glm::vec3 random_unit_vector(curandState_t* states, int i, int j){
    auto p = random_in_unit_sphere(states, i, j);
    return glm::normalize(p);
}

__device__
inline bool near_zero(const glm::vec3 v) {
    auto s = 1e-8f;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

__device__
inline glm::vec3 sample_square(curandState_t* states, int &i, int &j) {
    curandState_t x = states[i];
    curandState_t y = states[j];
    auto a = random_float(&x) - 0.5f;
    auto b = random_float(&y) - 0.5f;
    states[i] = x; // save back the value
    states[j] = y;
    return glm::vec3(a, b, 0.0f);
}

__device__
inline float linear_to_gamma(float linear_component)
{
    if (linear_component > 0.0f)
        return std::sqrt(linear_component);

    return 0;
}

__device__
inline uint32_t colorToUint32_t(glm::vec3& c)
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
inline glm::vec3 random_on_hemisphere(curandState_t* states,  int i, int j,const glm::vec3& normal) {
    glm::vec3 on_unit_sphere = random_unit_vector(states, i, j);
    if (glm::dot(on_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__
inline glm::vec3 random_in_unit_sphere(curandState_t* states,  int i, int j) {
    while (true) {
        glm::vec3 p = random_vector_in_range(states, i, j, -1.0f ,1.0f);
        if (glm::dot(p,p) < 1.0f){
            return p;
        }
    }
}

__device__
inline glm::vec3 random_in_unit_disk(curandState_t* states,  int i, int j){
    curandState_t x = states[i];
    curandState_t y = states[j];
    while (true) {
        auto p = glm::vec3(random_float_in_range(&x, -1, 1), random_float_in_range(&y, -1, 1), 0);
        if (glm::dot(p,p) < 1.0f)
            return p;
    }
}

 __device__
inline glm::vec3 defocus_disk_sample(curandState_t* states,  int i, int j, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v) {
    // returns a random point in the camera defocus disk
    glm::vec3 p = random_in_unit_disk(states, i, j);
    return center + p.x * defocusDisk_u + p.y * defocusDisk_v;
}

__device__
inline glm::vec3 random_vector_in_range(curandState_t* states,  int i, int j, float min, float max){
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
inline glm::vec3 random_vector(curandState_t* states,  int i, int j){
    curandState_t x = states[i];
    curandState_t y = states[j];
    float a = random_float(&x);
    float b = random_float(&y);
    float c = random_float(&x); //a * b;
    states[i] = x; // save value back
    states[j] = y;
    return glm::vec3(a, b, c);

}

__device__ inline float random_float_in_range(curandState_t* state, float a, float b) {
    // return a + (b - a) * curand_uniform_float(state);  // this does not include b  e.g -1 to 1.0  it does not include 1.0
    return a + (b - a) * (curand_uniform_double(state) - 0.5) * 2.0;  // this approach includes the upper limit   -1 to 1.0  it includes 1.0
}

/**
 * @return a random integer in [min, max] including the upper limit
 */
__device__ inline int random_int(curandState_t* state, int a, int b) {
    return static_cast<int>(a + (b - a) * (curand_uniform_double(state) - 0.5) * 2.0);  // this approach includes the upper limit   -1 to 1.0  it includes 1.0
}

__device__
inline glm::vec3 ray_color(curandState_t* state,  int i, int j, int depth, const glm::vec3& background, const ray &r, const hittable* world) {
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
inline ray get_ray(curandState_t* states, int &i, int &j, glm::vec3& pixel00_loc, glm::vec3& cameraCenter, glm::vec3& delta_u, glm::vec3& delta_v, float& defocusAngle, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v) {
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

__global__ void init_random2(unsigned int seed, curandState_t* states, int image_width, int image_height, int pixels_per_block){
    int pixel_x = blockIdx.x;
    int pixel_y = blockIdx.y * pixels_per_block + threadIdx.y;

    if (pixel_x >= image_width || pixel_y >= image_height)
        return;

    int idx = pixel_y * image_width + pixel_x;
    curand_init(seed, idx, 0, &states[idx]);
}


// __global__ void init_random(
//     unsigned int seed,
//     curandState_t* states,
//     int image_width,
//     int image_height
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= image_width || j >= image_height)
//         return;

//     int pixel_index = j * image_width + i;

//     // Initialize the random state
//     curand_init(seed, pixel_index, 0, &states[pixel_index]);
// }


// __global__ void rayTracer_kernel(curandState_t* states, Camera* cam, uint32_t* image, hittable* world) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= cam->image_width || j >= cam->image_height) return;
    
//     glm::vec3 color = {0.0f, 0.0f, 0.0f};
//     for (int sample = 0; sample < cam->samples_per_pixel; sample++){
//         ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center, cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle, cam->defocus_disk_u, cam->defocus_disk_v);
//         color  += ray_color(states, i, j, cam->max_depth, cam->background, r, world);
//     }
    
//     color *= cam->pixel_sample_scale;
//     image[cam->image_width * j + i] = colorToUint32_t(color);  
// }

// __global__ void rayTracer_kernel(curandState_t* states, Camera* cam, float* image, hittable* world) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int sample = blockIdx.z * blockDim.z + threadIdx.z;
//     if (i >= cam->image_width || j >= cam->image_height || sample >= cam->samples_per_pixel) return;
//     // compute unique thread index for random states
//     int pixel_index = j * cam->image_width + i;
//     // int thread_index = pixel_index * cam->samples_per_pixel + sample;
//     // Initialize the random state for this thread
//     // curandState_t local_state = states[thread_index];
    
//     ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center, cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle, cam->defocus_disk_u, cam->defocus_disk_v);
//     //* Compute the color for this sample
//     glm::vec3 sample_color = ray_color(states, i, j, cam->max_depth, cam->background, r, world);
    
//     int image_base_index = pixel_index * 3;  // multiple * 3 for RGB components
//     atomicAdd(&image[image_base_index + 0], sample_color.r);
//     atomicAdd(&image[image_base_index + 1], sample_color.g);
//     atomicAdd(&image[image_base_index + 2], sample_color.b);

//     //* Update the random state
//     // states[thread_index] = local_state;

//     // color *= cam->pixel_sample_scale;
//     // image[cam->image_width * j + i] = colorToUint32_t(color);  
// }


// __global__ void rayTracer_kernel_shared(curandState_t* states, Camera* cam, float* image, hittable* world) {
//     // Define block and thread indices
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x * blockDim.x;
//     int by = blockIdx.y * blockDim.y;

//     // Calculate pixel coordinates
//     int i = bx + tx;
//     int j = by + ty;

//     if (i >= cam->image_width || j >= cam->image_height)
//         return;

//     // Shared memory for accumulating colors
//     extern __shared__ float shared_colors[];

//     // Each pixel has 3 color components (R, G, B)
//     int local_idx = (ty * blockDim.x + tx) * 3;
//     shared_colors[local_idx + 0] = 0.0f;
//     shared_colors[local_idx + 1] = 0.0f;
//     shared_colors[local_idx + 2] = 0.0f;

//     __syncthreads();

//     // Calculate global thread index for random state
//     int idx = j * cam->image_width + i;

//     // Total threads in the block
//     int total_threads_in_block = blockDim.x * blockDim.y;
//     int thread_id_in_block = ty * blockDim.x + tx;

//     // Distribute samples among threads
//     int samples_per_thread = cam->samples_per_pixel / total_threads_in_block;
//     int leftover_samples = cam->samples_per_pixel % total_threads_in_block;

//     // Assign leftover samples to first 'leftover_samples' threads
//     if (thread_id_in_block < leftover_samples)
//         samples_per_thread++;

//     // Accumulate color contributions
//     for (int s = 0; s < samples_per_thread; ++s) {
//         // Generate ray and compute color
//         ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center,
//                         cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle,
//                         cam->defocus_disk_u, cam->defocus_disk_v);
//         glm::vec3 color = ray_color(states, i, j, cam->max_depth, cam->background, r, world);

//         // Accumulate color in shared memory
//         atomicAdd(&shared_colors[local_idx + 0], color.r);
//         atomicAdd(&shared_colors[local_idx + 1], color.g);
//         atomicAdd(&shared_colors[local_idx + 2], color.b);
//     }

//     __syncthreads();

//     // Only one thread per pixel writes back to global memory
//     if (thread_id_in_block == 0) {
//         int pixel_index = j * cam->image_width + i;
//         int image_base_index = pixel_index * 3;

//         image[image_base_index + 0] += shared_colors[local_idx + 0];
//         image[image_base_index + 1] += shared_colors[local_idx + 1];
//         image[image_base_index + 2] += shared_colors[local_idx + 2];
//     }
// }

__global__ void rayTracer_kernel_shared(curandState_t* states, Camera* cam, float* image, hittable* world) {
    // Number of threads assigned to each pixel
    const int threads_per_pixel = blockDim.x;  // e.g., 32
    // Number of pixels processed per block
    const int pixels_per_block = blockDim.y;   // e.g., 8

    // Thread index within the pixel
    int thread_in_pixel = threadIdx.x;         // 0 to threads_per_pixel - 1
    // Pixel index within the block
    int pixel_in_block = threadIdx.y;          // 0 to pixels_per_block - 1

    // Compute global pixel coordinates
    int pixel_x = blockIdx.x;
    int pixel_y = blockIdx.y * pixels_per_block + pixel_in_block;

    if (pixel_x >= cam->image_width || pixel_y >= cam->image_height)
        return;

    // Index for random states
    int idx = pixel_y * cam->image_width + pixel_x;

    // Shared memory index for this pixel
    extern __shared__ float shared_colors[];
    int shared_mem_idx = pixel_in_block * 3;

    // Initialize shared memory for the pixel (one thread does this)
    if (thread_in_pixel == 0) {
        shared_colors[shared_mem_idx + 0] = 0.0f;
        shared_colors[shared_mem_idx + 1] = 0.0f;
        shared_colors[shared_mem_idx + 2] = 0.0f;
    }
    __syncthreads();

    // Calculate samples per thread
    int samples_per_thread = cam->samples_per_pixel / threads_per_pixel;
    int leftover_samples = cam->samples_per_pixel % threads_per_pixel;
    if (thread_in_pixel < leftover_samples)
        samples_per_thread++;

    // Accumulate color contributions
    glm::vec3 color(0.0f);
    for (int s = 0; s < samples_per_thread; ++s) {
        // Generate ray and compute color
        ray r = get_ray(states, pixel_x, pixel_y, cam->pixel00_loc, cam->center,
                        cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle,
                        cam->defocus_disk_u, cam->defocus_disk_v);
        color += ray_color(states, pixel_x, pixel_y, cam->max_depth, cam->background, r, world);
    }

    // Accumulate color in shared memory using atomic operations (on shared memory)
    atomicAdd(&shared_colors[shared_mem_idx + 0], color.r);
    atomicAdd(&shared_colors[shared_mem_idx + 1], color.g);
    atomicAdd(&shared_colors[shared_mem_idx + 2], color.b);

    __syncthreads();

    // One thread per pixel writes the accumulated color back to global memory
    if (thread_in_pixel == 0) {
        int pixel_index = pixel_y * cam->image_width + pixel_x;
        int image_base_index = pixel_index * 3;

        image[image_base_index + 0] += shared_colors[shared_mem_idx + 0];
        image[image_base_index + 1] += shared_colors[shared_mem_idx + 1];
        image[image_base_index + 2] += shared_colors[shared_mem_idx + 2];
    }
}


// __global__ void rayTracer_kernel_no_atomic(curandState_t* states, Camera* cam, float* image, hittable* world) {
//     // Calculate pixel coordinates
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= cam->image_width || j >= cam->image_height)
//         return;

//     int pixel_index = j * cam->image_width + i;
//     int image_base_index = pixel_index * 3;

//     glm::vec3 color = glm::vec3(0.0f);

//     int idx = pixel_index;

//     // Each thread processes all samples for its pixel
//     for (int sample = 0; sample < cam->samples_per_pixel; ++sample) {
//         // Generate ray and compute color
//         ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center,
//                         cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle,
//                         cam->defocus_disk_u, cam->defocus_disk_v);
//         color += ray_color(states, i, j, cam->max_depth, cam->background, r, world);
//     }

//     // Write the color to the image buffer without atomic operations
//     image[image_base_index + 0] = color.r;
//     image[image_base_index + 1] = color.g;
//     image[image_base_index + 2] = color.b;
// }


__global__ void rayTracer_kernel_batched(curandState_t* states, Camera* cam, float* image, hittable* world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam->image_width || j >= cam->image_height)
        return;

    int pixel_index = j * cam->image_width + i;

    glm::vec3 color = glm::vec3(0.0f);

    // Process samples in a loop
    for (int sample = 0; sample < cam->samples_per_pixel; ++sample) {
        // Generate ray and compute color
        ray r = get_ray(states, i, j, cam->pixel00_loc, cam->center, cam->pixel_delta_u, cam->pixel_delta_v, cam->defocus_angle, cam->defocus_disk_u, cam->defocus_disk_v);
        color += ray_color(states, i, j, cam->max_depth, cam->background, r, world);
    }
    // Write the color to the image buffer
    int image_base_index = pixel_index * 3;

    atomicAdd(&image[image_base_index + 0], color.r);
    atomicAdd(&image[image_base_index + 1], color.g);
    atomicAdd(&image[image_base_index + 2], color.b);
}

// __global__ void rayTracer_kernel_no_batches(
//     curandState_t* states,
//     Camera* cam,
//     float* image,
//     hittable* world
// ) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i >= cam->image_width || j >= cam->image_height)
//         return;

//     int pixel_index = j * cam->image_width + i;

//     glm::vec3 color = glm::vec3(0.0f);

//     // Process samples in a loop
//     for (int sample = 0; sample < cam->samples_per_pixel; ++sample) {
//         // Generate ray and compute color
//         ray r = get_ray(
//             states,
//             i, j,
//             cam->pixel00_loc,
//             cam->center,
//             cam->pixel_delta_u,
//             cam->pixel_delta_v,
//             cam->defocus_angle,
//             cam->defocus_disk_u,
//             cam->defocus_disk_v
//         );

//         color += ray_color(
//             states,
//             i, j,
//             cam->max_depth,
//             cam->background,
//             r,
//             world
//         );
//     }
//     // Write the color to the image buffer
//     int image_base_index = pixel_index * 3;
//     image[image_base_index + 0] = color.r;
//     image[image_base_index + 1] = color.g;
//     image[image_base_index + 2] = color.b;
    
// }


__global__ void addToColorBuffer(float* data, uint32_t* image, int width, int height, float scale){
    // Calculate pixel coordinates
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Ensure we don't go out of bounds
    if (x >= width || y >= height) return;

    // Compute the linear index
    int idx = y * width + x;
    int color_idx = idx * 3;

    // Correctly extract RGB components
    glm::vec3 color(data[color_idx + 0], data[color_idx + 1], data[color_idx + 2]);

    // Apply scaling
    color *= scale;

    // Convert color to uint32_t
    image[idx] = colorToUint32_t(color);
}



// scenes 1 - 10
void bouncing_spheres(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

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

void checkered_spheres(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

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

void earth(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

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

void perlin_spheres(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

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

void quads(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 80.0f;
    cam.lookfrom = glm::vec3( 0.0f, 0.0f,  9.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;
    
    auto left_red = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 0.2, 0.2)));
    auto back_green = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 1.0, 0.2)));
    auto right_blue = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.2, 1.0)));
    auto upper_orange = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 0.5, 0.0)));
    auto lower_teal = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.8, 0.8)));

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

void simple_light(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

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

    auto mat1 = createMaterial(memoryManager, Type::DIFFUSE, createTexture(memoryManager, Type::SOLID, glm::vec3(4.0f, 4.0f, 4.0f)) );

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


void cornell_box(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 278.0f, 278.0f, -800.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
    
    std::vector<hittable> h_hittables_list;

    // hittable hittable_obj;  // holds any hittable temporarily

    

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
    auto bvh1 = createBVH(memoryManager, box1);
    auto bvh2 = createBVH(memoryManager, box2);
    
    h_hittables_list.push_back(bvh1);
    h_hittables_list.push_back(bvh2);
    
    size_t number_of_hittables = h_hittables_list.size();

    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
  
    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
}

void cornell_box_instances(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

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
    boxes1.push_back(*translated);

    rotated = memoryManager.allocateHost<hittable>(hittable::make_rotateY(box2, -18));
    translated = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(130.0f, 0.0f, 65.0f)));
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

void cornell_smoke(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 40.0f;
    cam.lookfrom = glm::vec3( 278.0f, 278.0f, -800.0f);
    cam.lookat   = glm::vec3( 278.0f, 278.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.0f, 0.0f, 0.0f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;  // hold all items in general native and bvh items
    
    
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
    auto smoked   = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(translated, 0.01f, d_negro));
    h_hittables_list.push_back(createBVH(memoryManager, smoked));

    rotated     = memoryManager.allocateHost<hittable>(hittable::make_rotateY(box2, -18));
    translated  = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(130, 0, 65)));
    smoked     = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(translated, 0.01f, white));
    h_hittables_list.push_back(createBVH(memoryManager, smoked));
    
   
    size_t number_of_hittables = h_hittables_list.size();
    printf("size: %d\n", (int)number_of_hittables);
    
    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
    
    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);

    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
}



void finalScene(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){
    
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
    std::vector<hittable> blueGroup;
    auto boundary = memoryManager.allocateHost<hittable>(hittable::make_sphere(glm::vec3(360, 150, 145), 70, createMaterial(memoryManager, Type::DIELECTRIC, NULL, glm::vec3(0), 0, 1.5)));
    blueGroup.push_back(*boundary);
    // h_hittables_list.push_back(*boundary);
    auto blue_tex = createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.4, 0.9));
    auto blue_mat = createMaterial(memoryManager, Type::ISOTROPIC, blue_tex);
    auto blueobj = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(boundary, 0.2, blue_mat));
    blueGroup.push_back(*blueobj);
    auto bvh1 = createBVH(memoryManager, blueGroup);
    h_hittables_list.push_back(bvh1);
    // h_hittables_list.push_back(*blueobj);
    
    // ??? what sphere is this one?
    auto boundary1 = memoryManager.allocateHost<hittable>(hittable::make_sphere(glm::vec3(0, 0, 0), 5000, createMaterial(memoryManager, Type::DIELECTRIC, NULL, glm::vec3(0), 0, 1.5) ));
    auto white_tex = createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 1.0, 1.0));
    auto white_mat = createMaterial(memoryManager, Type::ISOTROPIC, white_tex);
    auto bigObj = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(boundary1, .0001f, white_mat));
    auto bvh2 = createBVH(memoryManager, bigObj);
    h_hittables_list.push_back(bvh2);
    // h_hittables_list.push_back(*memoryManager.allocateHost<hittable>(hittable::make_constantMedium(boundary1, .0001f, white_mat)));
    
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
        newSize = currentSize * 2;
        checkCuda(cudaDeviceSetLimit(cudaLimitStackSize, newSize));
         printf("New Stack Size: %d bytes\n", (int)newSize);
    }
    HybridMemoryManager memoryManager;
   
    hittable*   d_hittables_list    = memoryManager.deferDeviceAllocation<hittable>();
    hittable*   d_world             = memoryManager.deferDeviceAllocation<hittable>();
   
    switch (cam.scene)  
    {
    case 1:
        printf("Bouncing Spheres --->\t");
        bouncing_spheres(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 2:
        printf("Checkered Spheres --->\t");
        checkered_spheres(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 3:
        printf("Earth --->\t");
        earth(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 4:
        printf("Perlin Spheres --->\t");
        perlin_spheres(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 5:
        printf("Quads --->\t");
        quads(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 6:
        printf("Simple Light --->\t");
        simple_light(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 7:
        printf("Cornell Box- --->\t");
        cornell_box(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 8:
        printf("Cornell Box Instances --->\t");
        cornell_box_instances(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 9:
        printf("Smoke --->\t");
        cornell_smoke(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 10:
        printf("Final Scene --->\t");
        finalScene(cam, memoryManager, d_hittables_list, d_world);
        break;
    default:
        break;
    }
    Camera*     d_cam       = memoryManager.allocateDevice<Camera>();
    float*   d_image     = memoryManager.allocateDevice<float>(cam.image_width * cam.image_height * sizeof(float));    // for display buffer
    uint32_t* d_buffer     = memoryManager.allocateDevice<uint32_t>(cam.image_width * cam.image_height * sizeof(uint32_t));    // for display buffer
    
    memoryManager.copyToDevice(d_cam, &cam);
    
    clock_t start, stop;
    start = clock();

     // Define threads per pixel and pixels per block
    const int threads_per_pixel = 8; //16;  // Adjust as needed
    const int pixels_per_block =  8;//8;    // Adjust as needed

    // Set up block and grid sizes
    dim3 blockSize(threads_per_pixel, pixels_per_block);
    dim3 gridSize(cam.image_width,
                  (cam.image_height + pixels_per_block - 1) / pixels_per_block);

    // Calculate shared memory size
    size_t shared_mem_size = pixels_per_block * 3 * sizeof(float);

    // Allocate and initialize random states
    int num_pixels = cam.image_width * cam.image_height;
    curandState_t* d_states = memoryManager.allocateDevice<curandState_t>(num_pixels * sizeof(curandState_t));
    init_random2<<<gridSize, blockSize.y>>>(seed, d_states, cam.image_width, cam.image_height, pixels_per_block);

    // Launch the kernel
    rayTracer_kernel_shared<<<gridSize, blockSize, shared_mem_size>>>(d_states, d_cam, d_image, d_world);
    checkCuda(cudaGetLastError());






   
    // int threads = 16;  // Adjust based on your GPU's capabilities
    // dim3 blockSize(threads, threads);
    // int blocks_x = (cam.image_width + blockSize.x - 1) / blockSize.x;
    // int blocks_y = (cam.image_height + blockSize.y - 1) / blockSize.y;
    // dim3 gridSize(blocks_x, blocks_y);

    // int num_threads = threads * threads * blocks_x * blocks_y;
    // curandState_t* d_states = memoryManager.allocateDevice<curandState_t>(num_threads * sizeof(curandState_t));  // random calculations in GPU
    // init_random<<<gridSize, blockSize>>>(seed, d_states);

    // size_t shared_mem_size = blockSize.x * blockSize.y * 3 * sizeof(float);
    // rayTracer_kernel_shared<<<gridSize, blockSize, shared_mem_size>>>(d_states, d_cam, d_image, d_world);
    // checkCuda(cudaGetLastError());
    // checkCuda(cudaDeviceSynchronize());


    // const int batch_size = 32;
    // int total_samples = cam.samples_per_pixel;
    // int num_batches = (total_samples + batch_size - 1) / batch_size;
    // for (int batch = 0; batch < num_batches; ++batch) {
    //     int samples_in_batch = min(batch_size, total_samples - batch * batch_size);

    //     // Set the number of samples to process in this batch
    //     cam.samples_per_pixel = samples_in_batch;
    //     memoryManager.copyToDevice(d_cam, &cam);

    //     // Launch the kernel
    //     // rayTracer_kernel_batched<<<gridSize, blockSize>>>(d_states, d_cam, d_image, d_world);
    //     size_t shared_mem_size = blockDim.x * blockDim.y * 3 * sizeof(float);
    //     rayTracer_kernel_shared<<<gridSize, blockSize, shared_mem_size>>>(d_states, d_cam, d_image, d_world);
    //     checkCuda(cudaGetLastError());
        
    // }
    // // Restore the original samples per pixel and update scaling
    // cam.samples_per_pixel = total_samples;
    // cam.pixel_sample_scale = 1.0f / (float)total_samples;

    // Set up block and grid sizes
    dim3 blockSize1(16, 16);
    dim3 gridSize1((cam.image_width + blockSize.x - 1) / blockSize.x, (cam.image_height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    addToColorBuffer<<<gridSize1, blockSize1>>>(d_image, d_buffer, cam.image_width, cam.image_height, cam.pixel_sample_scale);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Took %f seconds with %d samples per pixel and %d max depth\n", timer_seconds, cam.samples_per_pixel, cam.max_depth);

    checkCuda(cudaMemcpy(colorBuffer, d_buffer, cam.image_width * cam.image_height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    //*...
    //* Memory manager will take care of cleaning memory allocations at exit
}
