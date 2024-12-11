#include "CUDAHandler.h"
#include "hittable.h"
#include "sphere.h"
#include "mem_manager.h"
#include "texture.h"

#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
// #include <cuda_gl_interop.h>

#define MAX_STACK_SIZE 20


__device__ inline glm::vec3 random_on_hemisphere(curandStatePhilox4_32_10_t* rngState,const glm::vec3& normal);
__device__ inline glm::vec3 random_in_unit_sphere(curandStatePhilox4_32_10_t* rngState);
__device__ inline glm::vec3 random_vector_in_range(curandStatePhilox4_32_10_t* rngState, float min, float max);
__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n);
__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat);
__device__ inline glm::vec3 random_in_unit_disk(curandStatePhilox4_32_10_t* rngState);
__device__ inline glm::vec3 defocus_disk_sample(curandStatePhilox4_32_10_t* rngState, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v);
__device__ inline glm::vec3 random_unit_vector(curandStatePhilox4_32_10_t* rngState);
__device__ inline bool      near_zero(const glm::vec3 v);
__device__ inline float     reflectance(float cosine, float refraction_index);
__device__ inline float     random_float(curandStatePhilox4_32_10_t* state);
__device__ inline float     random_float_in_range(curandStatePhilox4_32_10_t* rngState, float a, float b);
__device__ inline glm::vec3 sample_square(curandStatePhilox4_32_10_t* rngState);

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

texture* getTextureFromModel(HybridMemoryManager& memoryManager, Texture& tex) {

    texture* d_texture = memoryManager.allocateDevice<texture>();
    int size = tex.width * tex.height *  tex.pixel_size;
    unsigned char* d_bdata = memoryManager.allocateDevice<unsigned char>(size);
    memoryManager.copyToDevice(d_bdata, tex.bdata, size);
    auto h_image = texture::image_texture(d_bdata, tex.width, tex.height, tex.scanline, tex.pixel_size);
    memoryManager.copyToDevice(d_texture, &h_image);

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
 * @brief Creates a vector of a 3D box (six sides) that contains the two opposites vertices a & b
 * 
 * @param sides 
 * @param a 
 * @param b 
 * @param mat 
 */
inline hittable* createModel(HybridMemoryManager& memoryManager, Builder0& builder, glm::vec3 offset) {
    // Allocate raw memory for 6 hittable objects
    int number_of_triangles = builder.indices.size();
    hittable* triangles = static_cast<hittable*>(::operator new[](sizeof(hittable) * number_of_triangles)); //  1 object for now
    memoryManager.host_allocations.push_back(triangles); // Track allocation for cleanup    
    AaBb bbox;


    for (auto& v : builder.vertices){
        v.point += offset;
    }

    for (int i = 0; i < number_of_triangles; i++){
        int q = builder.indices[i].indices[0];
        int u = builder.indices[i].indices[1];
        int v = builder.indices[i].indices[2];
        auto Q = glm::vec3(builder.vertices[q].point);
        auto U = glm::vec3(builder.vertices[u].point);
        auto V = glm::vec3(builder.vertices[v].point);
        auto QU = U - Q;
        auto QV = V - Q;
        auto color = builder.indices[i].color;
        auto mat = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, color));
        new (&triangles[i]) hittable(hittable::make_triangle(Q, QU, QV, mat)); 
        bbox = AaBb(bbox, triangles[i].triangle.bounding_box());
    }

    auto model = memoryManager.allocateHost<hittable>(hittable::make_hittableList(triangles, number_of_triangles)); 
    model->hittableList.bbox = bbox;
    return model;
}

/**
 * @brief Creates a vector of a 3D box (six sides) that contains the two opposites vertices a & b
 * 
 * @param sides 
 * @param a 
 * @param b 
 * @param mat 
 */
// inline hittable* createModel(HybridMemoryManager& memoryManager, Builder& builder, glm::vec3 offset, glm::vec3 scale=glm::vec3(1.0f), texture* tex=nullptr) {
//     // Allocate raw memory for 6 hittable objects
//     int number_of_triangles = builder.indices.size() / 3;
//     hittable* triangles = static_cast<hittable*>(::operator new[](sizeof(hittable) * number_of_triangles)); //  1 object for now
//     memoryManager.host_allocations.push_back(triangles); // Track allocation for cleanup    
//     AaBb bbox;


//     for (auto& v : builder.vertices){
//         v.position += offset;
//         v.position *= scale;
//     }
//     int j = 0;
//     for (int i = 0; i < builder.indices.size() ; i+=3, j++){
//         int q = builder.indices[i + 0];
//         int u = builder.indices[i + 1];
//         int v = builder.indices[i + 2];
//         auto Q = glm::vec3(builder.vertices[q].position);
//         auto U = glm::vec3(builder.vertices[u].position);
//         auto V = glm::vec3(builder.vertices[v].position);
//         auto QU = U - Q;
//         auto QV = V - Q;
//         auto color = builder.vertices[q].color;
//         if(!tex){
//             tex = createTexture(memoryManager, Type::SOLID, color);
//         }
//         auto mat = createMaterial(memoryManager, Type::LAMBERTIAN, tex);
//         new (&triangles[j]) hittable(hittable::make_triangle(Q, QU, QV, mat)); 
//         bbox = AaBb(bbox, triangles[j].triangle.bounding_box());
//     }

//     auto model = memoryManager.allocateHost<hittable>(hittable::make_hittableList(triangles, number_of_triangles)); 
//     model->hittableList.bbox = bbox;
//     return model;
// }

inline hittable* createModel(HybridMemoryManager& memoryManager, Builder& builder, glm::vec3 offset, glm::vec3 scale=glm::vec3(1.0f), texture* tex=nullptr, material* otherMaterial=nullptr) {
    // Allocate raw memory for 6 hittable objects
    int number_of_triangles = builder.indices.size() / 3;
    hittable* triangles = static_cast<hittable*>(::operator new[](sizeof(hittable) * number_of_triangles)); //  1 object for now
    memoryManager.host_allocations.push_back(triangles); // Track allocation for cleanup    
    AaBb bbox;

    // Map to store materials to avoid duplications
    std::unordered_map<int, material*> materialMap;
    texture* deffuseTex;

    for (auto& v : builder.vertices){
        v.position += offset;
        v.position *= scale;
        
    }
    int j = 0;
    for (int i = 0; i < builder.indices.size() ; i+=3, j++){
        int q = builder.indices[i + 0];
        int u = builder.indices[i + 1];
        int v = builder.indices[i + 2];   //0, 2, 1 order

        const auto& vertex0 = builder.vertices[q];
        const auto& vertex1 = builder.vertices[u];
        const auto& vertex2 = builder.vertices[v];


        auto Q = glm::vec3(vertex0.position);
        auto U = glm::vec3(vertex1.position);
        auto V = glm::vec3(vertex2.position);
        auto QV = V - Q;
        auto QU = U - Q;
        

        // get material ID
        int material_id = vertex0.material_id;
        const auto& builderMaterial = builder.materialList[material_id]; 
        if (vertex1.material_id != material_id || vertex2.material_id != material_id){
            std::cerr << "Warning: Triangle vertices have different material IDs." << std::endl;
        }

                
        if (builderMaterial.diffuseTexture.bdata && material_id >= 0 && material_id < builder.materialList.size()){
            auto textureData = builderMaterial.diffuseTexture;
            deffuseTex = getTextureFromModel(memoryManager, textureData); 

        } else if(tex){ // provide texture not from model file

            deffuseTex = tex;

        } else {    // just get the color of the model by vertex (if anything assigned to)
            auto color = vertex0.color;
            deffuseTex = createTexture(memoryManager, Type::SOLID, color);

        } 
         
        // create material
        if (otherMaterial)  //overrides loaded diffuseTexture
            new (&triangles[j]) hittable(hittable::make_triangle(Q, QU, QV, otherMaterial));  
        else {
            auto mat = createMaterial(memoryManager, Type::LAMBERTIAN, deffuseTex);
            new (&triangles[j]) hittable(hittable::make_triangle(Q, QU, QV, mat)); 
        }
        triangles[j].triangle.uv0 = vertex0.uv;
        triangles[j].triangle.uv1 = vertex1.uv;
        triangles[j].triangle.uv2 = vertex2.uv;

        bbox = AaBb(bbox, triangles[j].triangle.bounding_box());
    }

    auto model = memoryManager.allocateHost<hittable>(hittable::make_hittableList(triangles, number_of_triangles)); 
    model->hittableList.bbox = bbox;
    return model;
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

//* bubble effect
inline hittable* createSparsedSpheres(HybridMemoryManager& memoryManager, const glm::vec3& a, const glm::vec3& b, material* mat, int numSpheres, float factor=0.09) {
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
    std::uniform_real_distribution<float> radiusDist(factor, glm::length(b - a) * factor); // Spheres' radii

    // Vector to store centers for collision avoidance
    std::vector<glm::vec3> sphereCenters;

    for (int i = 0; i < numSpheres; ++i) {
        glm::vec3 center;
        float radius;
        bool isValidPosition = false;

        // Attempt to find a valid position for the sphere
        for (int attempts = 0; attempts < 100; ++attempts) {
            center = glm::vec3(distX(gen), distY(gen), distZ(gen));
            radius = radiusDist(gen);

            // Check for overlap with existing spheres
            isValidPosition = true;
            for (const auto& existingCenter : sphereCenters) {
                if (glm::length(center - existingCenter) < radius * 2.0f) {
                    isValidPosition = false;
                    break;
                }
            }

            // If valid, break out of the attempts loop
            if (isValidPosition) break;
        }

        // If no valid position is found after several attempts, skip this sphere
        if (!isValidPosition) continue;

        // Store the new sphere's center
        sphereCenters.push_back(center);

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
    auto conglomerate = memoryManager.allocateHost<hittable>(hittable::make_hittableList(spheres, static_cast<int>(sphereCenters.size())));
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

hittable createBVHSoA(HybridMemoryManager& memoryManager, std::vector<hittable> group) {
    auto d_group = memoryManager.allocateDevice<hittable>(group.size());
    memoryManager.copyToDevice(d_group, group.data(), group.size());
    int number_of_nodes = (2 * group.size() - 1);
    auto h_nodeSoA = memoryManager.allocateHost<NodeSoA>();
    // allocate individual arrays of the structure
    h_nodeSoA->bbox = memoryManager.allocateDevice<AaBb>(number_of_nodes);
    h_nodeSoA->left_child_index = memoryManager.allocateDevice<int>(number_of_nodes);
    h_nodeSoA->right_child_index = memoryManager.allocateDevice<int>(number_of_nodes);
    h_nodeSoA->rope_index = memoryManager.allocateDevice<int>(number_of_nodes);
    h_nodeSoA->is_leaf = memoryManager.allocateDevice<bool>(number_of_nodes);
    h_nodeSoA->start = memoryManager.allocateDevice<size_t>(number_of_nodes);
    h_nodeSoA->end = memoryManager.allocateDevice<size_t>(number_of_nodes);
    h_nodeSoA->object_index = memoryManager.allocateDevice<int>(number_of_nodes);

    // auto d_nodeSoA = memoryManager.allocateDevice<NodeSoA>();
    // memoryManager.copyToDevice(d_nodeSoA, h_nodeSoA);
    
    return hittable::make_bvhNode(h_nodeSoA, d_group, group.size());
}

hittable createBVHSoA(HybridMemoryManager& memoryManager, hittable* object) {
    auto d_object = memoryManager.allocateDevice<hittable>(1);
    memoryManager.copyToDevice(d_object, object, 1);
    int number_of_nodes = (2 * 1 - 1);
    auto h_nodeSoA = memoryManager.allocateHost<NodeSoA>();
    // allocate individual arrays of the structure
    h_nodeSoA->bbox = memoryManager.allocateDevice<AaBb>(number_of_nodes);
    h_nodeSoA->left_child_index = memoryManager.allocateDevice<int>(number_of_nodes);
    h_nodeSoA->right_child_index = memoryManager.allocateDevice<int>(number_of_nodes);
    h_nodeSoA->rope_index = memoryManager.allocateDevice<int>(number_of_nodes);
    h_nodeSoA->is_leaf = memoryManager.allocateDevice<bool>(number_of_nodes);
    h_nodeSoA->start = memoryManager.allocateDevice<size_t>(number_of_nodes);
    h_nodeSoA->end = memoryManager.allocateDevice<size_t>(number_of_nodes);
    h_nodeSoA->object_index = memoryManager.allocateDevice<int>(number_of_nodes);

    // auto d_nodeSoA = memoryManager.allocateDevice<NodeSoA>();
    // memoryManager.copyToDevice(d_nodeSoA, h_nodeSoA);
    
    return hittable::make_bvhNode(h_nodeSoA, d_object, 1);
}




// void setNodesAsSoA(HybridMemoryManager& memoryManager, hittable* object){

//     // allocate host memory
//     object->bvhNode.nodeSoA = memoryManager.allocateHost<NodeSoA>();


//     auto nodeSoA = memoryManager.allocateHost<NodeSoA>();

//     int size = 2 * object->bvhNode.objects_size -1;
//     // int size = object->bvhNode.objects_size;
//     // int size = 10;
//     auto& bbox = nodeSoA->bbox;
    
//     bbox = memoryManager.allocateDevice<AaBb>(size);
//     auto& left_child_index = nodeSoA->left_child_index;
//     left_child_index = memoryManager.allocateDevice<int>(size);
//     auto& right_child_index = nodeSoA->right_child_index;
//     right_child_index = memoryManager.allocateDevice<int>(size);
//     auto& rope_index = nodeSoA->rope_index;
//     rope_index = memoryManager.allocateDevice<int>(size);
//     auto& is_leaf = nodeSoA->is_leaf;
//     is_leaf = memoryManager.allocateDevice<bool>(size);
//     auto& start = nodeSoA->start;
//     start = memoryManager.allocateDevice<size_t>(size);
//     auto& end = nodeSoA->end;
//     end = memoryManager.allocateDevice<size_t>(size);
//     auto& object_index = nodeSoA->object_index;
//     object_index = memoryManager.allocateDevice<int>(size);
    

//     for (int i = 0; i < size; i++) {
        
//         bbox[i] = object->bvhNode.nodes[i].bbox;
//         left_child_index[i] = object->bvhNode.nodes[i].left_child_index;
//         right_child_index[i] = object->bvhNode.nodes[i].right_child_index;
//         rope_index[i] = object->bvhNode.nodes[i].rope_index;
//         is_leaf[i] = object->bvhNode.nodes[i].is_leaf;
//         start[i] = object->bvhNode.nodes[i].start;
//         end[i] = object->bvhNode.nodes[i].end;
//         object_index[i] = object->bvhNode.nodes[i].object_index;
//     }

//     object->bvhNode.nodeSoA = nodeSoA;
    
// }



__device__
static bool lambertian_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, lambertian_data& lambertian, curandStatePhilox4_32_10_t& rngState) {
    auto scatter_direction = rec.normal + random_unit_vector(&rngState);

    // Catch degenerate scatter direction
    if (near_zero(scatter_direction))
        scatter_direction = rec.normal;

    scattered = ray(rec.p, scatter_direction, r_in.time());
    // attenuation = lambertian.albedo;
    attenuation = lambertian.tex->value(rec.u, rec.v, rec.p);
    
    return true;
}

__device__
static bool metal_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, metal_data& metal, curandStatePhilox4_32_10_t& rngState) {
    glm::vec3 reflected = reflect(r_in.direction, rec.normal);
    reflected = glm::normalize(reflected) + (metal.fuzz * random_unit_vector(&rngState));
    scattered = ray(rec.p, reflected, r_in.time());
    attenuation = metal.albedo;

    return (glm::dot(scattered.direction, rec.normal) > 0);
}

__device__
static bool dielectric_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, dielectric_data& dielectric, curandStatePhilox4_32_10_t& rngState) {
    attenuation = glm::vec3(1.0, 1.0, 1.0);
    float ri = rec.front_face ? (1.0f / dielectric.refraction_index) : dielectric.refraction_index;

    glm::vec3 unit_direction = glm::normalize(r_in.direction);
    float cos_theta = min(glm::dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    bool cannot_refract = ri * sin_theta > 1.0f;
    glm::vec3 direction;

    
    if (cannot_refract || reflectance(cos_theta, ri) > random_float(&rngState) )
        direction = reflect(unit_direction, rec.normal);
    else   
        direction = refract(unit_direction, rec.normal, ri);

    scattered = ray(rec.p, direction, r_in.time());
    return true;
}

__device__ __host__
static glm::vec3 emitted(float u, float v, const glm::vec3& p, diffuseLight_data& diffuse){
    return diffuse.tex->value(u, v, p);
}

__device__ 
static bool isotropic_scatter(const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, isotropic_data& isotropic, curandStatePhilox4_32_10_t& rngState){
    scattered = ray(rec.p, random_unit_vector(&rngState), r_in.time());
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
inline float random_float(curandStatePhilox4_32_10_t* rngState){
    return curand_uniform(rngState);
}

__device__ 
inline glm::vec3 random_unit_vector(curandStatePhilox4_32_10_t* rngState){
    auto p = random_in_unit_sphere(rngState);
    return glm::normalize(p);
}

__device__
inline bool near_zero(const glm::vec3 v) {
    auto s = 1e-8f;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}


__device__
inline glm::vec3 sample_square(curandStatePhilox4_32_10_t* rngState) {
    
    auto a = random_float(rngState) - 0.5f;  
    auto b = random_float(rngState) - 0.5f;
    
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
inline glm::vec3 random_on_hemisphere(curandStatePhilox4_32_10_t* rngState,const glm::vec3& normal) {
    glm::vec3 on_unit_sphere = random_unit_vector(rngState);
    if (glm::dot(on_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__
inline glm::vec3 random_in_unit_sphere(curandStatePhilox4_32_10_t* rngState) {
    while (true) {
        glm::vec3 p = random_vector_in_range(rngState, -1.0f ,1.0f);
        if (glm::dot(p,p) < 1.0f){
            return p;
        }
    }
}

__device__ inline float clamp(float value, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(value, maxVal));
}


__device__
inline glm::vec3 random_in_unit_disk(curandStatePhilox4_32_10_t* rngState){
        
    while (true) {
        auto p = glm::vec3(random_float_in_range(rngState, -1, 1), random_float_in_range(rngState, -1, 1), 0);
        if (glm::dot(p,p) < 1.0f)
            return p;
    }
}

 __device__
inline glm::vec3 defocus_disk_sample(curandStatePhilox4_32_10_t* rngState, glm::vec3& center, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v) {
    // returns a random point in the camera defocus disk
    glm::vec3 p = random_in_unit_disk(rngState);
    return center + p.x * defocusDisk_u + p.y * defocusDisk_v;
}

__device__
inline glm::vec3 random_vector_in_range(curandStatePhilox4_32_10_t* rngState, float min, float max){
    
    float a = random_float_in_range(rngState, min, max);
    float b = random_float_in_range(rngState, min, max);
    float c = random_float_in_range(rngState, min, max);
  
    return glm::vec3(a, b, c);
}


__device__ inline float random_float_in_range(curandStatePhilox4_32_10_t* rngState, float a, float b) {
    // return a + (b - a) * curand_uniform_float(state);  // this does not include b  e.g -1 to 1.0  it does not include 1.0
    return a + (b - a) * (curand_uniform(rngState) - 0.5) * 2.0;  // this approach includes the upper limit   -1 to 1.0  it includes 1.0
}

/**
 * @return a random integer in [min, max] including the upper limit
 */
__device__ inline int random_int(curandState_t* state, int a, int b) {
    return static_cast<int>(a + (b - a) * (curand_uniform_double(state) - 0.5) * 2.0);  // this approach includes the upper limit   -1 to 1.0  it includes 1.0
}

__device__
inline glm::vec3 ray_color(curandStatePhilox4_32_10_t& rngState, int depth, const glm::vec3& background, const ray &r, const hittable* world) {
    ray cur_ray = r;
    glm::vec3 cur_attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 final_color     = glm::vec3(0.0f, 0.0f, 0.0f);
    
    // Loop through the ray bounces up to the specified depth
    #pragma unroll
    for (int k = 0; k < depth; k++){
        hit_record rec;
        
        float randNumber = random_float(&rngState);
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
        switch(rec.mat->type){
            case Type::METAL:
                did_scatter = metal_scatter(cur_ray, rec, attenuation, scattered, rec.mat->metal, rngState);
                break;
            case Type::LAMBERTIAN:
                did_scatter = lambertian_scatter(cur_ray, rec, attenuation, scattered, rec.mat->lambertian, rngState);
                break;
            case Type::DIELECTRIC:
                did_scatter = dielectric_scatter(cur_ray, rec, attenuation, scattered, rec.mat->dielectric, rngState);        
                break;
            case Type::ISOTROPIC:
                did_scatter = isotropic_scatter(cur_ray, rec, attenuation, scattered, rec.mat->isotropic, rngState);        
                break;
            default:
                return final_color;
                break;
        }

        // if (rec.mat->type == Type::METAL){
        //     did_scatter = metal_scatter(cur_ray, rec, attenuation, scattered, rec.mat->metal, rngState);
        // } else if (rec.mat->type == Type::LAMBERTIAN){
        //     did_scatter = lambertian_scatter(cur_ray, rec, attenuation, scattered, rec.mat->lambertian, rngState);
        // } else if (rec.mat->type == Type::DIELECTRIC){
        //     did_scatter = dielectric_scatter(cur_ray, rec, attenuation, scattered, rec.mat->dielectric, rngState);    
        // } else if (rec.mat->type == Type::ISOTROPIC){
        //     did_scatter = isotropic_scatter(cur_ray, rec, attenuation, scattered, rec.mat->isotropic, rngState);    
        // }
        //* If scattering did not occur, return the accumulated color
        // if(!did_scatter) {
        //     return final_color;
        // }
        
        //* Update the current ray and attenuation for the next bounce
        cur_ray = scattered;
        cur_attenuation *= attenuation;
    }
    
    // Return the accumulated color after all bounces
    return final_color;
}


__device__
inline ray get_ray(curandStatePhilox4_32_10_t& rngState, glm::vec3& pixel00_loc, glm::vec3& cameraCenter, glm::vec3& delta_u, glm::vec3& delta_v, float& defocusAngle, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v, int i, int j) {
    /* Construct a camara ray originating from the defocus disk and directed at a randdomly sampled point around the pixel locations i, j */
    
        
    auto offset = sample_square(&rngState);
    auto pixel_sample = pixel00_loc 
                        + ((i + offset.x) * delta_u)
                        + ((j + offset.y) * delta_v);
    
    auto ray_origin =  (defocusAngle <= 0) ? cameraCenter : defocus_disk_sample(&rngState, cameraCenter, defocusDisk_u, defocusDisk_v);
    auto ray_direction = pixel_sample - ray_origin;
    
    auto ray_time = random_float(&rngState);
    

    return ray(ray_origin, ray_direction, ray_time);
}

// Use high-resolution clock to generate a seed
unsigned long long seed = static_cast<unsigned long long>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()
);



__global__ void rayTracer_kernel_shared(
    cudaSurfaceObject_t surface,
    Camera* cam,
    hittable* world,
    unsigned long long seed)
{
    // Thread indices
    int thread_in_pixel = threadIdx.x;  // Threads per pixel
    int pixel_in_block = threadIdx.y;   // Pixels per block

    // Compute pixel coordinates
    int pixel_x = blockIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + pixel_in_block;

    if (pixel_x >= cam->image_width || pixel_y >= cam->image_height)
        return;

    // Shared memory index
    extern __shared__ float shared_colors[];
    int shared_mem_idx = pixel_in_block * 3;

    // Initialize shared memory
    if (thread_in_pixel == 0) {
        shared_colors[shared_mem_idx + 0] = 0.0f;
        shared_colors[shared_mem_idx + 1] = 0.0f;
        shared_colors[shared_mem_idx + 2] = 0.0f;
    }
    __syncthreads();

    // Samples per thread
    int samples_per_thread = cam->samples_per_pixel / blockDim.x;
    int leftover_samples = cam->samples_per_pixel % blockDim.x;
    if (thread_in_pixel < leftover_samples)
        samples_per_thread++;

    // Unique sequence for RNG (now correctly used)
    unsigned long long sequence = ((unsigned long long)pixel_y * gridDim.x + pixel_x) * blockDim.x + thread_in_pixel;
    // Total random numbers per sample (adjust based on actual usage)
    const int N_per_sample = 256; // estimate or calculate precisely

    // Accumulate color
    glm::vec3 color(0.0f);
    int sample_base = thread_in_pixel * samples_per_thread;
    #pragma unroll
    for (int s = 0; s < samples_per_thread; ++s) {
        int sample_index = sample_base + s;

        //* Initialize RNG state once per sample
        curandStatePhilox4_32_10_t rngState;
        curand_init(seed, sequence, sample_index * N_per_sample, &rngState);

        // Generate ray using the sequence from the kernel
        ray r = get_ray(
            rngState,
            cam->pixel00_loc,
            cam->center,
            cam->pixel_delta_u,
            cam->pixel_delta_v,
            cam->defocus_angle,
            cam->defocus_disk_u,
            cam->defocus_disk_v,
            pixel_x,
            pixel_y);

        // Compute color
        color += ray_color(rngState, cam->max_depth, cam->background, r, world);
    }

    // Accumulate color in shared memory
    atomicAdd(&shared_colors[shared_mem_idx + 0], color.r);
    atomicAdd(&shared_colors[shared_mem_idx + 1], color.g);
    atomicAdd(&shared_colors[shared_mem_idx + 2], color.b);

    __syncthreads();

    // // Write back to global memory
    // if (thread_in_pixel == 0) {
    //     int pixel_index = pixel_y * cam->image_width + pixel_x;
    //     int image_base_index = pixel_index * 3;

    //     image[image_base_index + 0] += shared_colors[shared_mem_idx + 0];
    //     image[image_base_index + 1] += shared_colors[shared_mem_idx + 1];
    //     image[image_base_index + 2] += shared_colors[shared_mem_idx + 2];
    // }

    // Write back to surface memory
    if (thread_in_pixel == 0) {
        glm::vec3 final_color = glm::vec3(
            shared_colors[shared_mem_idx + 0],
            shared_colors[shared_mem_idx + 1],
            shared_colors[shared_mem_idx + 2]
        ) / static_cast<float>(cam->samples_per_pixel);

        // Apply gamma correction (if needed)
        final_color = glm::sqrt(final_color);

        // Convert to RGBA
        uchar4 rgba;
        rgba.x = static_cast<unsigned char>(clamp(final_color.r * 255.99f, 0.0f, 255.0f));
        rgba.y = static_cast<unsigned char>(clamp(final_color.g * 255.99f, 0.0f, 255.0f));
        rgba.z = static_cast<unsigned char>(clamp(final_color.b * 255.99f, 0.0f, 255.0f));
        rgba.w = 255; // Fully opaque

        // Write to the surface
        int flipped_y = cam->image_height - 1 - pixel_y;
        // surf2Dwrite(rgba, surface, pixel_x * sizeof(uchar4), pixel_y);
        surf2Dwrite(rgba, surface, pixel_x * sizeof(uchar4), flipped_y);
    }
}


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
    auto bvhItem = createBVHSoA(memoryManager, small_spheres);
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
    auto black_dot  = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(.1, .1, .1)));
    auto blue_dot = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.2, 1.0)));
    auto teal_dot = createMaterial(memoryManager, Type::LAMBERTIAN, createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.8, .8)));

    /* Quads */
    hittable hittable_obj;
    hittable_obj = hittable::make_triangle(glm::vec3(-3.0f, -2.0f, 5.0f), glm::vec3(0.0f, 0.0f, -4.0f), glm::vec3(0.0f, 4.0f,  0.0f), left_red);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_triangle(glm::vec3(-3.0f, 2.0f, 1.0f), glm::vec3(0.0f ,0.0f ,4.0f), glm::vec3(0.0f, -4.0f,  0.0f), left_red);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_sphere(glm::vec3(-00.0f, 0.0f, -0.0f),0.1, black_dot);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_sphere(glm::vec3(-1.0f,  0.0f, 3.0f),0.1, blue_dot);
    h_hittables_list.push_back(hittable_obj);
    hittable_obj = hittable::make_sphere(glm::vec3(-2.0f, 0.0f, 4.0f),0.1, teal_dot);
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

void triangles(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3( -2.0f, 1.0f,  9.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.0f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;
    
    Builder0 builder;
    builder.vertices.push_back({{-0.3f, 0.25f, -0.3}});
    builder.vertices.push_back({{0.3f, 0.25f, -0.3}});
    builder.vertices.push_back({{0.3f, 0.25f, 0.3}});
    builder.vertices.push_back({{-0.3f, 0.25f, 0.3}});
    builder.vertices.push_back({{-0.4f, -0.25, -0.4}});
    builder.vertices.push_back({{0.4f, -0.25, -0.4}});
    builder.vertices.push_back({{0.4f, -0.25, 0.4}});
    builder.vertices.push_back({{-0.4f, -0.25, 0.4}});
    builder.indices = {
        {{0, 1, 3}, {0.0f, 0.1f, 1.0f}},   // top cover     // PINK
        {{3, 1, 2}, {0.0f, 0.1f, 1.0f}},

        {{5, 4, 7}, {0.0f, 0.0f, 1.0f}},    // bottom cover
        {{7, 6, 5}, {0.0f, 0.0f, 1.0f}},

        {{0, 3, 7}, {0.0f, 1.0f, 0.0f}},    // left cover
        {{7, 4, 0}, {0.0f, 1.0f, 0.0f}},

        {{2, 1, 5}, {0.0f, 1.0f, 0.0f}},    // right cover
        {{5, 6, 2}, {0.0f, 1.0f, 0.0f}},

        {{1, 0, 5}, {1.0f, 0.0f, 0.0f}},    // back cover
        {{5, 0, 4}, {1.0f, 0.0f, 0.0f}},
        
        {{3, 2, 6}, {1.0f, 0.5f, 0.0f}},    // front cover  BLACK COLOR
        {{6, 7, 3}, {1.0f, 0.5f, 0.0f}},
    };
     

    auto offset = glm::vec3(0.0, 0.25, 0.0);
    auto obj1 = createModel(memoryManager, builder, offset);
    auto bvh1 = createBVHSoA(memoryManager, obj1);
    h_hittables_list.push_back(bvh1);

    // ground
    auto tex = createTexture(memoryManager, Type::CHECKER, glm::vec3(0.2f, 0.3f, 0.1f), glm::vec3(0.9f, 0.9f, 0.9f),{}, {}, 0.32f);
    auto ground = createMaterial(memoryManager, Type::LAMBERTIAN, tex);
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,-1000.0, 0.0), 1000, ground);
    h_hittables_list.push_back(hittable_obj);

    auto silver = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(0.7f, 0.6f, 0.5f), 0.0, {});
    hittable_obj = hittable::make_sphere(glm::vec3(2.5f, 1.0f, 0.0f), 1.0f, silver);
    h_hittables_list.push_back(hittable_obj);


    
    size_t number_of_hittables = h_hittables_list.size();

    memoryManager.allocateDeferred(d_hittable_list, number_of_hittables);
    memoryManager.copyToDevice(d_hittable_list, h_hittables_list.data(), number_of_hittables);
     

    hittable h_world = hittable::make_hittableList(d_hittable_list, number_of_hittables);
    memoryManager.allocateDeferred(d_world, 1);
    memoryManager.copyToDevice(d_world, &h_world, 1);
    
}


void loadingModels(Camera& cam, HybridMemoryManager& memoryManager, hittable* &d_hittable_list, hittable* &d_world){

    cam.vfov = 60.0f;
    cam.lookfrom = glm::vec3( -1.0f, 2.0f,  11.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f,  0.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);
    cam.defocus_angle = 0.1f;
    cam.focus_dist = 10.0f;
    cam.background = glm::vec3(0.70f, 0.80f, 1.00f);
    cam.initialize();
   
    
    std::vector<hittable> h_hittables_list;
    glm::vec3 offset, scale;
    hittable* obj1, bvh1;
    Builder builder;
    texture* tex;


    createModelFromFile(builder, "images/woodFloor.obj");
    offset = glm::vec3(0.0, 0.0, 0.0);
    scale = glm::vec3(7.0);
    obj1 = createModel(memoryManager, builder, offset, scale);
    bvh1 = createBVHSoA(memoryManager, obj1);
    // h_hittables_list.push_back(bvh1);
    
    h_hittables_list.push_back(bvh1);

    
    createModelFromFile(builder, "images/brick94Cube.obj");
    offset = glm::vec3(-2.5, 1.0, -2.0);
    scale = glm::vec3(1.0);
    obj1 = createModel(memoryManager, builder, offset, scale);
    bvh1 = createBVHSoA(memoryManager, obj1);
    h_hittables_list.push_back(bvh1);

    createModelFromFile(builder, "images/brick97Cube.obj");
    offset = glm::vec3(-3.0, 1.0, 3.0);
    scale = glm::vec3(1.0);
    obj1 = createModel(memoryManager, builder, offset, scale);
    bvh1 = createBVHSoA(memoryManager, obj1);
    h_hittables_list.push_back(bvh1);


    createModelFromFile(builder, "images/cube.obj");
    offset = glm::vec3(4.0, 1.0, 5.0);
    // offset = glm::vec3(0.0);
    scale = glm::vec3(0.5);
    // auto tex = createTexture(memoryManager, Type::IMAGE, {}, {}, "texture/WoodFloor043_4K-JPG/WoodFloor043_4K-JPG_Color.jpg");
    obj1 = createModel(memoryManager, builder, offset, scale);
    auto rotated = memoryManager.allocateHost<hittable>(hittable::make_rotateY(obj1, 45));  // apply rotation first
    auto translate = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(0, 3, 0))); // then tranlation.
    bvh1 = createBVHSoA(memoryManager, rotated);    // last, convert to bvh the rotation/translation
    h_hittables_list.push_back(bvh1);
    
    

    //ghosted monkey
    createModelFromFile(builder, "images/monkey.obj");
    offset = glm::vec3(0.0, 2.0, 7.0);
    scale = glm::vec3(1.0);
    // tex = createTexture(memoryManager, Type::IMAGE, {}, {}, "texture/Metal024_4K-JPG/Metal024_4K-JPG_Color.jpg");
    auto yellow = createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 2.0, .2));
    auto monkey = createModel(memoryManager, builder, offset, scale, yellow);
    auto dark = createTexture(memoryManager, Type::SOLID, glm::vec3(0.0f, 0.0f, 0.1f));
    auto smoke = createMaterial(memoryManager, Type::ISOTROPIC, dark);
    auto ghostMonkey = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(monkey, 1.0f, smoke));
    bvh1 = createBVHSoA(memoryManager, ghostMonkey);
    
    h_hittables_list.push_back(bvh1);

    // golden monkey
    createModelFromFile(builder, "images/monkey.obj");
    offset = glm::vec3(2.5, 2.0, 7.0);
    scale = glm::vec3(1.0);
    auto goldenMetal = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(1.0, 0.8, 0.1), 0.3);
    auto goldenMonkey = createModel(memoryManager, builder, offset, scale, {}, goldenMetal);
    bvh1 = createBVHSoA(memoryManager, goldenMonkey);
    h_hittables_list.push_back(bvh1);

    // // shiny blue monkey
    createModelFromFile(builder, "images/monkey.obj");
    auto glass3 = createMaterial(memoryManager, Type::DIELECTRIC, {}, {}, {}, 1.2f);
    offset = glm::vec3(-2.5, 2.0, 7.0);
    scale = glm::vec3(1.0);
    monkey = createModel(memoryManager, builder, offset, scale, {}, glass3);
    h_hittables_list.push_back(createBVHSoA(memoryManager, monkey));
    auto blue_tex = createTexture(memoryManager, Type::SOLID, glm::vec3(0.2, 0.4, 0.9));
    auto blue_mat = createMaterial(memoryManager, Type::ISOTROPIC, blue_tex);
    auto shinyMonkey = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(monkey, 4.0, blue_mat));
    bvh1 = createBVHSoA(memoryManager, shinyMonkey);
    h_hittables_list.push_back(bvh1);

    //* Shiny red sphere
    auto red_tex = createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 0.0, 0.0));
    auto red_mat = createMaterial(memoryManager, Type::ISOTROPIC, red_tex);
    auto boundary = memoryManager.allocateHost<hittable>(hittable::make_sphere(glm::vec3(-1.0, .75, 4.5), 0.75, createMaterial(memoryManager, Type::DIELECTRIC, NULL, glm::vec3(0), 0, 1.3)));
    h_hittables_list.push_back(*boundary);
    auto redObj = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(boundary, 12.0, red_mat));
    h_hittables_list.push_back(*redObj);

    // ground
    tex = createTexture(memoryManager, Type::CHECKER, glm::vec3(0.2f, 0.3f, 0.1f), glm::vec3(0.9f, 0.9f, 0.9f),{}, {}, 0.32f);
    auto ground = createMaterial(memoryManager, Type::LAMBERTIAN, tex);
    auto hittable_obj = hittable::make_sphere(glm::vec3(0.0,-1000.0, 0.0), 1000, ground);
    h_hittables_list.push_back(hittable_obj);

    // metal sphere
    auto silver = createMaterial(memoryManager, Type::METAL, {}, glm::vec3(0.7f, 0.6f, 0.5f), 0.0, {});
    hittable_obj = hittable::make_sphere(glm::vec3(1.0f, 1.0f, 0.0f), 1.0f, silver);
    h_hittables_list.push_back(hittable_obj);

    // glass sphere on top of brick
    auto glass = createMaterial(memoryManager, Type::DIELECTRIC, {}, {}, {}, 1.5f);
    hittable_obj = hittable::make_sphere(glm::vec3(-2.5f, 3.0, -2.0), 1.0, glass);
    h_hittables_list.push_back(hittable_obj);

    auto glass2 = createMaterial(memoryManager, Type::DIELECTRIC, {}, {}, {}, 1.05f);
    
    auto a = glm::vec3(-8, 1, 5);  // imaginary rectangle corner to opposite corner
    auto b = glm::vec3(8, 6, -5);
    auto bubbles = createSparsedSpheres(memoryManager, a, b, glass2, 30 , 0.02);
    // auto translated = memoryManager.allocateHost<hittable>(hittable::make_translate(conglomerate, glm::vec3(0, 3, -30)));
    bvh1 = createBVHSoA(memoryManager, bubbles);
    h_hittables_list.push_back(bvh1);
    
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
    auto bvh1 = createBVHSoA(memoryManager, box1);
    auto bvh2 = createBVHSoA(memoryManager, box2);
    
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
    auto bvhItem1 = createBVHSoA(memoryManager, boxes1);
    h_hittables_list.push_back(bvhItem1);
    auto bvhItem2 = createBVHSoA(memoryManager, boxes2);
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
    h_hittables_list.push_back(createBVHSoA(memoryManager, smoked));

    rotated     = memoryManager.allocateHost<hittable>(hittable::make_rotateY(box2, -18));
    translated  = memoryManager.allocateHost<hittable>(hittable::make_translate(rotated, glm::vec3(130, 0, 65)));
    smoked     = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(translated, 0.01f, white));
    h_hittables_list.push_back(createBVHSoA(memoryManager, smoked));
    
   
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
    auto bvh1 = createBVHSoA(memoryManager, blueGroup);
    h_hittables_list.push_back(bvh1);
    // h_hittables_list.push_back(*blueobj);
    
    // ??? what sphere is this one?
    auto boundary1 = memoryManager.allocateHost<hittable>(hittable::make_sphere(glm::vec3(0, 0, 0), 5000, createMaterial(memoryManager, Type::DIELECTRIC, NULL, glm::vec3(0), 0, 1.5) ));
    auto white_tex = createTexture(memoryManager, Type::SOLID, glm::vec3(1.0, 1.0, 1.0));
    auto white_mat = createMaterial(memoryManager, Type::ISOTROPIC, white_tex);
    auto bigObj = memoryManager.allocateHost<hittable>(hittable::make_constantMedium(boundary1, .0001f, white_mat));
    auto bvh2 = createBVHSoA(memoryManager, bigObj);
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

    auto bvhItem1 = createBVHSoA(memoryManager, boxesGroup);
    h_hittables_list.push_back(bvhItem1);

    auto bvhItem2 = createBVHSoA(memoryManager, conglomerateGroup);
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

CUDAHandler::CUDAHandler(GLuint textureID, Camera& cam) : cam(cam), cudaResource(NULL) 
{
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

CUDAHandler::~CUDAHandler()
{
    cudaGraphicsUnregisterResource(cudaResource);
}

// void RayTracer::cudaCall(Camera &cam, uint32_t *colorBuffer)
void CUDAHandler::updateRaytracer()
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

    //* Map the resource for CUDA
    cudaArray_t array;
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);

    //* Create a CUDA surface object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);
    
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
    case 11:
        printf("triangles --->\t");
        triangles(cam, memoryManager, d_hittables_list, d_world);
        break;
    case 12:
        printf("loading models --->\t");
        loadingModels(cam, memoryManager, d_hittables_list, d_world);
        break;
    default:
        break;
    }
    Camera*     d_cam       = memoryManager.allocateDevice<Camera>();
    
    memoryManager.copyToDevice(d_cam, &cam);
    
    clock_t start, stop;
    start = clock();

     // Define threads per pixel and pixels per block
    const int threads_per_pixel = 16;   // Adjust as needed
    const int pixels_per_block =  4;    // Adjust as needed

    // Set up block and grid sizes
    dim3 blockSize(threads_per_pixel, pixels_per_block);
    dim3 gridSize(cam.image_width,
                  (cam.image_height + pixels_per_block - 1) / pixels_per_block);

    // Calculate shared memory size
    size_t shared_mem_size = pixels_per_block * 3 * sizeof(float);


    // Launch the kernel
    rayTracer_kernel_shared<<<gridSize, blockSize, shared_mem_size>>>(surface, d_cam, d_world, seed);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Took %f seconds with %d samples per pixel and %d max depth\n", timer_seconds, cam.samples_per_pixel, cam.max_depth);

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource);

  
    //*...
    //* Memory manager will take care of cleaning memory allocations at exit
}
