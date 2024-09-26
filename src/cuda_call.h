
#pragma once
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

class RayTracer {

    public:
    RayTracer(){};



    void cudaCall(int image_width, int image_height, int max_depth,  glm::vec3 center, glm::vec3 pixel00_loc, glm::vec3 pixel_delta_u, glm::vec3 pixel_delta_v, int samples_per_pixel, float& defocusAngle, glm::vec3& defocusDisk_u, glm::vec3& defocusDisk_v, uint32_t* colorBuffer);

    

    private:
        
};