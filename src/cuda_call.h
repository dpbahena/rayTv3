
#pragma once


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include "camera.h"

class RayTracer {

    public:
    RayTracer(){};

    void cudaCall(Camera& cam, uint32_t* colorBuffer);

    private:
};