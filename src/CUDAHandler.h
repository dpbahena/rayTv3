
#pragma once

// #include "camera.h"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
// #include <GL/glut.h>
// #include <cuda_runtime.h>
// #include <cuda_gl_interop.h>



class Camera;

class CUDAHandler {

    public:
    CUDAHandler(GLuint textureID);
    ~CUDAHandler();

    // void cudaCall(Camera& cam, uint32_t* colorBuffer);
    void updateRaytracer(Camera& cam);

    

    private:

    cudaExtent extent;
    cudaGraphicsResource_t cudaResource;
    void writeToSurface(cudaSurfaceObject_t surface);
};