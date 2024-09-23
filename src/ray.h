#pragma once

#include <glm/glm.hpp>

class ray {
    public:
        __host__ __device__
        ray() {}

        __device__ __host__ 
        ray(const glm::vec3& origin, const glm::vec3& direction, float time) : origin(origin), direction(direction), tm(time){}
        
        __device__ __host__ 
        ray(const glm::vec3& origin, const glm::vec3& direction) : ray(origin, direction, 0){}
        
        __host__ __device__
        float time() const { return tm; }

        __host__ __device__
        glm::vec3 at(float t) const { return origin + t * direction; }

        

        glm::vec3 origin;
        glm::vec3 direction;
        private:
        float tm;
    
};