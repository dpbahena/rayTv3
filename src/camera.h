#pragma once

#include "cuda_call.h"
#include <glm/glm.hpp>


      
class Camera {
    public:

        double  aspect_ratio        = 1.0;
        int     image_width         = 100;
        int     samples_per_pixel   = 10;
        int     max_depth           = 10;
        double  vfov                = 90;   // vertical field of view
        glm::vec3 lookfrom = glm::vec3(0,0,0);   // Point camera is looking from
        glm::vec3 lookat   = glm::vec3(0,0,-1);  // Point camera is looking at
        glm::vec3   vup      = glm::vec3(0,1,0);     // Camera-relative "up" direction

        float defocus_angle = 0;  // Variation angle of rays through each pixel
        float focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

        Camera(uint32_t* colorBuffer);
        void render();
        
           

    private:
        int     image_height;   // Rendered image height
        glm::vec3  center;         // camera center
        glm::vec3  pixel00_loc;    // Location of pixel 0,0
        glm::vec3    pixel_delta_u;  // offset to pixel to the right
        glm::vec3    pixel_delta_v;  // offset to pixel below
        glm::vec3    u, v, w;        // Camera frame basis vectors
        glm::vec3   defocus_disk_u;       // Defocus disk horizontal radius
        glm::vec3   defocus_disk_v;       // Defocus disk vertical radius
        double  pixel_sample_scale;  // Color scale factor for a sum of pixel samples

        uint32_t* colorBuffer;
        RayTracer gpuOperations;




        

        void initialize();
    
        
        
            

       

        
};