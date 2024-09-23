#include "camera.h"

Camera::Camera(uint32_t* colorBuffer) : colorBuffer(colorBuffer) {}


void Camera::render() {
    initialize();
    gpuOperations.cudaCall(image_width, image_height, max_depth, center, pixel00_loc, pixel_delta_u, pixel_delta_v, samples_per_pixel, defocus_angle, defocus_disk_u, defocus_disk_v, colorBuffer);
}



void Camera::initialize() {
        // Calculate the image height and ensure that it's at least 1
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1)? 1 : image_height;

    pixel_sample_scale = 1.0f / (float)samples_per_pixel;


    center = lookfrom;

    // Determine the viewport dimensions

    // auto focal_length = glm::length(lookfrom - lookat);
    float theta = glm::radians(vfov);
    float h = tan(theta / 2.0f);
    float viewport_height = 2.0f * h * focus_dist;
    float viewport_width = viewport_height * (double(image_width) / image_height);

    /* Calculate the u,v, w unit basic vectors for the camera coordinate frame */
    w = glm::normalize(lookfrom - lookat);
    u = glm::normalize(cross(vup, w));
    v = glm::cross(w, u);


    // Calculate the vectors across the horizontal and down the vertical viewport edges
    auto viewport_u = (float)viewport_width * u;  // Vector across viewport horizontal edge
    auto viewport_v = (float)viewport_height * -v; // Vector across viewport vertical edge

    // Calculate the horizontal and vertical delta vectors from pixel to pixel
    pixel_delta_u = viewport_u / (float)image_width;
    pixel_delta_v = viewport_v / (float)image_height;

    // calculate the location fo the upper left pixel
    auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2.0f - viewport_v / 2.0f;
    pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors.
    float defocus_radius = focus_dist * tan(glm::radians(defocus_angle / 2.0f));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
    


}


    
    


