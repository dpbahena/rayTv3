#include "window.hpp"
#include "renderer.hpp"
#include "camera.h"








int main(int, char**) {

    // Window
    Window win{"Dario", 1440, 720};
    Renderer myRender{win};
    Camera cam{myRender.colorBuffer};

    cam.aspect_ratio = win.getExtent().width / static_cast<float>(win.getExtent().height);
    cam.image_width = win.getExtent().width;
    cam.samples_per_pixel = 500;
    cam.max_depth = 100;

    cam.vfov = 20.0f;
    cam.lookfrom = glm::vec3(-2.0f, 2.0f,  1.0f);
    cam.lookat   = glm::vec3( 0.0f, 0.0f, -1.0f);
    cam.vup      = glm::vec3( 0.0f, 1.0f,  0.0f);

    cam.defocus_angle = 10.0f;
    cam.focus_dist = 3.4f; 
    

    bool rendered = false;

    while(win.windowIsOpen()) {

        if(!rendered) {
            clock_t start, stop;
            start = clock();
            cam.render();  // calculate the raytracing
            stop = clock();
            myRender.render();
            rendered = true;
            double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
            std::cerr << "took " << timer_seconds << " seconds with " << cam.samples_per_pixel << " samples per pixel and depth of " << cam.max_depth << "\n";
        }

        
        /* Check for keyboard input */
        SDL_Event event;   // check for keyboard input
        SDL_PollEvent(&event);

        switch (event.type)
        {
            case SDL_QUIT:
                win.closeWindow();
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        win.closeWindow();
                        break;
                    default:
                        break;
                }
            default:
                break;
        }


    }

}