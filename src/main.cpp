#include "window.hpp"
#include "renderer.hpp"
#include "camera.h"
#include "cuda_call.h"








int main(int arg, char** argv) {

   

    // Window
    // Window win{"Dario", 1440, 720};
    Window win{"Dario", 300, 200};
    Renderer myRender{win};
    // Camera cam{myRender.colorBuffer};
    Camera cam;
    // uint32_t* colorBuffer;
    RayTracer gpuOperations;

     if (arg == 5) {
        cam.samples_per_pixel = atoi(argv[1]);
        cam.max_depth = atoi(argv[2]);
        cam.isBvh = atoi(argv[3]); // method to use (with bvh bbox or no brute method)
        cam.scene = atoi(argv[4]); // scene to view

     } else {  // default valules
        cam.samples_per_pixel = 5;
        cam.max_depth = 2;
        cam.isBvh = false; //true;  // use bvh
        cam.scene = 8;
        printf("Using default values:  ./rayTracer 100 40 1 2\n");
        printf("Usage:  ./raytracer <# samples per pixel: 5-500> <max depth: 5-100>  <bvh?: 0-1> <scene: 1-5\n");
     }

    cam.aspect_ratio = win.getExtent().width / static_cast<float>(win.getExtent().height);
    cam.image_width = win.getExtent().width;
         
    printf("Raytrace with %d samples with %d depth\n", cam.samples_per_pixel, cam.max_depth);

    bool rendered = false;

    while(win.windowIsOpen()) {

        if(!rendered) {
            // cam.render();  // calculate the raytracing
            // cam.initialize();
            gpuOperations.cudaCall(cam, myRender.colorBuffer);


            myRender.render();
            rendered = true;
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
                    case SDLK_SPACE: // redraw display
                        cam.scene = (cam.scene % 8) + 1;  //* Cycle between 1, 2, 3, etc or n
                        rendered = false;
                    default:
                        break;
                }
            default:
                break;
        }


    }

}