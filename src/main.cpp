#include "window.hpp"
#include "renderer.hpp"
#include "camera.h"
#include "cuda_call.h"








int main(int arg, char** argv) {

   

    // Window
    Window win{"Dario", 1440, 720};
    // Window win{"Dario", 600, 400};
    Renderer myRender{win};
    // Camera cam{myRender.colorBuffer};
    Camera cam;
    // uint32_t* colorBuffer;
    RayTracer gpuOperations;

     if (arg == 6) {
        cam.samples_per_pixel = atoi(argv[1]);
        cam.max_depth = atoi(argv[2]);
        cam.scene = atoi(argv[3]); // scene to view
        cam.streams = atoi(argv[4]); // number of streams
        cam.ends = atoi(argv[5]);  // ends program after first run
        
    } else if (arg == 5) {
        cam.samples_per_pixel = atoi(argv[1]);
        cam.max_depth = atoi(argv[2]);
        cam.scene = atoi(argv[3]); // scene to view
        cam.streams = atoi(argv[4]); // number of streams
        cam.ends = false;  // ends program after first run
     
    } else {  // default valules
        cam.samples_per_pixel = 20;
        cam.max_depth = 5;
        cam.scene = 1;
        cam.streams = 1;
        cam.ends = false; // true ends program after first run
        printf("Using default values:  ./rayTracer 20 5 1 1 0 \n");
        printf("Usage:  ./raytracer <# samples per pixel: 5-500> <max depth: 5-100>  <scene: 1-10> <streams 1-100> <ends? 1-0>\n");
     }

    cam.aspect_ratio = win.getExtent().width / static_cast<float>(win.getExtent().height);
    cam.image_width = win.getExtent().width;
         
    // printf("Raytrace with %d samples with %d depth\n", cam.samples_per_pixel, cam.max_depth);

    bool rendered = false;
    bool ends = false;

    while(win.windowIsOpen() && !ends) {

        if(!rendered) {
            gpuOperations.cudaCall(cam, myRender.colorBuffer);
            myRender.render();
            rendered = true;
        }
        ends = cam.ends;
        

        
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
                        cam.scene = (cam.scene % 12) + 1;  //* Cycle between 1, 2, 3, etc or n
                        rendered = false;
                    default:
                        break;
                }
            default:
                break;
        }


    }

}