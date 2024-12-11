// #include "window.hpp"
// #include "renderer.hpp"
#include "camera.h"
#include "CUDAHandler.h"
#include "GLManager.h"
#include <stdio.h>





GLManager* glManager;
CUDAHandler* cudaHandler;


void display() {
    // cudaHandler
    glManager->render();
}

int main(int argc, char** argv) {

   

    // Window
    // Window win{"Dario", 1440, 720};
    // Window win{"Dario", 600, 400};
    // Renderer myRender{win};
    const int width = 1440;
    const int height = 720;
    Camera cam;
    // RayTracer gpuOperations;

     if (argc == 5) {
        cam.samples_per_pixel = atoi(argv[1]);
        cam.max_depth = atoi(argv[2]);
        cam.scene = atoi(argv[3]); // scene to view
        cam.ends = atoi(argv[4]);  // ends program after first run
        
    } else if (argc == 4) {
        cam.samples_per_pixel = atoi(argv[1]);
        cam.max_depth = atoi(argv[2]);
        cam.scene = atoi(argv[3]); // scene to view
        cam.ends = false;  // ends program after first run
     
    } else {  // default valules
        cam.samples_per_pixel = 5;
        cam.max_depth = 3;
        cam.scene = 1;
        cam.ends = false; // true ends program after first run
        printf("Using default values:  ./rayTracer 100 40 1 \n");
        printf("Usage:  ./raytracer <# samples per pixel: 5-500> <max depth: 5-100>  <scene: 1-10\n");
    }

    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(800, 800);
    glutCreateWindow("CUDA OpenGL Intro");

    glManager = new GLManager(width, height);
    glManager->initializeGL();

    cam.aspect_ratio = glManager->getExtent().width / static_cast<float>(glManager->getExtent().height);
    cam.image_width = glManager->getExtent().width;

    cudaHandler = new CUDAHandler(glManager->getTextureID());
         
    printf("Raytrace with %d samples with %d depth\n", cam.samples_per_pixel, cam.max_depth);

    // bool rendered = false;
    // bool ends = false;

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


void displayCallback() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind texture
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Draw a quad with the texture
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    // Swap buffers for display
    glutSwapBuffers();
}