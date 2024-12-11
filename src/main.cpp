
#include "camera.h"
#include "CUDAHandler.h"
#include "GLManager.h"
#include <stdio.h>





GLManager* glManager;
CUDAHandler* cudaHandler;


bool needsUpdate = true; // Global or static variable to track update state

// Keyboard callback function
void handleKeypress(unsigned char key, int x, int y) {
    if (key == ' ') { // Space key
        needsUpdate = true; // Set the flag to update texture
    }
}

void display() {
    if (needsUpdate) {
        cudaHandler->updateRaytracer(); // Update only when space is pressed
        needsUpdate = false; // Reset the flag
    }
    glManager->render();
}

int main(int argc, char** argv) {

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
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA OpenGL Intro");

    glManager = new GLManager(width, height);
    glManager->initializeGL();

    cam.aspect_ratio = glManager->getExtent().width / static_cast<float>(glManager->getExtent().height);
    cam.image_width = glManager->getExtent().width;

    cudaHandler = new CUDAHandler(glManager->getTextureID(), cam);
         
    printf("Raytrace with %d samples with %d depth\n", cam.samples_per_pixel, cam.max_depth);


    glutDisplayFunc(display);
    glutKeyboardFunc(handleKeypress); // Set the keyboard callback
    glutIdleFunc(display);

    glutMainLoop();

    return 0;

    
}
