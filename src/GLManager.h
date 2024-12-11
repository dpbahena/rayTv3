#pragma once

#include <GL/glut.h>

typedef struct  {
    uint32_t  width;
    uint32_t height;
}Extent2D;


class GLManager {
    public:
        GLManager(int width, int height);
        ~GLManager();

        void initializeGL();
        void render();
        GLuint getTextureID() const;
        Extent2D getExtent() { return extent;}
        
        

    private:
        GLuint textureID;
        Extent2D extent;
        // int width, height;

        void createTexture();

};