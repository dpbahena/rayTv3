#pragma once

#include <SDL2/SDL.h>
#include <string>
#include <iostream>


    typedef struct  {
        uint32_t  width;
        uint32_t height;
    }Extent2D;


    class Window {
    public:
        Window(std::string name, int w = 800, int h = 600 );
        ~Window();

        Window(const Window &) = delete;
        Window &operator=(const Window &) = delete;

    SDL_Renderer *renderer = nullptr;
    SDL_Texture *colorBufferTexture;

    // helpers
        bool windowIsOpen() { return windowIsRunning; }

    // getters
        Extent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};}
        SDL_Window* getWindow() {return window;}
    // setters
        void closeWindow() { windowIsRunning = false;}


    private:

    // main functions
        void initWindow();
        void displayMode();

    // helpers

    // variable members
        int width, height, x_winPos, y_winPos;
        


        std::string windowName;
        SDL_Window *window = nullptr;
        
        


        bool windowIsRunning = false;



    };


    Window::Window(std::string name, int w, int h) : width(w), height(h), windowName(name)
    {
        initWindow();
    
    }

    Window::~Window()
    {
        SDL_DestroyWindow(window);
        SDL_Quit();
       
        SDL_DestroyTexture(colorBufferTexture);
        SDL_DestroyRenderer(renderer);
    }

    void Window::initWindow()
    {
       if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        throw std::runtime_error ("Failed to initialize SDL window");
        }

        displayMode();

        window = SDL_CreateWindow(windowName.c_str(), x_winPos, y_winPos, width, height, 0); // 0 flag for border
        
        if (!window) {
            throw std::runtime_error("Failed creating SDL window");
        }
        renderer = SDL_CreateRenderer(window, -1, 0); // no flags
        colorBufferTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                                         SDL_TEXTUREACCESS_STREAMING,
                                                         width,
                                                         height);


        windowIsRunning = true;
    }

    void Window::displayMode()
    {
        /* Use SDL to query what is the fullscreen max widht and height and if there is 2nd display */
        SDL_Rect secondDisplayBounds;
        int numDisplays = SDL_GetNumVideoDisplays();
        if (numDisplays > 1) {
            std::cout  << "Number of Displays : " << SDL_GetNumVideoDisplays() << std::endl;
            SDL_GetDisplayBounds(1, &secondDisplayBounds); // Index 1 represents the second display
                
            width = secondDisplayBounds.w;
            height = secondDisplayBounds.h;
            // full screen
            x_winPos = secondDisplayBounds.x;
            y_winPos = secondDisplayBounds.y;

        } else {  // center the window in current display
            SDL_Rect firstDisplayBounds;
            SDL_GetDisplayBounds(0, &firstDisplayBounds);
            // printf(" 1 display %d, %d\n", firstDisplayBounds.w, firstDisplayBounds.y);
            x_winPos = firstDisplayBounds.w - width;// SDL_WINDOWPOS_CENTERED;
            y_winPos = 0; // SDL_WINDOWPOS_CENTERED;
            
        
        }



    }