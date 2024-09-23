cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

# Define a variable for the project name
set(PROJECT_NAME rayTracer)

# Use the variable for the project and executable names
project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Set the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find SDL2 package
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Collect all source files in the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE HEADERS "src/*.hpp" )

# Add the main executable and specify all source files
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link SDL2 library
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})

# Specify properties for CUDA compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)


# Suppress specific CUDA warning #20012
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>)
