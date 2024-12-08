#!/bin/bash

# Save the current directory
original_dir=$(pwd)

# Create and navigate to the /build directory
mkdir -p build
cd build || exit

# Configure the build system with CMake
# Set the build type here: Debug or Release
# cmake -DCMAKE_BUILD_TYPE=Release ..
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Return to the original directory
cd "$original_dir"
