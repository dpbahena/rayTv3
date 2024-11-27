#!/bin/bash

# Change to the /build directory
# cd build || exit

# Run the executable with the provided parameters
# sudo /usr/local/cuda-12.6/bin/ncu ./build/rayTracer "$@"
sudo /usr/local/cuda-12.6/bin/ncu --kernel-name "rayTracer_kernel"  ./build/rayTracer 5 3 10 1

# Return to the original directory
# cd -

