#!/bin/bash

# Change to the /build directory
cd build || exit

# Run the executable with the provided parameters
./rayTracer "$@"

# Return to the original directory
cd -

