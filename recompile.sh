#!/bin/bash

# Save the current directory
original_dir=$(pwd)

# Change to the /build directory
cd build || exit

# Run make
make clean
make
# Return to the original directory
cd "$original_dir"

