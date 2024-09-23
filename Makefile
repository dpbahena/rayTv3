
# TINYOBJ_INCLUDE_PATH = libraries/tinyobjloader
# STB_INCLUDE_PATH = libraries/stb
# NLOHMANN_JSON_PATH = libraries/json

# Compiler flags
# CFLAGS = -std=c++17 -Wall -g -I$(TINYOBJ_INCLUDE_PATH) -I$(STB_INCLUDE_PATH) -I$(NLOHMANN_JSON_PATH)
CFLAGS = -std=c++17 -Wall -g 
# LDFLAGS = -lSDL2 -lcurl
LDFLAGS = -lSDL2

# CUDA compiler and flags
NVCC = nvcc
# NVCCFLAGS = -std=c++17 -g  -I$(TINYOBJ_INCLUDE_PATH) -I$(STB_INCLUDE_PATH) -I$(NLOHMANN_JSON_PATH)
NVCCFLAGS = -std=c++17 -g

# Source files
CPP_SOURCES = $(wildcard src/*.cpp)
CUDA_SOURCES = $(wildcard src/*.cu)

# Directories
OBJ_DIR = build
BIN_DIR = bin

# Object files
CPP_OBJECTS = $(CPP_SOURCES:src/%.cpp=$(OBJ_DIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:src/%.cu=$(OBJ_DIR)/%.o)

# Executable name
TARGET = $(BIN_DIR)/raytracer

# Build the executable
$(TARGET): $(CPP_OBJECTS) $(CUDA_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CPP_OBJECTS) $(CUDA_OBJECTS) $(LDFLAGS)

# Compile C++ source files
$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	g++ $(CFLAGS) $(NDEBUG_FLAG) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: src/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(NDEBUG_FLAG) -c $< -o $@

# Create the object files directory
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Create the binary files directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

.PHONY: test clean rebuild all debug release

test: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET) $(OBJ_DIR) $(BIN_DIR)

rebuild: clean all

all: $(TARGET)

# Debug build (default)
debug: CFLAGS += -DDEBUG
debug: NVCCFLAGS += -DDEBUG
debug: FORCE
debug: clean $(TARGET)

# Release build (with NDEBUG defined)
release: NDEBUG_FLAG = -DNDEBUG
release: CFLAGS += -O3
release: NVCCFLAGS += -O3
release: FORCE
release: clean $(TARGET)

# Force target to ensure rebuild
.PHONY: FORCE
FORCE: