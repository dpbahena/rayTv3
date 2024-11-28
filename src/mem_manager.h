#pragma once
#include <unordered_map>


#define checkCuda(result) { gpuAssert((result), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(code == cudaSuccess);
   }
}



class HybridMemoryManager {
public:
    // Containers to store pointers for host and device memory
    std::vector<void*> host_allocations;
    std::vector<void*> device_allocations;

    // A map to track deferred device pointers
    std::unordered_map<void**, void*> deferred_device_allocations;

    // Host Allocation
    template <typename T, typename... Args>
    T* allocateHost(Args&&... args) {
        T* obj = new T(std::forward<Args>(args)...); // Allocate memory
        host_allocations.push_back(obj);            // Track the allocation
        return obj;
    }

    template <typename T>
    T* allocateHost(size_t count) {
        T* obj = new T[count];               // Allocate an array
        host_allocations.push_back(obj);     // Track the allocation
        return obj;                          // Return the pointer
    }

    // Device Allocation
    template <typename T>
    T* allocateDevice(size_t count = 1) {
        T* device_ptr;
        checkCuda(cudaMalloc(&device_ptr, sizeof(T) * count));
        device_allocations.push_back(device_ptr); // Track the allocation
        return device_ptr;
    }

    // Deferred Device Allocation
    template <typename T>
    T*& deferDeviceAllocation() {
        T** device_ptr = new T*(nullptr); // Allocate on the heap
        deferred_device_allocations[reinterpret_cast<void**>(device_ptr)] = nullptr; // Track it
        return *device_ptr; // Return the dereferenced pointer
    }

    // Complete Deferred Allocation
    template <typename T>
    void allocateDeferred(T*& device_ptr, size_t count) {
        if (device_ptr == nullptr) {
            checkCuda(cudaMalloc(&device_ptr, sizeof(T) * count));
            device_allocations.push_back(device_ptr); // Track the allocation
        } else {
            throw std::runtime_error("Device pointer already allocated!");
        }
    }

    // Copy from Host to Device
    template <typename T>
    void copyToDevice(T* device_ptr, const T* host_ptr, size_t count = 1) {
        checkCuda(cudaMemcpy(device_ptr, host_ptr, sizeof(T) * count, cudaMemcpyHostToDevice));
    }

    // Cleanup: Frees all tracked allocations
    ~HybridMemoryManager() {
        // Free host allocations
        for (void* ptr : host_allocations) {
            delete static_cast<char*>(ptr); // Cast to char* to use delete
        }

        // Free device allocations
        for (void* ptr : device_allocations) {
            cudaFree(ptr);
        }

        // Clean up deferred pointers
        for (auto& [key, value] : deferred_device_allocations) {
            delete key; // Cleanup heap memory for deferred pointers
        }
    }
};
