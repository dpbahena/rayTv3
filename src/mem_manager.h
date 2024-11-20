#pragma once



// class MemoryManager {
//     public:
//         ~MemoryManager() {
//             // Cleanup all host and device pointers
//             clear();
//         }
//         //* Allocate a new host hittable object
//         template <typename T>
//         T* allocateHost(T* object) {
//             host_allocations.push_back(object);
//             return object;
//         }

//         //* Allocate a new device object using cudaMalloc
//         template <typename T>
//         T* allocateDevice(size_t count = 1) {
//             T* devicePtr;
//             cudaMalloc(&devicePtr, sizeof(T) * count);
//             device_allocations.push_back(static_cast<void*>(devicePtr));
//             return devicePtr;
//         }


//         // Register an existing device pointer for later deallocations
//         void registerDevicePointer(void* devicePtr) {
//             device_allocations.push_back(devicePtr);
//         }

//         // Clear Allocations
//         void clear() {
//             // Free all host allocations
//             for (auto ptr : host_allocations) {
//                 delete ptr;
//             }
//             host_allocations.clear();

//             // Free all device allocations
//             for (auto ptr : device_allocations) {
//                 cudaFree(ptr);
//             }
//             device_allocations.clear();
//         }


//     private:
//         std::vector<void*> device_allocations;  //* stores device pointers
//         std::vector<void*> host_allocations;    //* stores host pointers


// };