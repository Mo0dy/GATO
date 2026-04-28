#include <cstdio>

#include <cuda_runtime.h>

#include "utils/cuda.cuh"

int main()
{
    int device_count = 0;
    gpuErrchk(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "Expected at least one CUDA device.\n");
        return 1;
    }

    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDeviceProperties(&prop, 0));

    void* ptr = nullptr;
    gpuErrchk(cudaMalloc(&ptr, 1024));
    gpuErrchk(cudaFree(ptr));

    std::printf("cuda_access_smoke_ok=1 device_count=%d device_0=%s sm=%d%d\n",
                device_count,
                prop.name,
                prop.major,
                prop.minor);
    return 0;
}
