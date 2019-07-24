#include <GpuInfo.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

int getCudaDeviceCount()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}



int DLL_PUBLIC GetCudaDeviceCount() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
    /*return getCudaDeviceCount();*/
}

