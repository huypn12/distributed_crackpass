"""
@huypn12
Read hardware info
"""

import ctypes


LIB_CPU_INFO_PATH = "helper/libs/lib_cpu_info/CpuInfo.so"
# LIB_GPU_INFO_PATH = "helper/libs/lib_gpu_info/build/GpuInfo.so"

LIB_CPU_INFO = None
LIB_GPU_INFO = None
try:
    LIB_CPU_INFO = ctypes.cdll.LoadLibrary(LIB_CPU_INFO_PATH)
#    LIB_GPU_INFO = ctypes.cdll.LoadLibrary(LIB_GPU_INFO_PATH)
except Exception as ex:
    raise ex


def get_cpu_cores_count():
    """ Return number of physical CPU cores """
    func_ptr = LIB_CPU_INFO.GetCpuCoresCount
    n_cpu = func_ptr()
    return n_cpu


def get_gpu_cores_count():
    """ Return number of CUDA-capable devices """
#    func_ptr = LIB_GPU_INFO.GetCudaDevicesCount
#    n_gpu = func_ptr()
    n_gpu = 1
    return n_gpu



