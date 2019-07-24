#include <CpuInfo.hpp>


int HardwareConcurrency()
{
    return boost::thread::hardware_concurrency();
}


extern "C" {

    int DLL_PUBLIC GetCpuCoresCount() {
        return GetCpuCoresCount();
    }

}
