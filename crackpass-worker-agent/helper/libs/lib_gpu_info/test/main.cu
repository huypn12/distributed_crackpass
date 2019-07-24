#include <iostream>
#include <GpuInfo.cuh>

int main() {
	std::cout << getCudaDeviceCount() << std::endl;
	return 0;
}

