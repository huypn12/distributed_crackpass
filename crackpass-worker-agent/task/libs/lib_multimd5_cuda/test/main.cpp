#include <cstdio>
#include <unistd.h>

#include "../include/MD5Cuda.hpp"


int main(void)
{
    int n_gpu = 1;
    int gpu_list[] = {1};
    MD5Cuda *md5instance = new MD5Cuda(
            n_gpu, gpu_list,
            //"c64557998560d21c314de1fac9c2c309",
            "0b4e7a0e5fe84ad35fb5f95b9ceeac79",
            //"efbdfa34b283bb4b6931ff0f9ab1a164",
            //"6811fb3d2676760db9f498f26c53e0af",
            //"b1ffb6b5d22cd9f210fbc8b7fdaf0e19",
            //"700cc963c0160c5667e8a5666daf7032",
            //"acd572cb62a8bc6e4b518b7f7385abab",
            //"4e6e8dc9de76ee36434ee738de3adaae",
            //"d4b9cb49c1350bf345592f819ffc1ec1",
            //"d0e4b6ab75fc20fb73f939d96ae87864",
            //"69298b9997d985414465b30713b5fd1c",
            //"3c6f183c61e340b8c34d970a205a243d", // hash2crack
            "abcdefghijklmnopqrstuvwxyz" // charset
            );

    md5instance->Start();
    md5instance->PushRequest("aaaaaa", 6, 400000000);
    for(;;){
        int state_flag = md5instance->GetState();
        if( state_flag == TASK_STATE_FOUND_RESULT ) {
            std::cout << "Found result!" << std::endl;
            break;
        }
    }
    md5instance->Stop();
    return 0;
}
