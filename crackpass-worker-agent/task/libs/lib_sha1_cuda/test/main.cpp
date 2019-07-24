#include <cstdio>
#include <unistd.h>

#include "../include/SHA1Cuda.hpp"


int main(void)
{
    int n_gpu = 1;
    int gpu_list[] = {1};
    SHA1Cuda *sha1instance = new SHA1Cuda(
            n_gpu, gpu_list,
            "6c0c231fa004af781d28ec5be037bc01f89cb008",
            "abcdefghijklmnopqrstuvwxyz" // charset
            );

    sha1instance->Start();
    sha1instance->PushRequest("aaaa", 4, 400000000);
    for(;;){
        int state_flag = sha1instance->GetState();
        if( state_flag == TASK_STATE_FOUND_RESULT ) {
            std::cout << "Found result!" << std::endl;
            break;
        }
    }
    sha1instance->Stop();
    return 0;
}
