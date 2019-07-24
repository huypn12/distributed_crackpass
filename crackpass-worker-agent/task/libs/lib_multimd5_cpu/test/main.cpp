#include <iostream>
#include <unistd.h>

#include "MultiMD5Cpu.hpp"

int main(int argc, char const* argv[])
{
    const char *hashList[4] = {
        "d955989d558e6b230921b4f2e1d35d62", // "hadong"
        "efbdfa34b283bb4b6931ff0f9ab1a164", // "sontay"
        "0b4e7a0e5fe84ad35fb5f95b9ceeac79", // "aaaaaa"
        "f5e38c8717e69618008ca2cf9d8b2166"  // "anninh"
    };
    MultiMD5Cpu *md5_cpu = new MultiMD5Cpu(
            16, // n threads
            hashList, 4,
            "abcdefghijklmnopqrstuvwxyz" // charset
            );

    md5_cpu->Start();
    md5_cpu->PushRequest(
                "aaaaaa",   //basestr
                6,          //base len
                320000000   //offset
                );
    while( md5_cpu->GetState() != TASK_STATE_FINISHED )
    {
    }

    return 0;
}
