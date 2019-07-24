#include <iostream>
#include <unistd.h>

#include "../include/SHA1Cpu.hpp"

int main(int argc, char const* argv[])
{
    SHA1Cpu *sha1_cpu = new SHA1Cpu(
            32, // n threads
            //"6c0c231fa004af781d28ec5be037bc01f89cb008",   // sapa
            //"d6971690dbf480ea773110da946b505db3067042",   // hatay
            //"3abfcc786884db91ecd14509dc62fad0da2e3d76",   // hanam
            //"20a10d5fc2e5e219228c2eac03ccc4bb92836bb1",   // jetzt
            //"2aec56b6f154c2ff8f2c63d00bdb09d675943679",   // hanoi
            //"7535160fba8a3f2a0b1c320aa51567309ce2aae8",   // nicht
            //"d4b6fec84707c6c3a22122a83dbbd42c8e908f5c",   // cuccu
            //"9df8da34ccaadaf833f4d786b97ec97fc1506ebb",   // hadong
            //"4b4f65f1ae6a72e8a48c605527bad951f791fa56",   // sontay
            "1d14344b9f5147b46f0c397dd21cc94ce14af6de",     // taubay
            "abcdefghijklmnopqrstuvwxyz" // charset
            );

    sha1_cpu->Start();

    sha1_cpu->PushRequest(
            "aaaaaa",           //basestr
            6,                  //base len
            26*26*26*26*26*26   //offset
            );
    while( sha1_cpu->GetTaskState() != TASK_STATE_FOUND_RESULT ) {
    }
    char result[1024];
    sha1_cpu->PopResult(result);
    printf("Result found: %s\n", result);
    return 0;
}
