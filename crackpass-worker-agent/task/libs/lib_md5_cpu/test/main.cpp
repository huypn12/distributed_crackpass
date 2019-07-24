#include <iostream>
#include <unistd.h>
#include <cstdio>

#include "../include/MD5Cpu.hpp"

int main(int argc, char const* argv[])
{
    MD5Cpu *md5_cpu = new MD5Cpu(
            32, // n threads
            //"e80b5017098950fc58aad83c8c14978e",
            //"4124bc0a9335c27f086f24ba207a4912",
            //"47bce5c74f589f4867dbd57e9ca9f808",
            //"0bee89b07a248e27c83fc3d5951213c1",
            //"82e495da79fc6b1f98b0b3160405eccf",
            //"900150983cd24fb0d6963f7d28e17f72",
            //"207462d2fdaa209532ee2e5541860dd1",
            //"d6b9b83ff5281e8150ab1886527823a6",
            //"1b1cc7f086b3f074da452bc3129981eb",
            //"0775d0bfbec108de4a8271e34f9547cb",
            //"931168d1f9d2eb314552619d5aed743a",
            //"5f36d504f4082ab73f72cfeaf5d79548",
            //"46399f97b09b5ca519a524a3dfc68419",
            //"d955989d558e6b230921b4f2e1d35d62",
            //"c4d0ced267eb25080ed058f4a13ccf1a",
            //"ee06c3563dff07c2bd8d474f1142279e",
            //"2294d8d1b4922745078ba000d95c4852",
            //"6a2033c71c6ee931cac35347c605a306",
            //"0b4e7a0e5fe84ad35fb5f95b9ceeac79",
            //"efbdfa34b283bb4b6931ff0f9ab1a164",
            //"f5e38c8717e69618008ca2cf9d8b2166",
            //"e10e9ea7fba19a7aab4ba6c69838543c",
            //"207462d2fdaa209532ee2e5541860dd1",
            //"c137d413187fccdc2a0e9954df5fbad0",
            //"cb1858c331c50a0f7abdadd5e1e3d129",
            //"97bc1c1d7d34c2a618079a4cdcc45cbb",
            //:"46399f97b09b5ca519a524a3dfc68419",
            //"36d04a9d74392c727b1a9bf97a7bcbac",
            "275dcebb6981026f2fb3bc2d34921bb6",
            "abcdefghijklmnopqrstuvwxyz" // charset
            );

    md5_cpu->Start();

    md5_cpu->PushRequest(
            "aaaaa",  //basestr
            5,      //base len
            26*26*26*26*26    //offset
            );
    while( md5_cpu->GetState() != TASK_STATE_FOUND_RESULT ) {
    }

    char result[1024];
    md5_cpu->PopResult(result);
    printf("result=%s\n", result);
    return 0;
}
