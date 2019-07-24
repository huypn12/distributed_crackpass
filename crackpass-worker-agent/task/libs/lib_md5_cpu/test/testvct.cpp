#include <cstdlib>
#include <ctime>

#include <iostream>
#include <vector>
#include <algorithm>

#include "../Vector.h"


std::vector< Vct4Uint32_t > hashes;

void Init(int nHashes)
{
    std::srand( std::time(NULL) );
    for( int i = 0; i < nHashes; ++i ) {
        uint32_t base = std::rand() % 0x12345;
        Vct4Uint32_t tmpVct(base+i,base+i+1,base+i+2,base+i+3);
        hashes.push_back( tmpVct );
    }
}

int main(int argc, char const* argv[])
{
    int32_t nHashes = 100000;
    Init( nHashes );

    // Iteration test
    for( int i = 0; i < nHashes ; ++i ) {
        std::cout << std::hex << hashes[i].x << ";" <<
            std::hex << hashes[i].y << ";" <<
            std::hex << hashes[i].z << ";" <<
            std::hex << hashes[i].w << std::endl;
    }

    // Operator= test
    Vct4Uint32_t vct1(0,0,0,0);
    Vct4Uint32_t vct2;
    vct2 = vct1;
    std::cout << std::hex << vct2.x << ";" <<
        std::hex << vct2.y << ";" <<
        std::hex << vct2.z << ";" <<
        std::hex << vct2.w << std::endl;

    // Operator== test
    Vct4Uint32_t vct3(0,0,0,1);
    std::cout << (vct3 == vct1) << std::endl ;

    // Operator > test
    std::cout << (vct3 > vct1) << std::endl;
    return 0;

    // try sorting
    std::sort( hashes.begin(), hashes.end() );
    // check sorted property
    for( int i = 0;  )
}
