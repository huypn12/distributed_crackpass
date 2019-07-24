#include <iostream>

#include <string>
#include <vector>

#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "vector.hpp"



/**
 * @author: tannd
 * Convert md5 ascii string to integer form
 */
void MD5digestToHex(
        const std::string hashStr,
        std::vector<uint32_t> &hashInts
        )
{
    char* dummy = NULL;
    unsigned char bytes[16];
    for( int i = 0; i < 16; ++i )
        bytes[i] = (unsigned char) strtol(
                hashStr.substr(i * 2, 2).c_str(), &dummy, 16);
    hashInts[0] = bytes[ 0] | bytes[ 1] << 8 | bytes[ 2] << 16 | bytes[ 3] << 24;
    hashInts[1] = bytes[ 4] | bytes[ 5] << 8 | bytes[ 6] << 16 | bytes[ 7] << 24;
    hashInts[2] = bytes[ 8] | bytes[ 9] << 8 | bytes[10] << 16 | bytes[11] << 24;
    hashInts[3] = bytes[12] | bytes[13] << 8 | bytes[14] << 16 | bytes[15] << 24;
}

/**
 * Compare 2 vectors by the first dim
 */
int compLtVecByX( const vec4uint32_t vec1, const vec4uint32_t vec2 )
{
    return( vec1.x < vec2.x );
}

int readDigests( const char **hashDigests, const int nHashes )
{
    for( int hashIdx = 0; hashIdx < nHashes; hashIdx++ ) {
        const char *szHashDigest = hashDigests[hashIdx];
        std::string strDigest(szHashDigest);
        std::cout << "Read digest: " << strDigest << std::endl;
    }
}

int main(int argc, char const* argv[])
{
    int nDigests = 4;
    const char *digests[] = {
        "9dd4e461268c8034f5c8564e155c67a6",
        "9dd4e461268c8034f5cf564e155c67a6",
        "9dd4e461268ca034f5c8564e155c67a6",
        "9dd4e461268c8034f5c8564f155c67a6"
    };
    std::vector<vec4uint32_t> hashlist;
    for( int i = 0; i < 1000 )
    readDigests( digests, nDigests );
    return 0;
}
