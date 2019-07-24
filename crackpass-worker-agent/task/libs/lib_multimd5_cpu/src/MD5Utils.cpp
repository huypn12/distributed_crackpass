#include <cstdint>
#include <cstring>

#include <string>
#include <vector>

#include "Vector.hpp"

// MD5 utilities
void MD5DigestToHex(
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
 * Convert md5 hex digest to string
 */
void MD5HexToDigest(
        const vec4uint32_t hashvec,
        std::string &digest
        )
{
    std::string tmpStrX = std::to_string(hashvec.x);
    digest.insert(0, tmpStrX);
    std::string tmpStrY = std::to_string(hashvec.y);
    digest.insert(0, tmpStrY);
    std::string tmpStrZ = std::to_string(hashvec.z);
    digest.insert(0, tmpStrZ);
    std::string tmpStrW = std::to_string(hashvec.w);
    digest.insert(0, tmpStrW);
}


