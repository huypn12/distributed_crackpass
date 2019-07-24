#include <vector>
#include <cstdint>

#include "Vector.hpp"

/**
 * convert hash ascii string to vector repr.
 * @param: std::string      hash string
 * @param: std::vector<uint32_t>& hash vector
 * @return: void
 */
void MD5DigestToHex(
        const std::string hashStr,
        std::vector<uint32_t> &hashInts
        );


/**
 * Convert md5 hex digest to string
 */
void MD5HexToDigest(
        const vec4uint32_t hashvec,
        std::string &digest
        );

