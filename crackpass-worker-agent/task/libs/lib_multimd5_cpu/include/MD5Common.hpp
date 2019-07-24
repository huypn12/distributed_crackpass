#ifndef __MD5_COMMON__
#define __MD5_COMMON__


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
 * @param:      hashvec: vec4uint32_t
 * @param:      digest: std::string
 * @return:     void
 */
void MD5HexToDigest(
        const vec4uint32_t hashvec,
        std::string &digest
        );



#endif

