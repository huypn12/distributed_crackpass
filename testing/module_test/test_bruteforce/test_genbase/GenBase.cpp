#include <iostream>
#include <vector>

#include <cstdint>

#include "sha1.c"


/**
 * Gen base string by its id
 * @param:
 * @return
 */
void genBaseByOffset(
        std::vector<uint8_t> inBase,
        size_t baseLen, size_t offset,
        char *charset, size_t charsetLen,
        std::vector<uint8_t> &outBase
        )
{
    uint32_t counter = offset;
    uint32_t lenIdx  = 0;
    while (lenIdx < baseLen) {
        oc = counter / charsetLen;
        a = inBase[lenIdx] + counter - oc * charsetLen;
        if( a >= charsetLen ) {
            a -= charsetLen;
            carry = 1;
        } else {
            carry = 0;
        }
        counter = oc;
        outBase[lenIdx] = a;
        lenIdx++;
    }
}


/**
 * Generate string array by index array
 * @param:
 * @return:
 */
void getBaseString(
        std::vector<uint8_t> inBase,    // indexing array
        std::vector<uint8_t> &baseStr    // real data
        )
{
    for

}


/**
 * Archiving base string to buffer
 * @param:
 * @return:
 */
void genBuf(
        std::vector<uint8_t> base,
        std::vector<uint32_t> &buf
        )
{
    size_t baseLen = base.size();
    // Assuming little-endian: NVIDA GPU & X86_64
    for( uint32_t i = 0, j = 0; i < baseLen; i += 4, j++ ) {
        buf[j] |= base[i];
        buf[j] |= base[i + 1] << 8;
        buf[j] |= base[i + 2] << 16;
        buf[j] |= base[i + 3] << 24;
        if( j == (baseLen / 4) ) {
            size_t remains = baseLen % 4;

        }
    }
}


/**
 * kernel
 * @param:
 * @return:
 */
void bruteforce(
        uint32_t* hash2crack,
        uint8_t* base, size_t worksize,
        char* charset, size_t charsetLen
        )


