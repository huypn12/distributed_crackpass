#ifndef __MULTI_MD5_CPU_KERNEL_H__
#define __MULTI_MD5_CPU_KERNEL_H__


#include <cstdint>
#include <vector>

#include "Vector.hpp"


/**@author: tannd
 * @huypn: This code is not done with good practices, honestly
 * @param: hash_to_crack:   integers representation of hash string
 * @param: base:            base key, can be up to 16 characters ascii
 * @param: baseLen:         base key lenght
 * @param: charset:         character set
 * @param: chaset_length:   size of character set
 * @param: worksize:        number of keys to be generated and tested per each invocation
 */
int32_t MultiMD5CpuKernel(
        const std::vector<vec4uint32_t> hashList, const size_t nHashes,
        std::vector<vec4uint32_t> &kernelRes,
        const unsigned char *base, const uint32_t baseLength,
        const char *charset, const uint32_t charsetLength,
        const uint32_t workSize
        );


#endif
