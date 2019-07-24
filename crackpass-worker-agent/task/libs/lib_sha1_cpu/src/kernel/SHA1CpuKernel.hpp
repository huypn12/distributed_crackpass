#ifndef __SHA1_CPU_KERNEL_H__
#define __SHA1_CPU_KERNEL_H__

#include <cstdint>

/**@author: tannd
 * @huypn: This code is not done with good practices, honestly
 * @param: hash_to_crack:   integers representation of hash string
 * @param: base:            base key, can be up to 16 characters ascii
 * @param: baseLen:         base key lenght
 * @param: charset:         character set
 * @param: chaset_length:   size of character set
 * @param: worksize:        number of keys to be generated and tested per each invocation
 */
int32_t SHA1CpuKernel(
        uint8_t *hashToCrack,
        unsigned char *base, uint32_t baseLength,
        const char *charset, uint32_t charsetLength,
        uint32_t workSize, uint32_t &resultIdx
        );

#endif
