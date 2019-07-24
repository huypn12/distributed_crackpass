#include <cstring>

#include "SHA1CpuKernel.hpp"
#include "sha1.c"

#include <iostream>
#ifdef _DEBUG
#include <cstdio>
#endif


//@huypn KERNEL
int32_t SHA1CpuKernel(
        unsigned char *hashToCrack,
        unsigned char *base, uint32_t baseLen,
        const char *charset, uint32_t charsetLen,
        uint32_t workSize, uint32_t &resultIdx
        )
{
    for (uint32_t baseId = 0; baseId < workSize; baseId++)
    {
        // Generate buffer
        unsigned char buffer[32];
        memset(buffer, 0, 32 * sizeof(int));
        uint32_t carry = 0, oc = 0, a = 0;
        uint32_t counter = baseId;
        for (uint32_t b_len = 0; b_len < baseLen; b_len++) {
            oc = counter / charsetLen;
            a = base[b_len] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) {
                a -= charsetLen;
                carry = 1;
            } else {
                carry = 0;
            }
            buffer[b_len] = charset[a];
            counter = oc;
        }
        // Calculate SHA1
        SHA1_CTX ctx;
        SHA1Init(&ctx);
        SHA1Update(&ctx, buffer, baseLen);
        unsigned char hash[20];
        SHA1Final(hash, &ctx);
        // Compare to original hash
        unsigned char cmp = 1;
        for (int i = 0; i < 20; i++) {
            if (hashToCrack[i] != hash[i]) {
                cmp = 0;
                break;
            }
        }
        if (cmp == 1) {
            resultIdx = baseId;
            return 0;
        }
    }
    return -1;
}

