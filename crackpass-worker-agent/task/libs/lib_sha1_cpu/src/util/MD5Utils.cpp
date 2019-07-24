#include <cstdint>

// Device function prototypes
void GenInbaseByIdx(
        unsigned char *inBase, uint32_t inBaseLen,
        unsigned char *charset,uint32_t charsetLen,
        uint32_t inBaseIdx
        );

void MD5Calc(
        unsigned char *inBase, uint32_t inBaseLen,
        uint8_t md5buf[ 64 ]
        );


//TODO: @huypn-> Combine-attack & Rule-attack
void CombineWords(  );
void MixByRules(  );

// Kernel implementation, for reference
void MD5Kernel(
        const unsigned char *inBase, const inBaseLen,
        const uint32_t dataSize, const uint32_t chunkSize,
        const unsigned char *charset,
        const uint32_t charsetLen
        );

void GenInbase(
        const uint32_t *inBase,
        const uint32_t inBaseLen,
        const uint32_t charsetLen,
        const uint32_t inBaseIdx,
        uint32_t **outBase
        )
{
    int32_t oc = 0, a = 0, carry = 0;
    for( int curLen = 1; curLen <= inBaseLen; ++curLen ) {
        oc = counter / charsetLen;
        a = inBase[i] + counter - oc * charsetLen;
        if (a >= charsetLen) {
            a -= charsetLen;
            carry = 1;
        } else {
            carry = 0;
        }
        counter = oc;
        // appending a
        outBase[curLen] = a;
        ++curLen;
    }
}

