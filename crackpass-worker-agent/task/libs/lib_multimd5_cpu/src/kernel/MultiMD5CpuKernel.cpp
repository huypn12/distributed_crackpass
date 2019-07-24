#include <iostream>
#include <vector>

#include <cstring>
#include <cstdint>

#include "Vector.hpp"



//@huypn KERNEL
int32_t MultiMD5CpuKernel(
        const std::vector<vec4uint32_t> hashList, const size_t nHashes,
        std::vector<vec4uint32_t> &resultList,
        const unsigned char *base, const uint32_t baseLen,
        const char *charset, const uint32_t charsetLen,
        const uint32_t workSize
        )
{

    int kernelRet = -1;
    for (uint32_t baseId = 0; baseId < workSize; baseId++)
    {
        uint32_t X[16];
        memset( X, 0, 16 * sizeof(uint32_t) );
        uint32_t counter = baseId;
        uint32_t carry=0, oc=0, a=0;
        if (baseLen >= 1)
        {
            oc = counter / charsetLen;
            a = base[0] + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[0] = charset[a];
            counter = oc;
        }
        if (baseLen >= 2)
        {
            oc = counter / charsetLen;
            a = base[1] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[0] |= charset[a] << 8;
            counter = oc;
        }
        if (baseLen >= 3)
        {
            oc = counter / charsetLen;
            a = base[2] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[0] |= charset[a] << 16;
            counter = oc;
        }
        if (baseLen >= 4)
        {
            oc = counter / charsetLen;
            a = base[3] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[0] |= charset[a] << 24;
            counter = oc;
        }
        if (baseLen >= 5)
        {
            oc = counter / charsetLen;
            a = base[4] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[1] = charset[a];
            counter = oc;
        }
        if (baseLen >= 6)
        {
            oc = counter / charsetLen;
            a = base[5] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[1] |= charset[a] << 8;
            counter = oc;
        }
        if (baseLen >= 7)
        {
            oc = counter / charsetLen;
            a = base[6] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[1] |= charset[a] << 16;
            counter = oc;
        }
        if (baseLen >= 8)
        {
            oc = counter / charsetLen;
            a = base[7] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[1] |= charset[a] << 24;
            counter = oc;
        }
        if (baseLen >= 9)
        {
            oc = counter / charsetLen;
            a = base[8] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[2] = charset[a];
            counter = oc;
        }
        if (baseLen >= 10)
        {
            oc = counter / charsetLen;
            a = base[9] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[2] |= charset[a] << 8;
            counter = oc;
        }
        if (baseLen >= 11)
        {
            oc = counter / charsetLen;
            a = base[10] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[2] |= charset[a] << 16;
            counter = oc;
        }
        if (baseLen >= 12)
        {
            oc = counter / charsetLen;
            a = base[11] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[2] |= charset[a] << 24;
            counter = oc;
        }
        if (baseLen >= 13)
        {
            oc = counter / charsetLen;
            a = base[12] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[3] = charset[a];
            counter = oc;
        }
        if (baseLen >= 14)
        {
            oc = counter / charsetLen;
            a = base[13] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[3] |= charset[a] << 8;
            counter = oc;
        }
        if (baseLen >= 15)
        {
            oc = counter / charsetLen;
            a = base[14] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[3] |= charset[a] << 16;
            counter = oc;
        }
        if (baseLen >= 16)
        {
            oc = counter / charsetLen;
            a = base[15] + carry + counter - oc * charsetLen;
            if (a >= charsetLen) { a -= charsetLen; carry = 1; }
            else carry = 0;
            X[3] |= charset[a] << 24;
            counter = oc;
        }

        switch(baseLen)	// Evaluated at compile time
        {
            case 1:
                X[ 0] |= (uint32_t)(0x00008000);
                X[ 1] = X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] =
                    X[ 7] = X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 2:
                X[ 0] |= (uint32_t)(0x00800000);
                X[ 1] = X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] =
                    X[ 7] = X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 3:
                X[ 0] |= (uint32_t)(0x80000000);
                X[ 1] = X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] =
                    X[ 7] = X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 4:
                X[ 1] = (uint32_t)(0x00000080);
                X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 5:
                X[ 1] |= (uint32_t)(0x00008000);
                X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 6:
                X[ 1] |= (uint32_t)(0x00800000);
                X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 7:
                X[ 1] |= (uint32_t)(0x80000000);
                X[ 2] = X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 8:
                X[ 2] = (uint32_t)(0x00000080);
                X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 9:
                X[ 2] |= (uint32_t)(0x00008000);
                X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 10:
                X[ 2] |= (uint32_t)(0x00800000);
                X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 11:
                X[ 2] |= (uint32_t)(0x80000000);
                X[ 3] = X[ 4] = X[ 5] = X[ 6] = X[ 7] =
                    X[ 8] = X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 12:
                X[ 3] = 128;
                X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] =
                    X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 13:
                X[ 3] |= (uint32_t)(0x00008000);
                X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] =
                    X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 14:
                X[ 3] |= (uint32_t)(0x00800000);
                X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] =
                    X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 15:
                X[ 3] |= (uint32_t)(0x80000000);
                X[ 4] = X[ 5] = X[ 6] = X[ 7] = X[ 8] =
                    X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
            case 16:
                X[ 4] = (uint32_t)(0x00000080);
                X[ 5] = X[ 6] = X[ 7] = X[ 8] =
                    X[ 9] = X[10] = X[11] = X[12] = X[13] = 0;
                break;
        }

        X[14] = baseLen << 3;
        X[15] = 0;

        uint32_t A, B, C, D;

#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))

#define P(a,b,c,d,k,s,t)                                \
        {														\
            a += F(b,c,d) + X[k] + t; a = S(a,s) + b;			\
        }														\

        A = 0x67452301;
        B = 0xefcdab89;
        C = 0x98badcfe;
        D = 0x10325476;

#define F(x,y,z) (z ^ (x & (y ^ z)))

        P( A, B, C, D,  0,  7, 0xD76AA478 );
        P( D, A, B, C,  1, 12, 0xE8C7B756 );
        P( C, D, A, B,  2, 17, 0x242070DB );
        P( B, C, D, A,  3, 22, 0xC1BDCEEE );
        P( A, B, C, D,  4,  7, 0xF57C0FAF );
        P( D, A, B, C,  5, 12, 0x4787C62A );
        P( C, D, A, B,  6, 17, 0xA8304613 );
        P( B, C, D, A,  7, 22, 0xFD469501 );
        P( A, B, C, D,  8,  7, 0x698098D8 );
        P( D, A, B, C,  9, 12, 0x8B44F7AF );
        P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
        P( B, C, D, A, 11, 22, 0x895CD7BE );
        P( A, B, C, D, 12,  7, 0x6B901122 );
        P( D, A, B, C, 13, 12, 0xFD987193 );
        P( C, D, A, B, 14, 17, 0xA679438E );
        P( B, C, D, A, 15, 22, 0x49B40821 );

#undef F

#define F(x,y,z) (y ^ (z & (x ^ y)))

        P( A, B, C, D,  1,  5, 0xF61E2562 );
        P( D, A, B, C,  6,  9, 0xC040B340 );
        P( C, D, A, B, 11, 14, 0x265E5A51 );
        P( B, C, D, A,  0, 20, 0xE9B6C7AA );
        P( A, B, C, D,  5,  5, 0xD62F105D );
        P( D, A, B, C, 10,  9, 0x02441453 );
        P( C, D, A, B, 15, 14, 0xD8A1E681 );
        P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
        P( A, B, C, D,  9,  5, 0x21E1CDE6 );
        P( D, A, B, C, 14,  9, 0xC33707D6 );
        P( C, D, A, B,  3, 14, 0xF4D50D87 );
        P( B, C, D, A,  8, 20, 0x455A14ED );
        P( A, B, C, D, 13,  5, 0xA9E3E905 );
        P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
        P( C, D, A, B,  7, 14, 0x676F02D9 );
        P( B, C, D, A, 12, 20, 0x8D2A4C8A );

#undef F

#define F(x,y,z) (x ^ y ^ z)

        P( A, B, C, D,  5,  4, 0xFFFA3942 );
        P( D, A, B, C,  8, 11, 0x8771F681 );
        P( C, D, A, B, 11, 16, 0x6D9D6122 );
        P( B, C, D, A, 14, 23, 0xFDE5380C );
        P( A, B, C, D,  1,  4, 0xA4BEEA44 );
        P( D, A, B, C,  4, 11, 0x4BDECFA9 );
        P( C, D, A, B,  7, 16, 0xF6BB4B60 );
        P( B, C, D, A, 10, 23, 0xBEBFBC70 );
        P( A, B, C, D, 13,  4, 0x289B7EC6 );
        P( D, A, B, C,  0, 11, 0xEAA127FA );
        P( C, D, A, B,  3, 16, 0xD4EF3085 );
        P( B, C, D, A,  6, 23, 0x04881D05 );
        P( A, B, C, D,  9,  4, 0xD9D4D039 );
        P( D, A, B, C, 12, 11, 0xE6DB99E5 );
        P( C, D, A, B, 15, 16, 0x1FA27CF8 );
        P( B, C, D, A,  2, 23, 0xC4AC5665 );

#undef F

#define F(x,y,z) (y ^ (x | ~z))

        P( A, B, C, D,  0,  6, 0xF4292244 );
        P( D, A, B, C,  7, 10, 0x432AFF97 );
        P( C, D, A, B, 14, 15, 0xAB9423A7 );
        P( B, C, D, A,  5, 21, 0xFC93A039 );
        P( A, B, C, D, 12,  6, 0x655B59C3 );
        P( D, A, B, C,  3, 10, 0x8F0CCC92 );
        P( C, D, A, B, 10, 15, 0xFFEFF47D );
        P( B, C, D, A,  1, 21, 0x85845DD1 );
        P( A, B, C, D,  8,  6, 0x6FA87E4F );
        P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
        P( C, D, A, B,  6, 15, 0xA3014314 );
        P( B, C, D, A, 13, 21, 0x4E0811A1 );
        P( A, B, C, D,  4,  6, 0xF7537E82 );
        P( D, A, B, C, 11, 10, 0xBD3AF235 );
        P( C, D, A, B,  2, 15, 0x2AD7D2BB );
        P( B, C, D, A,  9, 21, 0xEB86D391 );

#undef F

        A += 0x67452301;
        B += 0xefcdab89;
        C += 0x98badcfe;
        D += 0x10325476;

        // Binary search for delivered hash string
        // Uses A as the key
        // The probability of two hashes containing the same heading bits
        // is relatively small -> ignore it
        /*
           for( unsigned int i = 0; i < nHashes; i++ )
           {
           if( A == hashList[i].x &&
           B == hashList[i].y &&
           C == hashList[i].z &&
           D == hashList[i].w)
           { // key is found
           vec4uint32_t result( baseId, i, 1, A );
           resultList.push_back(result);
           kernelRet = 0;
           break;
           }
           }
           */
        int left = 0;
        int right = nHashes - 1;
        int mid = 0;
        while( left <= right )
        {
            mid = (int) (left + right) / 2;
            if( (A == hashList[mid].x) &&
                    (B == hashList[mid].y) &&
                    (C == hashList[mid].z) &&
                    (D == hashList[mid].w))
            {
                /*-->DEBUG
                std::cout << "found id: " << mid << " baseid: " << baseId << std::endl;
                std::cout << "base: ";
                for( unsigned int i = 0; i < baseLen; i++ )
                {
                    std::cout << charset[base[i]];
                }
                std::cout << std::endl;
                std::cout << "B=" << A << " y=" << hashList[mid].x << std::endl;
                */
                vec4uint32_t result( baseId, mid, 1, A );
                resultList.push_back(result);
                kernelRet = 0;
                break;
            }
            else if( A > hashList[mid].x )
            {
                left = mid + 1;
            }
            else if( A < hashList[mid].x )
            {
                right = mid - 1;
            }
        }
    }
    return kernelRet;
}

