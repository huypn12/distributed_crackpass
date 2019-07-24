
#include <stdlib.h>

#include "sha1.h"


void HMAC_PBKDF2_SHA1(
        unsigned char *pass, size_t pass_len,
        unsigned char *salt, size_t salt_len,
        int iterCount, int dkLen
        )
{
    unsigned char tmp[SHA1_DIGEST_LEN];
    unsigned char tmp2[SHA1_DIGEST_LEN];
    int i,j;
    unsigned char count_buf[4];
    const uint8_t *addr[2];
    size_t len[2];

    addr[0] = (uint8_t *) salt;
    len[0] = salt_len;
    addr[1] = count_buf;
    len[1] = 4;


}
