
#ifndef _AES_H
#define _AES_H

#if defined(__cplusplus)
extern "C"
{
#endif

/* If a table pointer is needed in the AES context, include the define  */
/* #define AES_TABLE_PTR                                                */

/*  This include is used to find 8 and 32 bit unsigned integer types    */
#include "limits.h"

#if UCHAR_MAX == 0xff                       /* an unsigned 8 bit type   */
  typedef unsigned char      aes_08t;
#else
#error Please define aes_08t as an 8-bit unsigned integer type in aes.h
#endif

#if UINT_MAX == 0xffffffff                  /* an unsigned 32 bit type  */
  typedef   unsigned int     aes_32t;
#elif ULONG_MAX == 0xffffffff
  typedef   unsigned long    aes_32t;
#else
#error Please define aes_32t as a 32-bit unsigned integer type in aes.h
#endif

/* This BLOCK_SIZE is in BYTES.  It can have the values 16, 24, 32 or   */
/* undefined for use with aescrypt.c and aeskey.c, or 16, 20, 24, 28,   */
/* 32 or undefined for use with aescrypp.c and aeskeypp.c.   When the   */
/* BLOCK_SIZE is left undefined a version that provides a dynamically   */
/* variable block size is produced but this is MUCH slower.             */

#define BLOCK_SIZE  16

/* key schedule length (in 32-bit words)                                */

#if !defined(BLOCK_SIZE)
#define KS_LENGTH   128
#else
#define KS_LENGTH   (4 * BLOCK_SIZE)
#endif

typedef unsigned int aes_fret;   /* type for function return value      */
#define aes_bad      0           /* bad function return value           */
#define aes_good     1           /* good function return value          */
#ifndef AES_DLL                  /* implement normal or DLL functions   */
#define aes_rval     aes_fret
#else
#define aes_rval     aes_fret __declspec(dllexport) _stdcall
#endif

typedef struct
{   unsigned char   nonce[BLOCK_SIZE];          /* the CTR nonce          */
    unsigned char   encr_bfr[BLOCK_SIZE];       /* encrypt buffer         */
 //   aes_ctx         encr_ctx[1];                /* encryption context     */
 //   hmac_ctx        auth_ctx[1];                /* authentication context */
    unsigned int    encr_pos;                   /* block position (enc)   */
    unsigned int    pwd_len;                    /* password length        */
    unsigned int    mode;                       /* File encryption mode   */
} fcrypt_ctx;

typedef struct                     /* the AES context for encryption    */
{   aes_32t    k_sch[KS_LENGTH];   /* the encryption key schedule       */
    aes_32t    n_rnd;              /* the number of cipher rounds       */
    aes_32t    n_blk;              /* the number of bytes in the state  */
#if defined(AES_TABLE_PTR)         /* where global variables are not    */
    void      *t_ptr;              /* available this pointer is used    */
#endif                             /* to point to the fixed tables      */
} aes_ctx;

/* The block length (blen) is input in bytes when it is in the range    */
/* 16 <= blen <= 32 or in bits when in the range 128 <= blen <= 256     */
/* Only 16 bytes (128 bits) is legal for AES but the files aescrypt.c   */
/* and aeskey.c provide support for 16, 24 and 32 byte (128, 192 and    */
/* 256 bit) blocks while aescrypp.c and aeskeypp.c provide support for  */
/* 16, 20, 24, 28 and 32 byte (128, 160, 192, 224 and 256 bit) blocks.  */
/* The value aes_good is returned if the requested block size is legal, */
/* otherwise aes_bad is returned.                                       */

#if !defined(BLOCK_SIZE)
aes_rval aes_set_block_size(unsigned int blen, aes_ctx cx[1]);
#endif

/* The key length (klen) is input in bytes when it is in the range      */
/* 16 <= klen <= 32 or in bits when in the range 128 <= klen <= 256     */
/* The files aescrypt.c and aeskey.c provide support for 16, 24 and     */
/* 32 byte (128, 192 and 256 bit) keys while aescrypp.c and aeskeypp.c  */
/* provide support for 16, 20, 24, 28 and 32 byte (128, 160, 192, 224   */
/* and 256 bit) keys.  The value aes_good is returned if the requested  */
/* key size is legal, otherwise aes_bad is returned.                    */

//aes_rval aes_set_encrypt_key(unsigned char in_key[],
           //                             unsigned int klen, aes_ctx cx[1]);
aes_rval aes_encrypt_block(const unsigned char in_blk[],
                            unsigned char out_blk[], const aes_ctx cx[1]);

aes_rval aes_set_decrypt_key(const unsigned char in_key[],
                                        unsigned int klen, aes_ctx cx[1]);
aes_rval aes_decrypt_block(const unsigned char in_blk[],
                            unsigned char out_blk[], const aes_ctx cx[1]);

#if defined(__cplusplus)
}
#endif

#endif
