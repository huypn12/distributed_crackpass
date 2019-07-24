
#include "aestab.cu"


#define ke4(k,i) \
{   k[4*(i)+4] = ss[0] ^= ls_box(ss[3],3) ^ t_use(r,c)[i]; k[4*(i)+5] = ss[1] ^= ss[0]; \
    k[4*(i)+6] = ss[2] ^= ss[1]; k[4*(i)+7] = ss[3] ^= ss[2]; \
}
#define kel4(k,i) \
{   k[4*(i)+4] = ss[0] ^= ls_box(ss[3],3) ^ t_use(r,c)[i]; k[4*(i)+5] = ss[1] ^= ss[0]; \
    k[4*(i)+6] = ss[2] ^= ss[1]; k[4*(i)+7] = ss[3] ^= ss[2]; \
}

__device__
aes_rval aes_set_encrypt_key(unsigned char in_key[], unsigned int klen, aes_ctx cx[1])
{  
    aes_32t    ss[8];

    cx->n_blk = BLOCK_SIZE;

    if(((klen & 7) || klen < 16 || klen > 32) && ((klen & 63) || klen < 128 || klen > 256))
    {
        cx->n_rnd = 0; return aes_bad;
    }

    klen >>= (klen < 128 ? 2 : 5);
    cx->n_blk = (cx->n_blk & ~3) | 1;

    cx->k_sch[0] = ss[0] = word_in(in_key     );
    cx->k_sch[1] = ss[1] = word_in(in_key +  4);
    cx->k_sch[2] = ss[2] = word_in(in_key +  8);
    cx->k_sch[3] = ss[3] = word_in(in_key + 12);

        ke4(cx->k_sch, 0); ke4(cx->k_sch, 1);
        ke4(cx->k_sch, 2); ke4(cx->k_sch, 3);
        ke4(cx->k_sch, 4); ke4(cx->k_sch, 5);
        ke4(cx->k_sch, 6); ke4(cx->k_sch, 7);
        ke4(cx->k_sch, 8); kel4(cx->k_sch, 9);
        cx->n_rnd = 10; 


    return aes_good;
}

//encrypt

#define unused  77  /* Sunset Strip */

#define si(y,x,k,c) (s(y,c) = word_in(x + 4 * c) ^ k[c])
#define so(y,x,c)   word_out(y + 4 * c, s(x,c))

#if BLOCK_SIZE == 16

#if defined(ARRAYS)
#define locals(y,x)     x[4],y[4]
#else
#define locals(y,x)     x##0,x##1,x##2,x##3,y##0,y##1,y##2,y##3
 /*
   the following defines prevent the compiler requiring the declaration
   of generated but unused variables in the fwd_var and inv_var macros
 */
#define b04 unused
#define b05 unused
#define b06 unused
#define b07 unused
#define b14 unused
#define b15 unused
#define b16 unused
#define b17 unused
#endif
#define l_copy(y, x)    s(y,0) = s(x,0); s(y,1) = s(x,1); \
                        s(y,2) = s(x,2); s(y,3) = s(x,3);
#define state_in(y,x,k) si(y,x,k,0); si(y,x,k,1); si(y,x,k,2); si(y,x,k,3)
#define state_out(y,x)  so(y,x,0); so(y,x,1); so(y,x,2); so(y,x,3)
#define round(rm,y,x,k) rm(y,x,k,0); rm(y,x,k,1); rm(y,x,k,2); rm(y,x,k,3)

#endif

#if defined(ENCRYPTION) && !defined(AES_ASM)

/* Given the column (c) of the output state variable, the following
   macros give the input state variables which are needed in its
   computation for each row (r) of the state. All the alternative
   macros give the same end values but expand into different ways
   of calculating these values.  In particular the complex macro
   used for dynamically variable block sizes is designed to expand
   to a compile time constant whenever possible but will expand to
   conditional clauses on some branches (I am grateful to Frank
   Yellin for this construction)
*/

#if defined(BLOCK_SIZE)
#if BLOCK_SIZE == 16
# define fwd_var(x,r,c) s(x,((r+c)%nc))
#else
#define fwd_var(x,r,c) s(x,(r+c+(((r>1)&&(nc>9-r))?1:0))%nc)
#endif
#else
#define fwd_var(x,r,c)\
 ( r == 0 ?    s(x,c) \
 : r == 1 ?           \
    ( c == 0 ? s(x,1) \
    : c == 1 ? s(x,2) \
    : c == 2 ? s(x,3) \
    : c == 3 ? nc == 4 ? s(x,0) : s(x,4) \
    : c == 4 ? s(x,5) \
    : c == 5 ? nc == 8 ? s(x,6) : s(x,0) \
    : c == 6 ? s(x,7) : s(x,0)) \
 : r == 2 ? \
    ( c == 0 ? nc == 8 ? s(x,3) : s(x,2) \
    : c == 1 ? nc == 8 ? s(x,4) : s(x,3) \
    : c == 2 ? nc == 4 ? s(x,0) : nc == 8 ? s(x,5) : s(x,4) \
    : c == 3 ? nc == 4 ? s(x,1) : nc == 8 ? s(x,6) : s(x,5) \
    : c == 4 ? nc == 8 ? s(x,7) : s(x,0) \
    : c == 5 ? nc == 8 ? s(x,0) : s(x,1) \
    : c == 6 ? s(x,1) : s(x,2)) \
 : \
    ( c == 0 ? nc == 8 ? s(x,4) : s(x,3) \
    : c == 1 ? nc == 4 ? s(x,0) : nc == 8 ? s(x,5) : s(x,4) \
    : c == 2 ? nc == 4 ? s(x,1) : nc == 8 ? s(x,6) : s(x,5) \
    : c == 3 ? nc == 4 ? s(x,2) : nc == 8 ? s(x,7) : s(x,0) \
    : c == 4 ? nc == 8 ? s(x,0) : s(x,1) \
    : c == 5 ? nc == 8 ? s(x,1) : s(x,2) \
    : c == 6 ? s(x,2) : s(x,3)))
#endif

#if defined(FT4_SET)
#undef  dec_fmvars
#define dec_fmvars
#define fwd_rnd(y,x,k,c)    (s(y,c) = (k)[c] ^ four_tables(x,t_use(f,n),fwd_var,rf1,c))
#elif defined(FT1_SET)
#undef  dec_fmvars
#define dec_fmvars
#define fwd_rnd(y,x,k,c)    (s(y,c) = (k)[c] ^ one_table(x,upr,t_use(f,n),fwd_var,rf1,c))
#else
#define fwd_rnd(y,x,k,c)    (s(y,c) = fwd_mcol(no_table(x,t_use(s,box),fwd_var,rf1,c)) ^ (k)[c])
#endif

#if defined(FL4_SET)
#define fwd_lrnd(y,x,k,c)   (s(y,c) = (k)[c] ^ four_tables(x,t_use(f,l),fwd_var,rf1,c))
#elif defined(FL1_SET)
#define fwd_lrnd(y,x,k,c)   (s(y,c) = (k)[c] ^ one_table(x,ups,t_use(f,l),fwd_var,rf1,c))
#else
#define fwd_lrnd(y,x,k,c)   (s(y,c) = no_table(x,t_use(s,box),fwd_var,rf1,c) ^ (k)[c])
#endif

__device__
aes_rval aes_encrypt_block(const unsigned char in_blk[], unsigned char out_blk[], aes_ctx cx[1])
{   aes_32t        locals(b0, b1);
    aes_32t  *kp = cx->k_sch;
    dec_fmvars  /* declare variables for fwd_mcol() if needed */

    if(!(cx->n_blk & 1)) return aes_bad;
				
    state_in(b0, in_blk, kp);

    kp += (cx->n_rnd - 9) * nc;

    /*lint -e{616} control flows into case/default */
    switch(cx->n_rnd)
    {
    case 10:
        round(fwd_rnd,  b1, b0, kp         );
        round(fwd_rnd,  b0, b1, kp +     nc);
        round(fwd_rnd,  b1, b0, kp + 2 * nc);
        round(fwd_rnd,  b0, b1, kp + 3 * nc);
        round(fwd_rnd,  b1, b0, kp + 4 * nc);
        round(fwd_rnd,  b0, b1, kp + 5 * nc);
        round(fwd_rnd,  b1, b0, kp + 6 * nc);
        round(fwd_rnd,  b0, b1, kp + 7 * nc);
        round(fwd_rnd,  b1, b0, kp + 8 * nc);
        round(fwd_lrnd, b0, b1, kp + 9 * nc);
    default:
        ;
    }
    state_out(out_blk, b0);
    return aes_good;
}
__device__
void encr_data(unsigned char *data, unsigned long d_len, fcrypt_ctx *cx,aes_ctx *encr_ctx)
{   unsigned long i = 0, pos = cx->encr_pos;

    while(i < d_len)
    {
        if(pos == BLOCK_SIZE)
        {   unsigned int j = 0;
            // increment encryption nonce   
            while(j < 8 && !++cx->nonce[j])
                ++j;
            // encrypt the nonce to form next xor buffer   
            aes_encrypt_block(cx->nonce, cx->encr_bfr, encr_ctx);
            pos = 0;
        }

        data[i++] ^= cx->encr_bfr[pos++];	
    }

    cx->encr_pos = pos;
}
#endif


