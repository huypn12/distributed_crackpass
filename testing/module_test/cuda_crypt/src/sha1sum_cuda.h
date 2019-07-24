#ifndef __SHA1SUM_CUDA__
#define __SHA1SUM_CUDA__

#define ROTATE_LEFT(x, n) \
    (((x) << (n)) | ((x) >> (32 - n))

#define ROTATE_RIGHT(x, n) \
    (((x) << (n)) | ((x) >> (32 - n))

#endif
