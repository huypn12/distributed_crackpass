#ifndef __MD5_COMMON__
#define __MD5_COMMON__

typedef enum eMD5JobState {
    eMD5Init,
    eMD5Processing,
    eMD5OutOfData,
    eMD5WaitingData,
    eMD5Completed
} MD5JobState_t;

typedef struct Vct4UInt32 Vct4Uint32_t;
#endif

