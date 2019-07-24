#include "aescrypt.cu"
#include "inflate.cu"
#define dotdoc 1
#define dotpdf 2
#define dottxt 3
#define dotppt 4
#define dotxls 5

#define ERROR_USAGE              1
#define ERROR_PASSWORD_LENGTH    2
#define ERROR_OUT_OF_MEMORY      3
#define ERROR_INPUT_FILE         4
#define ERROR_OUTPUT_FILE        5
#define ERROR_BAD_PASSWORD       6
#define ERROR_BAD_AUTHENTICATION 7
#define CHUNK 1024
#define PASSWORD_VERIFIER

#define MAX_KEY_LENGTH        32
#define MAX_PWD_LENGTH       128
#define MAX_SALT_LENGTH       16
#define KEYING_ITERATIONS   1000

#ifdef  PASSWORD_VERIFIER
#define PWD_VER_LENGTH         2
#else
#define PWD_VER_LENGTH         0
#endif

#define GOOD_RETURN            0
#define PASSWORD_TOO_LONG   -100
#define BAD_MODE            -101

/*
    Field lengths (in bytes) versus File Encryption Mode (0 < mode < 4)

    Mode Key Salt  MAC Overhead
       1  16    8   10       18	//day la che do dang dung
       2  24   12   10       22
       3  32   16   10       26

   The following macros assume that the mode value is correct.
*/

#define KEY_LENGTH(mode)        (8 * (mode & 3) + 8)
#define SALT_LENGTH(mode)       (4 * (mode & 3) + 4)
#define MAC_LENGTH(mode)        (10)
#define threadsperblock 128
typedef struct {
    //Device id
    int device;
    //Host-side input data
  
    int    *d_salt;
    int    *d_pvv;
    int *d_ret;		
    unsigned char *d_in; //chua du lieu vao
    unsigned char *d_out;//chua du lieu ra tren device
	int wordCount; 
	int startIndex;
	int quantities;
    	char *devPass;	
    unsigned char *Key;
    unsigned char *d_Key;		
	//Mang chua cau truc mat khau, dung de ghep tu (hay con goi la can tu)
	char *d_pre_terminal;
	//Chua mang gom cac tu dien.
	char *deviceArrPtr;
    fcrypt_ctx *d_zcx;
    aes_ctx *d_acx;
    z_stream *d_strm;	
    struct inflate_state FAR *d_state;
	
} TGPUplan;


