#ifndef _ZIP_
#define _ZIP_

#define SHORT short
#define WORD unsigned int

typedef struct {
WORD 	signature;
SHORT version;
SHORT bit_flag;
SHORT compression_method;
SHORT lmf_time;
SHORT lmf_date;
WORD crc32;
WORD csize;
WORD ucsize;
SHORT fn_length;
SHORT ex_length;
} ZIP_LOCAL_FILE_HEADER;

/*Tra ve thong tin header cua file zip*/
void read_lfh(ZIP_LOCAL_FILE_HEADER* lfh, FILE* fp);

/*Ham kiem tra xem file co duoc ma hoa khong */
int get_bit_encrypted_ornot(ZIP_LOCAL_FILE_HEADER* lfh, FILE* fp);

/*Kiem tra file nhap vao co phai la file zip khong*/
int check_is_zip_file(FILE *fp);
char* get_fname(ZIP_LOCAL_FILE_HEADER* lfh, FILE* fp);

void display_salt_pvv_ac(ZIP_LOCAL_FILE_HEADER* lfh,FILE *fp,int S[],int *n2,int stored_pvv[],int *dkLen);

#endif

