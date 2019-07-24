#include <stdio.h>
#include <stdlib.h>
#include "zip.h"

void read_lfh(ZIP_LOCAL_FILE_HEADER* lfh, FILE* fp)
{
  if(fp != (FILE*) NULL)
  {
	fseek(fp, 0, SEEK_SET);
	fread(&lfh->signature, sizeof(WORD), 1, fp);
	fread(&lfh->version, sizeof(SHORT), 1, fp);
	fread(&lfh->bit_flag, sizeof(SHORT), 1, fp);
	fread(&lfh->compression_method, sizeof(SHORT), 1, fp);
	fread(&lfh->lmf_time, sizeof(SHORT), 1, fp);
	fread(&lfh->lmf_date, sizeof(SHORT), 1, fp);
	fread(&lfh->crc32, sizeof(WORD), 1, fp);
	fread(&lfh->csize, sizeof(WORD), 1, fp);
	fread(&lfh->ucsize, sizeof(WORD), 1, fp);
	fread(&lfh->fn_length, sizeof(SHORT), 1, fp);
	fread(&lfh->ex_length, sizeof(SHORT), 1, fp);
 }
}



int get_bit_encrypted_ornot(ZIP_LOCAL_FILE_HEADER* lfh, FILE* fp)
{
     long size;
     int bit0; //Dung de biet file co duoc ma hoa hay khong
     if(!fp)
        return -1;
     //Lay ve mot SHORT chua thong tin trang thai co
     size = lfh->bit_flag;
     bit0 = (int)(size & 0x01);
     return bit0;
}

int check_is_zip_file(FILE *fp)
{
   unsigned char list[2];
   if(!fp)
      return -1;
   //Dua bien file ve dau dong
   fseek(fp,0,SEEK_SET);
   fread(list,sizeof(unsigned char), 2, fp);
//   printf("\n%x %x",list[0], list[1]);
   if(list[0] == 'P' && list[1] == 'K')
      return 1;
   else
      return 0;
}
char* get_fname(ZIP_LOCAL_FILE_HEADER* lfh, FILE* fp)
{
    int fn_len = lfh->fn_length; // lay do dai ten file
    // khoi tao chuoi: do dai ten file + 1
    char* fname = (char*) malloc( sizeof(char) * fn_len + sizeof(char));
    fseek(fp, 30, SEEK_SET); // dat con tro voi OFFSET: 30
    fread(fname, sizeof(char), fn_len, fp); // lay ten file
    *(fname + fn_len) = 0; // ket thuc chuoi \0
    return fname;
}

void display_salt_pvv_ac(ZIP_LOCAL_FILE_HEADER* lfh,FILE *fp,int S[],int *n2,int stored_pvv[],int *dkLen)
{
   /*Dua con tro den dau truong extra-field-data
   Doc 11 byte cua truong nay
   Xac nhan su dung ma hoa manh AES*/
   long index;
   unsigned char list[2];
   unsigned char temp;
   unsigned char size;
   int i;

   index = 30 + lfh->fn_length;
   fseek(fp,index, SEEK_SET);
   fread(list, sizeof(unsigned char), 2,fp);
   if(list[0] == 0x01 && list[1] == 0x99)
   {
     //Gia tri o tren cho biet dau hieu ma hoa manh
     //Kiem tra ma hoa 128,192 hay 256 bit de biet la salt chiem may byte.
     index = 30 + lfh->fn_length + 8;
     fseek(fp,index, SEEK_SET);
     fread(&size, sizeof(unsigned char), 1,fp);
     if(size == 0x01)
     {
         *dkLen = 16;
         *n2 = 8;
     }
     else if(size == 0x02)
     {
         *dkLen = 24;
         *n2 = 12;
     }
     else if(size == 0x03)
     {
         *dkLen = 32;
         *n2 = 16;
     }
     //Hien thi thong tin ve salt;
     index = 30 + lfh->fn_length + lfh->ex_length;
     fseek(fp,index, SEEK_SET);
     printf("\nSalt value:\n");
     for(i=0; i<*n2; i++)
     {
     		fread(&temp, sizeof(unsigned char), 1,fp);
         printf("%3x", temp);
         S[i]=temp;
     }
     //Hien thi 2 byte password verification value
     printf("\nPassword Verification Value:\n");
     fread(list, sizeof(unsigned char), 2,fp);
     printf("%3x %3x",list[0],list[1]);
     //Luu tru cho muc dich kiem tra sau nay
     stored_pvv[0] = list[0];
     stored_pvv[1] = list[1];
     //Dua con tro tro den vi tri cua authentication value
     index = 30 + lfh->fn_length + lfh->ex_length + lfh->csize -10;
     fseek(fp,index, SEEK_SET);
     //doc ra lay 10 byte.
     printf("\nAuthentication Value:\n");
     for(i=0; i<10; i++)
     {
       fread(&temp, sizeof(unsigned char), 1,fp);
       printf("%3x", temp);
     }
   }
}
