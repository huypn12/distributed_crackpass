#ifndef _PBKDF2_1_
#define _PBKDF2_1_
/*Khai bao nguyen mau ham*/
unsigned long f_1(unsigned long B,unsigned long C,unsigned long D, int t);
void SHA1_1(int s[],int ln, unsigned long *H);
void HMAC_1(int S[],int saltLen, int P[],int length,unsigned long *H);
int PBKDF2_1(int S[],int saltLen,int stored_pvv[],int dkLen,int *P, int passLen, unsigned long *H);
void DisplayGuestPassword(char pre_terminal[20] , int pre_terminal_len,char hostArray[][80],int qualities, int startIndex, int endIndex, int *S, int saltLen, int *stored_pvv, int dkLen); 
#endif