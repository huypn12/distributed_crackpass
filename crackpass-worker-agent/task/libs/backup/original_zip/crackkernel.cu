#include <string.h>
#include "devicedefine.h"

#define Rol(word,bits) (((word) << (bits)) | ((word) >> (32-(bits))))
#define Ror(word,bits) (((word) >> (bits)) | ((word) << (32-(bits))))

__device__ inline
unsigned long f(unsigned long B,unsigned long C,unsigned long D, int t)
{
	if (t < 20)
		{
			return ((B & C)|((~B) & D));
		}
	else
	if ((t > 19) && (t < 40))
		{
			return (B ^ C ^ D);
		}
	else
	if ((t > 39) && (t < 60))
		{
			return ((B & C)|(B & D)|(C & D));
		}
	else
	if (t > 59)
		{
			return (B ^ C ^ D);
		}
	else return 0;
}

__device__ inline 
void SHA1(int s[],int ln, unsigned long *H)
{
	unsigned long K[80];
	unsigned long A,B,C,D,E,TEMP;
	int r,k;
        int ln_static = ln;
	H[0]=0x67452301;
	H[1]=0xefcdab89;
	H[2]=0x98badcfe;
	H[3]=0x10325476;
	H[4]=0xc3d2e1f0;
	r = (ln+1)/64;
   //r la so khoi chia message(salt) thanh cac khoi co do dai 512bit hay 64byte
   //kiem tra neu phan du chia cho 64 lon hon 54 thi tang them mot khoi nua, neu
   //khong thi thoi
	if (((ln+1)-r*64) > 56)	r=r+1;

	// initialize Constants
	for(int t=0; t<80; t++)
		{
			if (t<20)
				{
					K[t] = 0x5a827999;
				}

			if ((t>19)&&(t<40))
				{
					K[t] = 0x6ED9EBA1;
				}
			if ((t>39)&&(t<60))
				{
					K[t] = 0x8F1BBCDC;
				}
			if (t>59)
				{
					K[t] = 0xca62c1d6;
				}
		}
   //Lap lai phep xu ly cho moi khoi duoc chia
	for(int l=0; l <= r; l++)
	{
		unsigned long W[80];
		for (int i=0; i<80; i++) W[i]=0;

		//Initialize Text
		for (int i=0; i<16; i++)
			{
			for(int j=0; j<4; j++)
				{
					if (4*i+j <= ln)
					{
						k = s[64*l+4*i+j];
					}
					else
					{
						k = 0;
					}
				
					if (k<0)
					{
						k = k +256;
					}

					if (4*i+j == ln)
					{
						k = 0x80;
					}
					int temp=1;
					for (int z=0;z<3-j;z++) temp*=256;
					W[i]+=k*temp;
				}
			}
		if ((W[14]==0)&&(W[15]==0))
		{
			W[15]=8*ln_static;
		}

	// Hash Cycle

		for (int t = 16; t <80; t++)
			{
				W[t] = Rol(W[t-3]^W[t-8]^W[t-14]^W[t-16],1);
			}

		A = H[0];
		B = H[1];
		C = H[2];
		D = H[3];
		E = H[4];

		for(int t = 0; t < 80; t++)
		{	
			TEMP = Rol(A,5) + f(B,C,D,t) + E + W[t] + K[t];
			E = D;
			D = C;
			C = Rol(B,30);
			B = A;
			A = TEMP;
		}

		H[0] = H[0] + A;
		H[1] = H[1] + B;
		H[2] = H[2] + C;
		H[3] = H[3] + D;
		H[4] = H[4] + E;

		ln -=  64;
	}

}

__device__ inline
void HMAC(int S[],int saltLen, int P[],int passLen,unsigned long *H)
{
int s[160];
unsigned long Key[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
unsigned long X[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
unsigned long Y[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int k,i1,j1;

//Process string key into sub-key
//Hash key in case it is less than 64 bytes
for (int i=0;i<160;i++) s[i]=0;
if (passLen > 64)
{
	SHA1(P,passLen,H);
	Key[0] = H[0];
	Key[1] = H[1];
	Key[2] = H[2];
	Key[3] = H[3];
        Key[4] = H[4];
}
else
{
	for(int i=0; i<16; i++)
	{
		Key[i]=s[i]=0;
		for(int j=0; j<4; j++)
		{
			if (4*i+j < passLen)
			{
				k = P[4*i+j];
			}
			else
			{
				k = 0;
			}
			if (k<0)
			{
				k = k + 256;
			}
			int temp=1;
			for (int z=0;z<3-j;z++) temp*=256;
			Key[i]+=k*temp;
			
		}
	
	}
}

for(int i=0; i<16; i++)
{
	X[i] = Key[i]^0x36363636;
	Y[i] = Key[i]^0x5c5c5c5c;
	
}

//Turn X-Array into a String
unsigned long X1[16];
for (int i=0; i<16;i++) X1[i]=X[i];
i1=0;
for(int i=0; i<16; i++)
{
	for(int j=0; j<4; j++)
   {
		X[i] = X1[i];
		s[i1]= (X[i] >> 8*(3-j)) % 256;
    		i1++;
	}
}

for(j1=i1; j1<(saltLen+i1); j1++)
{
	s[j1] = S[j1-i1];
}

SHA1(s,saltLen+i1,H);

for(j1=0; j1 < i1 + saltLen; j1++)
{
 s[j1] = 0;
}

//Turn Y-Array into a String
unsigned long Y1[16];
for (int i=0; i<16;i++) Y1[i]=Y[i];
i1=0;
for(int i=0; i<16; i++)
{
	for(int j=0; j<4; j++)
	{
		Y[i]=Y1[i];	
		s[i1]= (Y[i] >> 8*(3-j)) % 256;
     		 i1++;
	}
}


unsigned long H1[5];
for(int i=0; i<5; i++) H1[i]=H[i];
		
//Append Hashed X-Array to Y-Array in string
for(int i=0; i<5; i++)
{
	for(int j=0; j<4; j++)
	{
		H1[i]=H[i];
		s[i1]= (H1[i] >> 8*(3-j)) % 256;
      		i1++;
	}
}

//Hash final concatenated string

SHA1(s,i1,H);

}

__device__ inline
int PBKDF2(int S[],int saltLen,int stored_pvv[],int dkLen,int *P, int passLen, unsigned char *Key)
{
   unsigned long T[17]={0},H[5]; 
   int U[20]={0};
   unsigned long L[5]={0,0,0,0,0};
   int i1, j1;
   dkLen = 2*dkLen + 2;

   int l = (dkLen/20)+1;
	
   for(int i=1; i<=l; i++)
	{
      for(i1 =0; i1 < saltLen; i1++)
      U[i1] = S[i1];
      U[i1++]=0x00;
      U[i1++]=0x00;
      U[i1++]=0x00;
      U[i1++]=i;
      //U chinh la S trong truong hop nay
      L[0] = L[1] = L[2] = L[3] = L[4] = 0;
	for(int j=1; j<=1000; j++)
		{
			HMAC(U,i1,P,passLen,H);
			L[0] = L[0]^H[0];
			L[1] = L[1]^H[1];
			L[2] = L[2]^H[2];
			L[3] = L[3]^H[3];
			L[4] = L[4]^H[4];
			for(j1= 0; j1 < i1; j1 ++) U[j1] = 0;
        		i1 =0;
			for(int x=0; x<5; x++)
			{
				for(int y=0; y<4; y++)
				{
					U[i1]= (H[x] >> 8*(3-y)) % 256;
             				i1++;
				}
			}
		}

		T[5*(i-1)] = L[0];
		T[5*(i-1)+1] = L[1];
		T[5*(i-1)+2] = L[2];
		T[5*(i-1)+3] = L[3];
		T[5*(i-1)+4] = L[4];
	}
     //Lay ra de kiem tra password verification value
   i1= (dkLen -2) /4;
   if ((stored_pvv[0] == ((T[i1] >> 24)&0x000000FF)) && (stored_pvv[1] == ((T[i1] >> 16) & 0x000000FF)) )
	{ 
	   for (int i=0;i<4;i++)
		{ 
			Key[4*i  ]=(unsigned char)(T[i]>>24);
			Key[4*i+1]=(unsigned char)(T[i]>>16);
			Key[4*i+2]=(unsigned char)(T[i]>>8);
			Key[4*i+3]=(unsigned char)(T[i]);	
		}
           return 1;
	}
 else return 0;
}	
	
/*Ham xu ly sinh mat khau dung tu dien*/
__device__
int d_strlen(char *str)
{
    int len = 0;
    for(int i=0; ; i++)
    {
		if(str[i] != '\0' && str[i] != '\n') len++;
		else break;
    }
	return len;
}

__device__
int d_atoi(char *str)
{
  int a[10] = {0,1,2,3,4,5,6,7,8,9};
  int result = 0;
  for(int i =0; i< d_strlen(str); i++)
  {
     result = result + a[str[i] - 48]*(int)__powf(10, d_strlen(str) - i - 1);
  }
  return result;
}

__device__
void d_strcpy(char *str1, char *str2)
{
   for(int i=0; i< d_strlen(str2); i++)
   {
       str1[i] = str2[i];
   }
   str1[d_strlen(str2)] = '\0';
}

__device__
void d_strcat(char *str1, char *str2)
{
	int len = d_strlen(str1);
	for(int i=0; i< d_strlen(str2); i++)
	{
		str1[len + i] = str2[i];
	}
  	str1[len + d_strlen(str2)] = '\0';
}

__device__
int d_strcmp(char *str1, char *str2)
{
   int len = d_strlen(str2);
   int result = 1;
   for(int i=0; i< len; i++)
   {
 	if (str1[i] == str2[i]) result = 1;
	else
	{
		result = 0;
		break;
	}
   }
   return result;
}

__global__ 
void RunKernel(int base,char *devPass, char *devPtr, int width, int qualities,int wordCount,char *pre_terminal,int pre_terminal_len, int *salt, int saltLen, int *stored_pvv, int dkLen,unsigned char *data,unsigned int len, unsigned char *out, unsigned char extension, unsigned char *Key,int* ret,fcrypt_ctx *zcx, aes_ctx *encr_ctx,z_stream *strm,struct inflate_state FAR *d_state)
{	
  int index = 0,ok=1;
  int begin;
  int end;
  char part1[20]="";
  char part2[20]="";
  char part3[20]="";
  char password[60] = "";
  char number_characters[5]=""; 
  unsigned char tmp_Key[16];
  int k1,k2,k3;  
  const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned char PDF[7]={0x25,0x50,0x44,0x46,0x2d,0x31,0x2e};
  unsigned char DOC[14]={0xd0,0xcf,0x11,0xe0,0xa1,0xb1,0x1a,0xe1,0x00,0x00,0x00,0x00,0x00,0x00};
  //pre_terminal duoc chia thanh ba phan, trong do part2 la can duoc ghep voi
  //tu co nghia trong tu dien. Part1 va part3 giua nguyen.

   //Luu vi tri phan chia
   for(int i= pre_terminal_len -1; i>=0; i--) 
   {
    if(pre_terminal[i] == 'H')
    {
      index++;
      if( ((index + 1)/2 == qualities) && (index % 2 == 1))
      {
        end = i;
      }
      if( ((index/2)== qualities) && (index % 2 == 0))
      {
        begin = i;
        break;
      }
    }
   }
   //Bat dau chia
   k1=k2=k3=0;
   for(int i=0; i< pre_terminal_len; i++) 
   {     
     if(i< begin)
     {
        part1[k1++] = pre_terminal[i];	 
     }
     else if(i>= begin && i<= end)
     {
        part2[k2++] = pre_terminal[i];	 
     }
     else
     {
     	 part3[k3++] = pre_terminal[i];	 	 
     }
   }
   part1[k1] = '\0';
   part2[k2] = '\0';
   part3[k3] = '\0';
	
   //Lay ra do dai xau can chen.
   index = 0;
   for(int i=0; i< k2;i++)
   {
      if(part2[i] >= '0' && part2[i] <='9')
	{
      		number_characters[index++] = part2[i];	
	}
   }
   number_characters[index] = '\0';
	
	if (base + id <= wordCount)
	{
		 d_strcpy((char*)devPass+ id*width, "");		
		//Lay ve buffer, chua mot tu
		char *buffer = (char*)((char*)devPtr + (base + id)*width);  //Chinh la row		 
		
		 if(d_strlen(buffer) == d_atoi(number_characters))
		 {			
			for(int g=0; g<60; g++)
			password[g] = '\0';
			d_strcpy(password,part1);
			d_strcat(password,buffer);
			d_strcat(password,part3);			
			
			if(qualities == 1)
			{				
				int myPass[60];
				for(int kk=0; kk< d_strlen(password); kk++) myPass[kk] = password[kk];
				if (PBKDF2(salt,saltLen,stored_pvv,dkLen,myPass,d_strlen(password),tmp_Key)) 
				{	
				   // for (int kk=0;kk<16;kk++) (Key+id*16)[kk]=tmp_Key[kk];
				   // ret[id]=id;
			 	   
				    //giai ma	
				    aes_set_encrypt_key(tmp_Key, KEY_LENGTH(1), encr_ctx+id);
				    encr_data(data+id*CHUNK,len,zcx+id,encr_ctx+id);

				    //giai nen	
				    (strm+id)->avail_in = 0;
				    (strm+id)->next_in = Z_NULL;
				    (void)inflateInit2(strm+id,-13,d_state+id);
				    
				    (strm+id)->avail_in = len;
				    (strm+id)->next_in = data+id*CHUNK;

				    (strm+id)->avail_out = CHUNK;
				    (strm+id)->next_out = out + id*CHUNK;

				    	
				    (void)inflate(strm+id, Z_NO_FLUSH,d_state+id);

				    //nhan dang 
				    switch(extension)
					{
						case dotpdf:  for (int i=0;i<7;i++) if (PDF[i]!=(out+id*CHUNK)[i]) ok=0;
							break;
						case dotdoc:  for (int i=0;i<14;i++) if (DOC[i]!=(out+id*CHUNK)[i]) ok=0;  
							break;
						case dottxt:  
							for (int i=0;i<len;i++) 
								if (((out+id*CHUNK)[i] < 0x20) || ((out+id*CHUNK)[i] > 0x7E)) ok=0;
							break;
						default:
							ok=0;
						break;
					}

				    //lay ket qua dung	
				    if (ok) d_strcpy((char*)devPass+ ((unsigned int)id)*width,password); 
				}					
			}
			else
			{
			   //giam qualities va goi de quy.				  
			}
		 }	   
	}      
}
/*Ket thuc phan them*/

