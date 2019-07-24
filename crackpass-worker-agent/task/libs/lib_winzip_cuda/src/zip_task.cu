#include <iostream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <cutil.h>
#include <multithreading.h>
//#include <cutil_inline.h> //@deprecated
#include <cuda_runtime_api.h>
#include "crackkernel.cu"
#include "entry.cpp"
#include "zip.c"
#include "pbkdf2.cpp"

using namespace std;
dim3 threads(128);
dim3 blocks(12,10);
int threadcount = threads.x*blocks.x*blocks.y;
//Chot dung de chia cot cua hang.
int width = 80;
unsigned char extension,in[CHUNK],found=0;
/*Bien luu thong tin ve tep da duoc ma hoa
  - mang chua gia tri PVV 
  - mang chua gia tri salt	
 */
int dkLen,saltLen,len;
int stored_pvv[2];		
int S[16];		

