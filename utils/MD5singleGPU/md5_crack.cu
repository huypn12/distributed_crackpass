
#include <iostream>
#include <string>
#include <sstream>
#include <cutil.h>
#include "md5_crack_kernel.cu"
#include "MyBignum.h"

using namespace std;

dim3 threads(128);
dim3 blocks(16,16);

unsigned char	* d_base,*base;
unsigned char	* d_charset,*charset;
string str_charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
int				* d_ret;
uint4 htc;
unsigned int	 timer = 0;
int start,stop,charset_length;

void initHOST(){
	base = new unsigned char[16];
	start=1;
	stop=6;
	memset(base, 0, sizeof(unsigned char) * 16);

	charset_length = (int)str_charset.length();
	charset = new unsigned char[charset_length];
	for (int i = 0; i < charset_length; ++i)
		charset[i] = str_charset[i];
}

void initCUDA()
{
	cudaMalloc((void**)&d_base, sizeof(unsigned char) * 16);
	cudaMemcpy(d_base, base, sizeof(unsigned char) * 16, cudaMemcpyHostToDevice);
	cudaBindTexture(0, texRefBaseKey, d_base);

	cudaMalloc((void**)&d_charset, sizeof(unsigned char) * charset_length);
	cudaMemcpy(d_charset, charset, sizeof(unsigned char) * charset_length, cudaMemcpyHostToDevice);
	cudaBindTexture(0, texRefCharset, d_charset);

	int minusone = -1;
	cudaMalloc((void**)&d_ret, sizeof(int));
	cudaMemcpy(d_ret, &minusone, sizeof(int), cudaMemcpyHostToDevice);

	cutCreateTimer(&timer);
}

void init(){
	initHOST();
	initCUDA();
}

int RunKernel(unsigned char * base, int charset_length, int length, dim3 blocks, dim3 threads, uint4 htc)
{
	cudaMemcpy(d_base, base, sizeof(unsigned char) * 16, cudaMemcpyHostToDevice);
	
	GetHashes<<<blocks, threads>>>(d_ret, htc, charset_length,length); 
	
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "#DEVICE ERROR#: ", cudaGetErrorString(error));
		return -2;
	}

	int ret;
	cudaMemcpy(&ret, d_ret, sizeof(int), cudaMemcpyDeviceToHost);

	return ret;
}

void FreeCUDA()
{
	cutDeleteTimer(timer);

	cudaUnbindTexture(texRefCharset);
	cudaUnbindTexture(texRefBaseKey);
	cudaFree(d_ret);
	cudaFree(d_charset);
	cudaFree(d_base);
}

bool IsMD5Hash(string & hash)
{
	if (hash.length() != 32)
		return false;

	for (int i = 0; i < 32; ++i)
		if (!((hash[i] >= '0' && hash[i] <= '9') || (hash[i] >= 'a' && hash[i] <= 'f')  || (hash[i] >= 'A' && hash[i] <= 'F')))
			return false;

	return true;
}

void GetIntsFromHash(string hash)
{
	char* dummy = NULL;
	unsigned char bytes[16];
	for (int i = 0; i < 16; ++i)
		bytes[i] = (unsigned char)strtol(hash.substr(i * 2, 2).c_str(), &dummy, 16);

	htc.x = bytes[ 0] | bytes[ 1] << 8 | bytes[ 2] << 16 | bytes[ 3] << 24;
	htc.y = bytes[ 4] | bytes[ 5] << 8 | bytes[ 6] << 16 | bytes[ 7] << 24;
	htc.z = bytes[ 8] | bytes[ 9] << 8 | bytes[10] << 16 | bytes[11] << 24;
	htc.w = bytes[12] | bytes[13] << 8 | bytes[14] << 16 | bytes[15] << 24;
}

void crack(string hash){
	init();
	if (IsMD5Hash(hash))
		GetIntsFromHash(hash);
	cutStartTimer(timer);

	const int maxthreadid = ((blocks.y - 1) * blocks.x + blocks.x - 1) * threads.x + threads.x - 1;

	MyBignum bn_threadcount(threads.x * blocks.x * blocks.y);	// # of threads, that are executed during one kernel launch
	MyBignum bn_base(charset_length);

	for (int i = start; i <= stop; ++i)
	{
		cout << "Do dai chia khoa dang xet " << i << ".\n";

		// Tinh so luong chuoi co i ky tu
		MyBignum bn_full(1);
		for (int j = 0; j < i; ++j)
			bn_full *= bn_base;

		MyBignum bn_counter(0);

		memset(base, 0, sizeof(unsigned char) * 16);
		do
		{
			bn_counter += bn_threadcount;

			// Thuc hien kernel
			int ret = RunKernel(base, charset_length, i, blocks, threads, htc);
			if (ret == -2)
			{
				FreeCUDA();
				return;	// CUDA error -> dung pha ma
			}
			else if (ret != -1)	//	Tim thay ma giai
			{
				cout << "\tTong thoi gian: " << cutGetTimerValue(timer) << "ms";
				cout << "\t-\t" << bn_counter.ToString() << " ma da thu.\n";
				cout << "----------------------------------------------------------------------";
				cout << "\nChia khoa la:\t";

				// Lay lai chuoi ky tu cua ket qua
				unsigned int counter = ret;
				for (int j = 0, a = 0, carry = 0; j < i; ++j, counter /= charset_length)
				{
					a = base[j] + carry + counter % charset_length;
					if (a >= charset_length) { carry = 1; a -= charset_length; }
					else carry = 0;
					cout << str_charset[a];
				}
				cout << endl << "======================================================================\n";

				FreeCUDA();
				return;
			}

			// Advance texRefBaseKey
			int counter = maxthreadid;
			for (int j = 0, a = 0, carry = 0; j < i; ++j, counter /= charset_length)
			{
				a = base[j] + carry + counter % charset_length;
				if (a >= charset_length) { carry = 1; a -= charset_length; }
				else carry = 0;
				base[j] = a;
			}

		}while(bn_counter < bn_full);
		cout << "\tTong thoi gian: " << cutGetTimerValue(timer) << "ms";
		cout << "\t-\t" << bn_counter.ToString() << " ma da thu.\n";
	}

	cutStopTimer(timer);
}

int main(){
	string MD5Hash,temp;
	cout << endl << "=====================================================================\n";
	cout<<"MD5 Hash to Crack:";
	cin>>MD5Hash;
	cout<<"\nBang ma can tim(0 = bang ma mac dinh):";
	cin>>temp;
	if (temp!="0")
		str_charset = temp;
	crack(MD5Hash);
}

