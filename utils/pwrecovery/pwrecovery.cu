#include <iostream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <cutil.h>
#include <multithreading.h>
#include <cutil_inline.h>
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

/*Danh cho da GPU, lay ve so GPU duoc CUDA ho tro, co the dung de tinh toan*/
const int MAX_GPU_COUNT = 8;
int GPU_N;
CUTThread threadID[MAX_GPU_COUNT];
TGPUplan plan[MAX_GPU_COUNT];
char *emptyArr;
char temp2[20];
float time_parallel = 0;
//Chua cac tu trong tu dien
char hostArray[869229][80]; 

/*Ket thuc*/	
fcrypt_ctx  h_zcx[1];

/*Bien chua thong tin van pham*/
int v_prev, v_curr;
int Sk[33];
/*Ket thuc bien chua thong tin van pham*/

void initHost(){
	/*Hien thi so GPU duoc ho tro*/
	cutilSafeCall(cudaGetDeviceCount(&GPU_N));
	if(GPU_N > MAX_GPU_COUNT) GPU_N = MAX_GPU_COUNT;
	printf("\nCUDA-capable device count: %i\n", GPU_N);	
	/*Ket thuc qua trinh hien thi*/
	
	emptyArr = (char*)malloc(sizeof(char)*width*threadcount);
	memset(emptyArr, '\0', sizeof(char)*width*threadcount);
	
	for (int i=0;i<GPU_N;i++)
	{
		//khoi tao plan->device
		plan[i].device=i;		
		// Chuong trinh moi giai quyet van de la quantities = 1
		plan[i].quantities = 1; 
	}		
	//khoi tao cho zcx
        h_zcx->mode = 1;
        h_zcx->encr_pos = BLOCK_SIZE;
        memset(h_zcx->nonce, 0, BLOCK_SIZE * sizeof(unsigned char)); 	
}
void freeCUDA()
{

	for (int i=0;i<GPU_N;i++)
	{
	cudaFree(plan[i].devPass);
	cudaFree(plan[i].d_pre_terminal);
	cudaFree(plan[i].deviceArrPtr);
	cudaFree(plan[i].d_salt);
	cudaFree(plan[i].d_pvv);
	cudaFree(plan[i].d_in);
	cudaFree(plan[i].d_out);
	}
}

static CUT_THREADPROC solverThread(TGPUplan *plan){
   /******************************************************************
	Khai bao bien
	******************************************************************/	
	//devPitch - truyen vao nhung khi lay gia tri ra thi lai khong dung den no.
	size_t devPitch;
	int pivot_base = 0;
	int ret[threadcount];
	//Khai bao mang hostPass de hien thi cac mat khau tra ve.
	char hostPass[threadcount][80];
	memset(hostPass,'\0', sizeof(char)*threadcount*80);
	/*****************************************************************
	Ket thuc khai bao bien
	******************************************************************/
	memset(ret,-1,sizeof(int)*threadcount);
	
	/*****************************************************************
	Cap phat bo nho tren moi GPU, truyen du lieu can cho tinh toan tu Host len Device
	*****************************************************************/	
	
	//Set device
       cutilSafeCall(cudaSetDevice(plan->device));		
	cudaMallocPitch((void**)&plan->devPass, &devPitch, width * sizeof(char), threadcount);	
	
	//Khoi tao plan->deviceArrPtr cho moi GPU
	cudaMallocPitch((void**)&plan->deviceArrPtr, &devPitch, width * sizeof(char), plan->wordCount);  
	cudaMemcpy2D(plan->deviceArrPtr, width*sizeof(char), hostArray + plan->startIndex, width*sizeof(char), width, plan->wordCount, cudaMemcpyHostToDevice);
		
	//Khoi tao gia tri kiem tra mat khau tren moi GPU
	cudaMalloc((void**)&plan->d_salt, sizeof(int) * 16);
	cudaMemcpy(plan->d_salt, S, sizeof(int) * 16, cudaMemcpyHostToDevice);	
	cudaMalloc((void**)&plan->d_pvv, sizeof(int) * 2);
	cudaMemcpy(plan->d_pvv, stored_pvv, sizeof(int) * 2, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&plan->d_pre_terminal, sizeof(char) * strlen(temp2)); 
	cudaMemcpy(plan->d_pre_terminal, temp2, sizeof(char) * strlen(temp2), cudaMemcpyHostToDevice);
 
	cudaMalloc((void**)&plan->d_out, threadcount*CHUNK*sizeof(unsigned char));
	cudaMalloc((void**)&plan->d_in,threadcount*CHUNK*sizeof(unsigned char));
	cudaMalloc((void**)&plan->d_Key,threadcount*16*sizeof(unsigned char));
	cudaMalloc((void**)&plan->d_ret,threadcount*sizeof(unsigned int));
	plan->Key = (unsigned char *)malloc(sizeof(unsigned char)*16*threadcount);



       //cap phat bo nho cho phan giai ma
	cudaMalloc((void**)&plan->d_zcx,threadcount*sizeof(fcrypt_ctx));
	cudaMalloc((void**)&plan->d_acx,threadcount*sizeof(aes_ctx));

 
	//cap phat bo nho cho phan giai nen 
	cudaMalloc((void**)&plan->d_strm, threadcount*sizeof(z_stream));
	cudaMalloc((void**)&plan->d_state,threadcount*sizeof(struct inflate_state FAR));
	
	
	/****************************************************************
	Ket thuc qua trinh truyen du lieu
	*****************************************************************/
	
	
	/****************************************************************
	Qua trinh goi Kernel nhieu lan, viec goi la doc lap giua cac Device
	*****************************************************************/
	pivot_base = plan->device*threadcount;	
	//cutilSafeCall(cudaThreadSynchronize());
	while((pivot_base < plan->wordCount)&&(!found))
		{		
			//Reset lai cac gia tri truoc moi lan chay Kernel 
			cudaMemcpy2D(plan->devPass, width*sizeof(char), emptyArr,width*sizeof(char), width, threadcount, cudaMemcpyHostToDevice);			
			cudaMemset (plan->d_out, 0, threadcount*CHUNK);
			for (int i=0;i<threadcount;i++)	{	
	         		cudaMemcpy(plan->d_in+i*CHUNK, in, CHUNK*sizeof(unsigned char), cudaMemcpyHostToDevice);
				cudaMemcpy(plan->d_zcx+i, h_zcx, sizeof(fcrypt_ctx), cudaMemcpyHostToDevice);}

			cudaMemset (plan->d_ret, -1, threadcount*sizeof(int));
			//chay kernel
			RunKernel<<<blocks, threads>>>(pivot_base, plan->devPass,plan->deviceArrPtr, width, plan->quantities, plan->wordCount, plan->d_pre_terminal,strlen(temp2), plan->d_salt, saltLen, plan->d_pvv, dkLen,plan->d_in,len,plan->d_out,extension,plan->d_Key,plan->d_ret,plan->d_zcx,plan->d_acx,plan->d_strm,plan->d_state); 
			cudaError_t error = cudaGetLastError();
			if (error != cudaSuccess)
			{
				fprintf(stderr, "#DEVICE ERROR#: ", cudaGetErrorString(error));
				freeCUDA();
				return ;
			}
			else 
			{		
				//Cap nhat lai pivot_base
				pivot_base += GPU_N*threadcount;		
				cudaMemcpy2D(hostPass, width*sizeof(char), plan->devPass, width*sizeof(char),width,threadcount, cudaMemcpyDeviceToHost);	
				cudaMemcpy(ret,plan->d_ret,sizeof(int)*threadcount,cudaMemcpyDeviceToHost);
				cudaMemcpy(plan->Key,plan->d_Key,sizeof(unsigned char)*16*threadcount,cudaMemcpyDeviceToHost);
				//cout << "\n----------------------------------------------------------------------\n";				
				//cout << "\tTong thoi gian: " << cutGetTimerValue(timer) << "ms";
				//cout << "\t" << pivot_base << "/" << GPU_N << " ma da thu.\n";
				
				for (int i1=0; i1 < threadcount; i1++)
				if (strcmp(hostPass[i1], "") != 0) 	//Tim thay ma giai			
				{									
					cout << "\nThe correct password is: ";					
					cout << hostPass[i1] << "\n";
					found=1;
				}
			}				
			cutilSafeCall(cudaThreadSynchronize());		
		} 	
	/*****************************************************************
	Ket thuc qua trinh goi kernel nhieu lan, doc lap giua cac Device.
	*****************************************************************/
	cudaFree(plan->devPass);
	cudaFree(plan->d_pre_terminal);
	cudaFree(plan->deviceArrPtr);
	cudaFree(plan->d_salt);
	cudaFree(plan->d_pvv);
	cudaFree(plan->d_out);
	cudaFree(plan->d_in);
	cudaFree(plan->d_Key);
	cudaFree(plan->d_ret);
	cudaFree(plan->d_zcx);
	cudaFree(plan->d_acx);
	cudaFree(plan->d_strm);
	cudaFree(plan->d_state);
	
	free(plan->Key);

	/*****************************************************************
	Lenh dinh thoi gian va lenh thoat tat ca cac tien trinh
	******************************************************************/	
		
	cudaThreadExit();	
	CUT_THREADEND;	 
	/*****************************************************************
	Ket thuc 
	******************************************************************/
}

void crack(){	
	unsigned int timer=0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	/*Moi tien trinh tren CPU quan ly mot GPU, ta co GPU_N nen can co GPU_N tien trinh song song tren Host quan ly */
	for(int GPUIndex = 0; GPUIndex < GPU_N; GPUIndex++)
		threadID[GPUIndex] = cutStartThread((CUT_THREADROUTINE)solverThread, &plan[GPUIndex]);
		
	//printf("main(): waiting...\n");
	cutWaitForThreads(threadID, GPU_N);
	cout <<cutGetTimerValue(timer) << "ms\n";
       cout << "\n---------------------------------------------------------------------------------------------------------------\n";
       time_parallel += cutGetTimerValue(timer);
	cutStopTimer(timer);
	cutDeleteTimer(timer);
}

void readGrammar(char *filename1, char *filename2, int *count)
{
	memset(Sk, 0, 33*sizeof(int));
	printf("\nA grammar:");
	*count = ReadRules(filename1); 	//argv[2]
	FILE *fp;   
    	char buffer[80] = "";   
    	fp =fopen(filename2, "r"); 		//argv[3]
    
    	//Khoi tao hostArray.
    	if (fp != NULL)
    	{
      		int h = 0;
	  	while(fgets(buffer, sizeof(buffer), fp) != NULL)
	  	{
			if(h==0)
			{
				v_prev= v_curr = strlen(buffer)-1;
				Sk[v_curr] = h;
			}
			v_curr = strlen(buffer)-1;
			if(v_curr != v_prev)
			{
				Sk[v_curr] = h;
				v_prev = v_curr;
			}
			strcpy(hostArray[h], buffer);
			h++;
			strcpy(buffer, "");
	  	}
	  	fclose(fp);
    	}
}

int checkInfo(char *filename)
{
	ZIP_LOCAL_FILE_HEADER* lfh;
	FILE* pt;
	pt = fopen(filename, "rb");
	lfh = (ZIP_LOCAL_FILE_HEADER*) malloc(sizeof(ZIP_LOCAL_FILE_HEADER));
	if(!pt) return -1;
	read_lfh(lfh, pt);	

	if(get_bit_encrypted_ornot(lfh, pt) != 1)
		{
				cout<< "File is not encrypted";
			return -1;
		}
	else 
		{
			char *cp;		
			cp = strrchr(get_fname(lfh, pt), '.');
			if (strcmp(cp, ".pdf")==0) extension = dotpdf;
			if (strcmp(cp, ".doc")==0) extension = dotdoc;
			if (strcmp(cp, ".txt")==0) extension = dottxt;	
			*cp=0;
			printf("File is encrypted , parameters:");
			/*---------------------------------------------------------------------------------------------
			Lay gia tri salt, authentication code, password verification value chi co,  khi file da encrypt
			----------------------------------------------------------------------------------------------*/
			display_salt_pvv_ac(lfh,pt,S,&saltLen,stored_pvv,&dkLen);
			fseek(pt, 30 + lfh->fn_length + lfh->ex_length + SALT_LENGTH(1) + PWD_VER_LENGTH, SEEK_SET);
			len = (int)fread(in, sizeof(unsigned char),CHUNK, pt);
			fclose(pt);			
		}
	return 1;
}
void multiFunction(int light, int count)
{	    
    struct entry *working_value = NULL;
    struct entry *head = NULL;
    struct entry *tail = NULL;
    int status = 0;       
    
	if(light == 6)
	{
		//Goi khoi tao host mot lan
		initHost();
	}

	char temp_preterminal[20] = "";
	char search_characters[4]=""; 
	char temp_base[20]="";    
       
	//Xay dung cay va duyet cau truc dang pre-terminal
	//1. Phan 1: all base structs
	for(int i = 1; i< count; i++)
	{
		if(strcmp(Column1[i],"S") == 0)
		{
			//Xoa search_character va temp_terminal
			strcpy(search_characters,"");
			strcpy(temp_preterminal,"");

			working_value = (struct entry *)malloc(sizeof(struct entry));
			strcpy(working_value->base,Column2[i]);
			working_value->pivot = 0;
			working_value->num_strings = 0;
			for(int j = 0; j< strlen(Column2[i]); j++)
			{
				if(Column2[i][j] == 'L' ||  Column2[i][j] == 'S' || Column2[i][j]=='D')
					working_value->num_strings++;
			}
			//Tinh xac suat va pre_terminal
			working_value->probability = Column3[i];
			//Duyet cau truc cua Column2 de tinh xac suat.
			int k;
			char temp[2];
			for(int j = 0; j< strlen(Column2[i]);)
			{
				k = 0;
				search_characters[k] = Column2[i][j++];
	            while((Column2[i][j] != 'D') && (Column2[i][j] != 'L')  && (Column2[i][j] != 'S'))
	            {
	               search_characters[++k] = Column2[i][j++];
	               if(Column2[i][j] == '\0') break; 
	            }
				//Thoat co nghia vi tri j da la bat dau mot ki tu moi, nhung chua gan. k tang len mot gia tri
				search_characters[++k] = '\0';
				//Kiem tra ki tu dau co phai la ki tu L. Neu la L chi cap nhat lai xau pre_terminal de phan biet. Khong
				//cap nhat xac suat.
				if (search_characters[0] == 'L')
				{
					temp[0] = 'H';
					temp[1] = '\0';
					strcat(temp_preterminal, temp);
					strcat(temp_preterminal,search_characters);
					strcat(temp_preterminal, temp);
				}
				else
				{
					//Neu khong phai, thi tim kiem va cap nhat lai xac suat
					for(int t = 1; t < count; t ++)
					{
						if(strcmp(Column1[t],search_characters) == 0)
						{
							strcat(temp_preterminal,Column2[t]);
							working_value->probability = working_value->probability * Column3[t];
							break;
						}
					}
				} //Ket thuc la ki tu D hoac S
				//Cap nhat xac suat lon nhat roi thoat
			}// Het vong for, thi da xac dinh duoc xac suat, va dong thoi la pre_terminal
			strcpy(working_value->pre_terminal,temp_preterminal);

			//Buoc cuoi cua giai doan 1: Them no vao queue uu tien
			if(status ==0)
				{
					working_value->next = NULL;
					working_value->prev = NULL;
					head = tail = working_value;
					status = 1;
				}
			else
				{
					//Them vao cuoi queue
					working_value->next = NULL;
					working_value->prev = tail;
					tail->next = working_value;
					tail = working_value;
				}
				working_value = NULL;
		}
		else
		{
			break;
		} //ket thuc cua if-else
	} //Ket thuc cua for.
   
    	/*Buoc 2. Vua xay dung cay, vua dua ra danh sach mat khau, lam dau vao cho giai thuat PBKDF2
   	cai nay co the dua vao mot ham, phan cap chuc nang
   	Co the toi uu chuc nang tim kiem, thuc hien pop nhanh hon.
   	Giai thuat nay co the thuc hien song song hoa duoc, giong nhu giai thuat tim kiem song song tren danh sach.
	*/
	int order=0;	
	working_value = Pop(head); 
	if(light == 6)
	{		
		printf("\n%-12s %-15s %-10s %-15s %-15s %-15s %-15s %-15s\n","Base","pre_terminal","pivot","num_strings","probability","order", "Keys","Time");
		cout << "\n----------------------------------------**-----------------------------------**----------------------------------\n";
	}
	else if(light == 3)
	{
		printf("\n%-12s %-15s %-10s %-15s %-15s %-15s\n","Base","pre_terminal","pivot","num_strings","probability","order");
		cout << "\n-------------------------------**----------------------------**-----------------------------\n";
	}
	
	while((working_value != NULL)&&(!found))
	{
		order++;  	   	
		int qualities = 0;
		int sk;
		for(int h = 0; h< strlen(working_value->pre_terminal); h++)
			if(working_value->pre_terminal[h] == 'L') 
			{
				qualities++;
				sk = (int)working_value->pre_terminal[h + 1] - 48;		
			}
		strcpy(temp2, working_value->pre_terminal);
		if(light == 6)
		{				
			/* truyen cac thong so pre_terminal  lay duoc tu thao tac Pop sang devce - GPU_N device*/ 
			for(int deviceIndex = 0; deviceIndex < GPU_N; deviceIndex++)
			{		
				plan[deviceIndex].wordCount = Sk[sk+1] - Sk[sk];	  
				plan[deviceIndex].startIndex = Sk[sk];	 
			}		 
			/*Goi song song GPU_N tien trinh tren CPU quan ly GPU_N GPU*/	
			//Sinh cac mat khau bang cach ghep cau truc pre_terminal voi tu dien chua cac tu co nghia.
			printf("\n%-12s %-15s %-10d %-15d %-15f %-15d %-15ld",working_value->base,working_value->pre_terminal,
			working_value->pivot,working_value->num_strings, working_value->probability,order,Sk[sk+1] - Sk[sk]);	
			crack();
		}
		else if(light == 3)
		{
			printf("%-12s %-15s %-10d %-15d %-15f %-15d\n",working_value->base,working_value->pre_terminal,
			working_value->pivot,working_value->num_strings, working_value->probability,order);
			DisplayGuestPassword(working_value->pre_terminal, strlen(working_value->pre_terminal),hostArray,1, Sk[sk], Sk[sk+1], S, saltLen, stored_pvv, dkLen,3);
		}
		else if(light == 4)
		{
			printf("%-12s %-15s %-10d %-15d %-15f %-15d\n",working_value->base,working_value->pre_terminal,
			working_value->pivot,working_value->num_strings, working_value->probability,order);
			DisplayGuestPassword(working_value->pre_terminal, strlen(working_value->pre_terminal),hostArray,1, Sk[sk], Sk[sk+1], S, saltLen, stored_pvv, dkLen,4);
		}
	

		//Tiep tuc xay dung cay, insert va pop entry
		for(int i= working_value->pivot; i< working_value->num_strings; i++)
		{
			strcpy(temp_base, working_value->base); 				// temp_base = "D1L3S2D1"
			/*Khai bao du lieu, chi co pham vi trong vong for nay */
			int k;               						// 	Chi so chay trung gian
			char temp[2]; 							// 	temp[2] = 'L' || 'S' || 'D'
			char temp1[2]; 							//	temp1[2] = 'H' -> Dung trong phan cach L.
			int index = -1; 							//	index cua variable, chi biet co replace duoc hay khong.
			strcpy(temp_preterminal,""); 					//	xoa xau temp_preterminal, de sau do dung lai (khai bao gan ham main)
         											// 	child_value->pre_terminal = temp_preterminal.
			int segment = 0; 							//	chi so base, cho biet cat tu xau base tu dau den dau.
         											// 	vi du 4L3$$4, cat S2 tu dau den dau

			char temp_copy[10];							//	xau tu segment cho den het (segment + (int)atoi(search_characters)

			/*Phan tich temp_base de lay chu so va chi thi la D, L hay S. No cho phep minh biet cach doc bao nhieu ki
			tu tu cau truc pre_terminal cua working_value sang child_working_value*/
			//Bien cho biet co chen them vao duoc hay khong
			bool agreement = false;
			float reprobability = working_value->probability;
			for(int j = 0; j < strlen(temp_base);)
			{
				strcpy(search_characters,"");// xoa search_characters, dung lai bien o phan tren.
            							 // chang han search_characters = 1 hoac 2 hoac 1, nho loc bo ki tu

                                            		 // D truoc D1, ki tu S truoc S2, ki tu D truoc D1 cua temp_base.
				/* Lay ki tu dau tien cua temp_base*/
				k=0;
				temp[0] = temp_base[j];
				temp[1] = '\0';
				/*end */
				j = j +1;
				while((temp_base[j] != 'D') && (temp_base[j] != 'L')  && (temp_base[j] != 'S'))
				{
					search_characters[k++] = temp_base[j++];
					if(temp_base[j] == '\0') break;
				}
             			//Ket thuc xau
				search_characters[k] = '\0';
				index++;
				//temp_preterminal
				if(temp[0] == 'L')
				{
					if(index == i)
					{
						agreement = false;
						break; //Thoat ra khoi for theo j.
					}
					temp1[0] = 'H';
					temp1[1] = '\0';
					strcat(temp_preterminal, temp1);
					strcat(temp_preterminal, temp);
					strcat(temp_preterminal, search_characters);
					strcat(temp_preterminal, temp1);
					//Phai cap nhat lai segment
					segment = segment + 3 + strlen(search_characters);
				}
				else
				{
					//Phai tinh den so sanh index voi chi so i.
					if(index != i)
					{
						//Chi don thuan la copy cau truc tu vi tri segment cho den het (segment + (int)atoi(search_characters))
						strcpy(temp_copy,"");      	// Chi luu tru tam thoi
						int q;
						for(q = segment; q < segment + (int)atoi(search_characters); q++)
						{
							temp_copy[q-segment] = working_value->pre_terminal[q];
						}
						temp_copy[q-segment] = '\0';
						//Cap nhat lai segment, de cho lan copy sau.
						segment = segment + (int)atoi(search_characters);
						strcat(temp_preterminal, temp_copy);
					}
					else if(temp[0] == 'L')
					{
						agreement = false;
						break; //Thoat ra khoi for theo j.
					}
					else //Neu vao trong day ma khong thay the xau moi thi huy bo.
					{
						//Ghep giua temp voi search_characters lai voi nhau de ra dang, chang han nhu S2 => Goi la search_str.
						//Trich xuat ki tu o working_value->pre_terminal, tai vi tri segment den segment + (int)atoi(search_characters).
						//duoc goi la pointed_str. Neu thay the duoc, thi cap nhat luon xac suat cua no, dong thoi tao ra them duoc nut
						//moi
						char search_str[4];
						char pointed_str[4];
						strcpy(search_str,temp);
						strcat(search_str,search_characters);

						strcpy(temp_copy,"");   		//ok da xoa temp_copy
						int q;
						for(q = segment; q < segment + (int)atoi(search_characters); q++)
						{
							temp_copy[q-segment] = working_value->pre_terminal[q];
						}
						temp_copy[q-segment] = '\0';
						strcpy(pointed_str, temp_copy);

						//Tim kiem de thay the. Chu yeu la do tim vi tri d.
						for(int d = 1; d < count; d++)
						{
							if(strcmp(Column1[d],search_str)==0)
							{
								if(strcmp(Column2[d], pointed_str)==0)
								{
									segment += strlen(pointed_str);
									if( (d+1 < count) && (strcmp(Column1[d+1],search_str)==0))
									{
									//Them moi duoc, nghia la con ki tu thay the, xu ly tai day
									//Neu thay the duoc, thi copy cho den het j
									strcat(temp_preterminal,Column2[d+1]);
									// Tinh lai xac suat
									reprobability = (reprobability*Column3[d+1])/Column3[d];
									agreement = true;
									break;
								}
								else
								{
									//Vi tri nay da het cho. Quay tro ve tang i len, cho den het.
									agreement = false;
									break;
								}

							}
						}
					}  //Ket thuc for tim kiem xau thay the

				} //Ket thuc else - index
               
            } //Ket thuc else - L

        } //Ket thuc vong lap theo temp_base.
        if(agreement == true)
        {
          //Them moi vao cuoi danh sach dang xet.
          struct entry *child_value;
          child_value = (struct entry *)malloc(sizeof(struct entry));
          strcpy(child_value->base,working_value->base);
          strcpy(child_value->pre_terminal,temp_preterminal);
          child_value->pivot = i;
          child_value->num_strings = working_value->num_strings;
          child_value->probability = reprobability;

          child_value->next = NULL;
          child_value->prev = tail;
          tail->next = child_value;
          tail = child_value;
        }
		} //Ket thuc for theo bien chay i

		//Sau do thi giai phong entry working_value.
		if(working_value->prev == NULL)
		{
			if(working_value->next == NULL)
			{
				free(working_value);
				head = tail = NULL;
			}
			else
			{
				(working_value->next)->prev = NULL;
				head = (working_value->next);
				free(working_value);
			}
		}
		else
		{
			if(working_value->next == NULL)
			{
				(working_value->prev)->next = NULL;
				tail = working_value->prev;
				free(working_value);
			}
			else
			{
				(working_value->next)->prev = working_value->prev;
				(working_value->prev)->next = working_value->next;
				free(working_value);
			}
		}
		working_value = Pop(head);
	}  	// Ket thuc vong lap while  
	if(light == 6)
	{			
		cout << "\nThe end ...\n";
	}
}

void checkCandidatePasswords()
{
	int P[60]={0};
	string password = "";
	int passLen;	
	cin.get();	
	printf("\nNhap mat khau kiem tra:\n");	
	getline(cin, password);
	passLen = password.length();
	for(int i = 0; i < passLen; i++)
       P[i] = password[i];
	if(PBKDF2_1(S,saltLen,stored_pvv,dkLen,P, passLen) != 0)
		printf("\nLa mat khau ung cu");
	else
		printf("\nKhong phai la mat khau ung cu");
	
}
int main(int argc, char *argv[]){	
	int isEncrypted = 0;
	char ch;
	int count;
	while(1)
	{
		printf("\n1.Thong tin co ban cua tep nen Zip va van pham");
		printf("\n2.Kiem tra mot mat khau la ung cu");
		printf("\n3.Sinh mat khau tuan tu");
		printf("\n4.Tap mat khau ung cu - tt tuan tu");
		printf("\n5.Sinh mat khau song song");
		printf("\n6.Pha mat khau song song");	
		printf("\n7.Thoat chuong trinh");
		printf("\nLua chon chuc nang(1->7):");
		fflush(stdin);
		fflush(stdin);
	       ch = getchar();
	       switch(ch)
		{
			case '1':					
					isEncrypted = checkInfo(argv[1]);
					printf("\nisEncrypted = %d", isEncrypted);
					if (isEncrypted == 1) readGrammar(argv[2], argv[3], &count);	
					cin.get();				
					break;
			case '2':					
					if(isEncrypted == 1)
					{
						checkCandidatePasswords();
					}
					else
					{
						printf("\nPhai goi chuc nang 1 truoc");
					}	
					cin.get();				
					break;
			case '3':					
					multiFunction(3,count);
					cin.get();
					break;
			case '4':					
					multiFunction(4,count);
					cin.get();
					break;
			case '5':					
					multiFunction(5,count);
					cin.get();
					break;
			case '6':					
					if (isEncrypted == 1)
					{						
						multiFunction(6,count);
					}
					else
					{
						printf("\nPhai goi chuc nang 1 truoc");
					}
					cin.get();
					break;
			case '7':exit(1);
		}
	}    			
}
