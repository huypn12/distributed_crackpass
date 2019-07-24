#include <stdio.h>
#include <stdlib.h>
#include "entry.h"

int ReadRules(char *fileName)
{
int count;
  	int i;
  	char ch;
  	char buffer[80];  	
	FILE *fp;
   
    fp =fopen(fileName, "r");     
  	if(fp != NULL)     
   {
   	count = 1;	
    	while(!feof(fp))
      {	
      	//1. Loai bo cac tab truoc Column1.
         i = 0;
         strcpy(buffer, "");
         while((ch = fgetc(fp)) == '\t');

         if(feof(fp)) break;

         while (ch != '\t')
         {
         	buffer[i] = ch;
            i++;
            ch = fgetc(fp);
         }

         buffer[i] = '\0';
         strcpy(Column1[count],buffer);	

         //2. Loai bo tab truoc Column2
         i = 0;
         strcpy(buffer, "");
         while((ch = fgetc(fp)) == '\t');

         while (ch != '\t')
         {
         	buffer[i] = ch;
            i++;
            ch = fgetc(fp);
         }

         buffer[i] = '\0';
         strcpy(Column2[count],buffer);	 

         //Loai bo tab truoc Column3
         while((ch= fgetc(fp)) == '\t');
         
         i = 0;
         strcpy(buffer, "");
         while (ch != '.')
         {
         	buffer[i] = ch;
            i++;
            ch = fgetc(fp);
         }
         buffer[i] = '\0';
         Column3[count] = atoi(buffer);

         //Tiep tuc xu ly sau dau phay
         i = 0;
         strcpy(buffer, "");

         ch = fgetc(fp);
         while ( (ch >= '0') && (ch <= '9'))   //Ki tu ket thuc xau
         {
         	buffer[i] = ch;
            i++;
            ch = fgetc(fp);
         }
         buffer[i] = '\0';
         Column3[count] = Column3[count] + (float)(atoi(buffer))/powf(10,i);

         //Hien thi lai ba thong tin
         printf("\n%3s %10s %f", Column1[count], Column2[count], Column3[count]);

         //Tang count len mot
         count++;         
      } //Het while, ta duoc cau truc cua 3 colum
      fclose(fp);
   }

return count;
}

struct entry * Pop(struct entry *head)
{
  struct entry *search_value;
  float max_probability;
  
  search_value = head;
   if(head != NULL)
   {
   	max_probability = search_value->probability;
   	while(search_value->next != NULL)
   	{
       	if(max_probability < (search_value->next)->probability)
       	{
          	max_probability =  (search_value->next)->probability;
       	}
       	search_value = search_value->next;
   	}
   	//pop theo thu tu tu uu tien xac suat
   	search_value = head;
   	while(search_value != NULL)
   	{
       	if(max_probability == search_value->probability)
       	{
           	//Thi lay ra va break luon
           	return search_value;           	
       	}
       	else
       	{
           	search_value = search_value->next;
       	}
   	}
   }
   //else
   //{
      return NULL;
   //}  
}

