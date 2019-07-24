#ifndef _ENTRY_
#define _ENTRY_
//Khai bao bien ngoai
char Column1[100][3] = {""};
char Column2[100][20] = {""};
float Column3[100] = {0};

struct entry
{
  char base[20];
  char pre_terminal[20];
  int pivot;
  int num_strings;
  float probability;
  struct entry *next;
  struct entry *prev;
};

//Khai bao cac nguyen mau ham
int ReadRules();
struct entry * Pop(struct entry *head);
FILE* ReadDict(char fileName[20]);

#endif

