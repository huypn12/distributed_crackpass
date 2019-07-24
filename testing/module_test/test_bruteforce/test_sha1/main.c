#include "sha.h"

#include <stdio.h>


unsigned char digest[20];
unsigned char message[3] = {'a', 'b', 'c' };
unsigned char *mess56 =
	"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

/* Correct solutions from FIPS PUB 180-1 */
char *dig1 = "A9993E36 4706816A BA3E2571 7850C26C 9CD0D89D";
char *dig2 = "84983E44 1C3BD26E BAAE4AA1 F95129E5 E54670F1";
char *dig3 = "34AA973C D4C4DAA4 F61EEB2B DBAD2731 6534016F";

/* Output should look like:-
 a9993e36 4706816a ba3e2571 7850c26c 9cd0d89d
 A9993E36 4706816A BA3E2571 7850C26C 9CD0D89D <= correct
 84983e44 1c3bd26e baae4aa1 f95129e5 e54670f1
 84983E44 1C3BD26E BAAE4AA1 F95129E5 E54670F1 <= correct
 34aa973c d4c4daa4 f61eeb2b dbad2731 6534016f
 34AA973C D4C4DAA4 F61EEB2B DBAD2731 6534016F <= correct
*/

int main()
{
	SHA_CTX sha;
	int i;
	BYTE big[1000];

	SHAInit(&sha);
	SHAUpdate(&sha, message, 3);
	SHAFinal(digest, &sha);

	for (i = 0; i < 20; i++)
	{
		if ((i % 4) == 0) printf(" ");
		printf("%02x", digest[i]);
	}
	printf("\n");
	printf(" %s <= correct\n", dig1);

	SHAInit(&sha);
	SHAUpdate(&sha, mess56, 56);
	SHAFinal(digest, &sha);

	for (i = 0; i < 20; i++)
	{
		if ((i % 4) == 0) printf(" ");
		printf("%02x", digest[i]);
	}
	printf("\n");
	printf(" %s <= correct\n", dig2);

	/* Fill up big array */
	for (i = 0; i < 1000; i++)
		big[i] = 'a';

	SHAInit(&sha);
	/* Digest 1 million x 'a' */
	for (i = 0; i < 1000; i++)
		SHAUpdate(&sha, big, 1000);
	SHAFinal(digest, &sha);

	for (i = 0; i < 20; i++)
	{
		if ((i % 4) == 0) printf(" ");
		printf("%02x", digest[i]);
	}
	printf("\n");
	printf(" %s <= correct\n", dig3);

    printf("# Extra test ...\n");

    unsigned char *mesg = "aaaaaa";
	SHAInit(&sha);
    SHAUpdate(&sha, mesg, 6);
	SHAFinal(digest, &sha);

	for (i = 0; i < 20; i++)
	{
		if ((i % 4) == 0) printf(" ");
		printf("%02x", digest[i]);
	}
	printf("\n");
	return 0;
}



