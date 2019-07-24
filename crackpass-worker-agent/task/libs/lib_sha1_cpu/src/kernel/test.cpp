#include "SHA1CpuKernel.cpp"

int main(int argc, char const* argv[])
{
    // Test SHA1 functionality
        /*
    const char *mesg = "abc";
    int buffer[MESG_LEN];
    for (int i = 0; i < MESG_LEN; i++) {
        buffer[i] = (int) mesg[i];
    }
    uint32_t hash[5];
    SHA1_1(buffer, MESG_LEN, hash);
    for (int i = 0; i < 5; i++) {
        printf("%x ", hash[i]);
    }
    printf("\n");
    */

    uint8_t u_base[3];
    for (int i = 0; i < 3; i++) {
        u_base[i] = 0;
    }

    uint32_t resultIdx;
    uint32_t *hash_to_crack;
    SHA1CpuKernel(hash_to_crack, u_base, 3, "abcdef", 6, 6*6*6, resultIdx);

    return 0;
}
