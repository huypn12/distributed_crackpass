#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "SHA1Cpu.hpp"
#include "SHA1CpuSlice.hpp"


SHA1Cpu::SHA1Cpu(
        const int nCpuCores,
        const char *hash, const char *charset
        )
{
    // Init resources number & running flag
    fTaskState = TASK_STATE_READY;
    fCpuCores = nCpuCores * 4;

    // Charset
    fHashStr = std::string( hash );
    fHashDigest.resize( 20 );
    SHA1strToDigest( fHashStr, fHashDigest );
    fCharsetStr  = std::string( charset );
    fCharsetLen = fCharsetStr.size();
    fCharsetUChar.resize( fCharsetLen );
    for( uint32_t i = 0; i < fCharsetLen; i++ ) {
        fCharsetUChar[i] = (unsigned char) fCharsetStr[i];
    }

    // Init input base
    fWorksize = 0;
    std::fill( fInputBase.begin(), fInputBase.end(), 0 );
    fCurrentBaseIdx = 0;
    fInputLen = 0;
}


SHA1Cpu::~SHA1Cpu()
{
    // forces all slaves to stop
    Stop();
    // remove workers
    fCpuSliceList.clear();
}


/**
 * @author: tannd
 * Convert md5 ascii string to integer form
 */
void SHA1Cpu::SHA1strToDigest(
        const std::string hashStr,
        std::vector<unsigned char> &hashDigest
        )
{


    char* dummy = NULL;
    //unsigned char bytes[20];
    for (int i = 0; i < 20; ++i) {
        unsigned char tmpByte = (unsigned char) strtol(
                hashStr.substr(i * 2, 2).c_str(), &dummy, 16);
        hashDigest[i] = tmpByte;
        //printf("%d %02x\n", i, tmpByte);
    }
    /*
    hashDigest[0] = bytes[ 0] | bytes[ 1] << 8 | bytes[ 2] << 16 | bytes[ 3] << 24;
    hashDigest[1] = bytes[ 4] | bytes[ 5] << 8 | bytes[ 6] << 16 | bytes[ 7] << 24;
    hashDigest[2] = bytes[ 8] | bytes[ 9] << 8 | bytes[10] << 16 | bytes[11] << 24;
    hashDigest[3] = bytes[12] | bytes[13] << 8 | bytes[14] << 16 | bytes[15] << 24;
    hashDigest[4] = bytes[16] | bytes[17] << 8 | bytes[18] << 16 | bytes[19] << 24;
    */
}


int SHA1Cpu::GetTaskState()
{
     return fTaskState;
}


void SHA1Cpu::CopyBasicParams(
        std::vector<unsigned char> & hashDigest,
        std::string &charsetStr
        )
{
    hashDigest = fHashDigest;
    charsetStr = fCharsetStr;
}


void SHA1Cpu::Start() {
    // trigger running flag
    fTaskState = TASK_STATE_WAITING_DATA;
    // Create CPU workers on start and delete them on stop
    for( int i = 0; i < fCpuCores; i++ ) {
        SHA1CpuSlice *cpuSlice = new SHA1CpuSlice( this );
        fCpuSliceList.push_back( cpuSlice );
    }
    // Activate CPU workers
    for( int i = 0; i < fCpuCores; i++ ) {
        fCpuSliceList[i]->StartSlice();
    }
}

void SHA1Cpu::Stop(  )
{
    // set state
    fTaskState = TASK_STATE_STOPPED;
    // Stop Cpu slaves
    for( int i = 0; i < fCpuCores; i++ ) {
        fCpuSliceList[i]->StopSlice();
    }
}


void SHA1Cpu::PushRequest(
        const char *inputBaseStr,
        const size_t inputLen,
        const int offset
        )
{
    // this reside outside
    while( fTaskState == TASK_STATE_RUNNING ) {
    }

    fMutex.lock();

    if( inputLen > fInputLen ) {
        fInputBase.resize( inputLen );
    }
    fInputLen = inputLen;
    // Replace each character by its index in the characterset
    for( unsigned int i = 0; i < inputLen; i++ ) {
        int idx = fCharsetStr.find( inputBaseStr[i] );
        fInputBase[i] = (unsigned char) idx;
    }
    // Update offset
    fWorksize = offset;
    // Update the state flag
    if( fTaskState == TASK_STATE_WAITING_DATA ) {
        fTaskState = TASK_STATE_RUNNING;
    }

    fMutex.unlock();
}


bool SHA1Cpu::PopRequest(
        std::vector<unsigned char> &inputBase,
        size_t &inputLen,
        int &offset
        )
{
    fMutex.lock();

    bool ret = false;
    // pre-conditional check
    if (fTaskState == TASK_STATE_READY ||               // ready = out of data
            fTaskState == TASK_STATE_WAITING_DATA ||    // waiting = request sent, waiting for response
            fTaskState == TASK_STATE_FOUND_RESULT) {    // no need of pushing more
        fMutex.unlock();
        return ret;
    }
    // Only pop request if there is request to pop :))
    int maxBaseIdx = fWorksize - 1;
    if (fCurrentBaseIdx < maxBaseIdx) {
        ret = true;
        // Get current idx
        int32_t newBaseIdx = fCurrentBaseIdx + offset;
        if (newBaseIdx > maxBaseIdx) {
            // If requested offset reaches or exceeds the end
            offset = maxBaseIdx - fCurrentBaseIdx;
            fCurrentBaseIdx = maxBaseIdx;
            fTaskState = TASK_STATE_WAITING_DATA;
        } else {
            // if not exceeded the end
            fCurrentBaseIdx = newBaseIdx;
        }
        // Assign current base
        inputBase = fInputBase ;
        inputLen = fInputLen;
        // Calculate new base
        // @huypn12 2016-05-04 12:36
        // replaced absolute idx by relative offset
        // using absolute idx caused defect (ignoring result)
        uint32_t counter = offset;//fCurrentBaseIdx;
        for( uint32_t j = 0, a = 0, carry = 0;
                j < fInputLen;
                ++j, counter /= fCharsetLen) {
            a = fInputBase[j] + carry + counter % fCharsetLen;
            if ( a >= fCharsetLen ) {
                carry = 1;
                a -= fCharsetLen;
            } else {
                carry = 0;
            }
            fInputBase[j] = a;
        }
    }  else {
        ret = false;
        fTaskState = TASK_STATE_WAITING_DATA;
    }
#ifdef __DEBUG
    std::cout << "Popped request id=" << fCurrentBaseIdx << ";base=";
    for (int i = 0; i < inputLen; i++) { std::cout << fCharsetStr[fInputBase[i]]; }
    std::cout << " of max=" << maxBaseIdx << ";charsetlen=" << fCharsetLen << std::endl;
#endif

    fMutex.unlock();

    return ret;
}


void SHA1Cpu::PushResult( const std::string result )
{
    fMutex.lock();

    // Preconditional check
    if (fTaskState == TASK_STATE_FOUND_RESULT) {
        fMutex.unlock();
        return;
    } else {
        // Push result
        fResultStr = result;
        // Result found, no need of occupying resourcesi
        fTaskState = TASK_STATE_FOUND_RESULT;
    }

    fMutex.unlock();
}


int SHA1Cpu::PopResult( char *result )
{
    fMutex.lock();

    int ret = 0;
    if( !fResultStr.empty() ) {
        size_t resultStrLen = fResultStr.length();
        strncpy( result, fResultStr.c_str(), resultStrLen);//fResultStr.length());
        result[resultStrLen] = '\0';
        //printf("Found result: %s\n", result);
    } else {
        ret = 1;
    }

    fMutex.unlock();

    return ret;
}





/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

extern "C" {

    //static class SHA1Cpu *sha1instance = 0;
#define MAX_INSTANCES_COUNT 128
    //static std::vector<SHA1Cpu *> instance_list;
    static std::map<int, SHA1Cpu *> instanceSet;

    int DLL_PUBLIC SHA1CpuInit(
            const int task_id,
            const int n_cpu,
            const char *hash_str,
            const char *charset_str
            )
    {
        SHA1Cpu *sha1instance = new SHA1Cpu(n_cpu, hash_str, charset_str);
        instanceSet.insert(std::make_pair(task_id, sha1instance));
        return( 0 );
    }


    int DLL_PUBLIC SHA1CpuStart(int task_id)
    {
        SHA1Cpu *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->Start();
            return 0;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CpuGetState(int task_id)
    {
        SHA1Cpu *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            int state = instancePtr->GetTaskState();
            return state;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CpuPushRequest(
            int task_id,
            const char *input_base_str,
            int input_len,
            unsigned long offset )
    {
        SHA1Cpu *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->PushRequest(input_base_str, input_len, offset);
            return 0;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CpuPopResult(int task_id, char* result)
    {
        SHA1Cpu *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->PopResult(result);
            return 0;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CpuStop(int task_id)
    {
        SHA1Cpu *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->Stop();
            return 0;
        } else {
            return -1;
        }
    }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/



