#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "MD5Cpu.hpp"
#include "MD5CpuSlice.hpp"


MD5Cpu::MD5Cpu(
                const int nCpuCores,
                const char *hash, const char *charset
              )
{
        // Init resources number & running flag
        fTaskState = TASK_STATE_READY;
        fCpuCores = nCpuCores * 4;

        // Charset
        fHashStr = std::string( hash );
        fHashInts.resize( 4 );
        MD5digestToHex( fHashStr, fHashInts );
        fCharsetStr  = std::string( charset );
        fCharsetLen = fCharsetStr.size();
        fCharsetUChar.resize( fCharsetLen );
        //@huypn: couldnt get why @tannd need this char->uchar conversion
        for( uint32_t i = 0; i < fCharsetLen; i++ ) {
                fCharsetUChar[i] = (unsigned char) fCharsetStr[i];
        }

        // Init input base
        fWorksize = 0;
        std::fill( fInputBase.begin(), fInputBase.end(), 0 );
        fCurrentBaseIdx = 0;
        //@huypn 2015-11-03T16:49:08
        //      -> critical; without initialization, seg_fault appears
        fInputLen = 0;
}


MD5Cpu::~MD5Cpu()
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
void MD5Cpu::MD5digestToHex(
                const std::string hashStr,
                std::vector<uint32_t> &hashInts
                )
{
        char* dummy = NULL;
        unsigned char bytes[16];
        for( int i = 0; i < 16; ++i )
                bytes[i] = (unsigned char) strtol(
                                hashStr.substr(i * 2, 2).c_str(), &dummy, 16);
        hashInts[0] = bytes[ 0] | bytes[ 1] << 8 | bytes[ 2] << 16 | bytes[ 3] << 24;
        hashInts[1] = bytes[ 4] | bytes[ 5] << 8 | bytes[ 6] << 16 | bytes[ 7] << 24;
        hashInts[2] = bytes[ 8] | bytes[ 9] << 8 | bytes[10] << 16 | bytes[11] << 24;
        hashInts[3] = bytes[12] | bytes[13] << 8 | bytes[14] << 16 | bytes[15] << 24;
}

int MD5Cpu::GetState()
{
        return fTaskState;
}

void MD5Cpu::CopyBasicParams(
                std::vector<uint32_t> & hashInts,
                std::string &charsetStr
                )
{
        hashInts = fHashInts;
        charsetStr = fCharsetStr;
}


void MD5Cpu::Start() {
        // trigger running flag
        fTaskState = TASK_STATE_WAITING_DATA;
        // Create CPU workers on start and delete them on stop
        for( int i = 0; i < fCpuCores; i++ ) {
                MD5CpuSlice *cpuSlice = new MD5CpuSlice( this );
                fCpuSliceList.push_back( cpuSlice );
        }
        // Activate CPU workers
        for( int i = 0; i < fCpuCores; i++ ) {
                fCpuSliceList[i]->StartSlice();
        }
}

void MD5Cpu::Stop(  )
{
        // set state
        fTaskState = TASK_STATE_STOPPED;
        // Stop Cpu slaves
        for( int i = 0; i < fCpuCores; i++ ) {
                fCpuSliceList[i]->StopSlice();
        }
}


void MD5Cpu::PushRequest(
                const char *inputBaseStr,
                const size_t inputLen,
                const int offset
                )
{

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


bool MD5Cpu::PopRequest(
                std::vector<unsigned char> &inputBase,
                size_t &inputLen,
                int &offset
                )
{
        fMutex.lock();

        bool ret = false;
        // pre-conditional check
        if( fTaskState == TASK_STATE_READY ||               // ready = out of data
                        fTaskState == TASK_STATE_WAITING_DATA ||    // waiting = request sent, waiting for response
                        fTaskState == TASK_STATE_FOUND_RESULT ) {   // no need of pushing more
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
                uint32_t counter = offset;
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
        } else {
                ret = false;
                fTaskState = TASK_STATE_WAITING_DATA;
        }
        // Release resources
        fMutex.unlock();

        return ret;
}

void MD5Cpu::PushResult( const std::string result )
{
        std::cout << "Result found: " << result << std::endl;

        fMutex.lock();

        if (fTaskState == TASK_STATE_FOUND_RESULT) {
                fMutex.unlock();
        }
        // Result found, no need of occupying resources
        fTaskState = TASK_STATE_FOUND_RESULT;
        // Push result
        fResultStr = result;



        fMutex.unlock();
}


int MD5Cpu::PopResult( char *result )
{
        fMutex.lock();

        int ret = 0;
        if( !fResultStr.empty() ) {
                strncpy( result, fResultStr.c_str(), fResultStr.length());
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

        //static class MD5Cpu *md5instance = 0;
#define MAX_INSTANCES_COUNT 4096
        //static std::vector<MD5Cpu *> instance_list(MAX_INSTANCES_COUNT) ;
        static std::map<int, MD5Cpu *> instance_set;

        /**
         * Create new task, insert into task list
         * @param task id
         * @param num cpu
         * @param hash string
         * @param charset string
         * @return
         */
        int DLL_PUBLIC MD5CpuInit(
                        const int task_id,
                        const int n_cpu,
                        const char *hash_str,
                        const char *charset_str
                        )
        {
                MD5Cpu *md5instance = new MD5Cpu(n_cpu, hash_str, charset_str);
                instance_set.insert(std::make_pair(task_id, md5instance));
                return( 0 );
        }


        int DLL_PUBLIC MD5CpuStart(int task_id)
        {
                MD5Cpu *instancePtr = instance_set[task_id];
                if (instancePtr != NULL) {
                        instancePtr->Start();
                        return 0;
                } else {
                        return -1;
                }
        }


        int DLL_PUBLIC MD5CpuGetState(int task_id)
        {
                MD5Cpu *instancePtr = instance_set[task_id];
                if (instancePtr != NULL) {
                        instancePtr->GetState();
                        return 0;
                } else {
                        return -1;
                }
        }


        int DLL_PUBLIC MD5CpuPushRequest(
                        int task_id,
                        const char *input_base_str,
                        int input_len,
                        unsigned long offset )
        {
                MD5Cpu *instancePtr = instance_set[task_id];
                if (instancePtr != NULL) {
                        instancePtr->PushRequest(input_base_str, input_len, offset);
                }
                return -1;
        }


        int DLL_PUBLIC MD5CpuPopResult(int task_id, char* result)
        {
                MD5Cpu *instancePtr = instance_set[task_id];
                if (instancePtr != NULL) {
                        instancePtr->PopResult(result);
                }
                return -1;
        }


        int DLL_PUBLIC MD5CpuStop(int task_id)
        {
                MD5Cpu *instancePtr = instance_set[task_id];
                if (instancePtr != NULL) {
                        instancePtr->Stop();
                }
                return -1;
        }
}

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/



