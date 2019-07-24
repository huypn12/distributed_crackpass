/*
 ******************************************************************************
 *                                                                            *
 *                                                                            *
 *  @file:                                                                    *
 *  @author:                                                                  *
 *  @desc:                                                                    *
 *  @created:                                                                 *
 *                                                                            *
 *                                                                            *
 *                                                                            *
 *                                                                            *
 *                                                                            *
 *                                                                            *
 ******************************************************************************
 */

#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "SHA1Cuda.hpp"
#include "SHA1CudaSlice.hpp"


/*
 *******************************************************************************
 *                              DEBUGGING MACROS                               *
 *******************************************************************************
 */
#ifdef _DEBUG

#define PRINT_DBG( mesg, pos, level ) { \
    std::cout << "_DEBUG:[" << level << "] @obj(SHA1Cuda)." << pos << "MESG=" << mesg << std::endl; \
}

#define PRINT_ERR( mesg, pos ) {        \
    PRINT_DBG( mesg, pos, "ERROR" );    \
}

#define PRINT_INFO( mesg, pos ) {       \
    PRINT_DBG( mesg, pos, "INFO" );     \
}

#endif




/*
 *******************************************************************************
 *                  CONSTRUCTOR & DESTRUCTOR                                   *
 *******************************************************************************
 */
SHA1Cuda::SHA1Cuda(
        int32_t nCudaDevices,
        int32_t* cudaDevicesList,
        const char* hashStr,
        const char* charset
        )
{
    fTaskState = TASK_STATE_READY;

    fCudaDeviceCount = nCudaDevices;
    for (int i = 0; i < fCudaDeviceCount; i++) {
        fCudaDeviceList.push_back(cudaDevicesList[i]);
    }

    fHashStr = std::string(hashStr);
    fHashDigest.resize(20);
    SHA1strToDigest( fHashStr, fHashDigest );

    fCharsetStr = std::string( charset );
    fCharsetLen = fCharsetStr.size();
    fCharsetUChar.resize( fCharsetLen );
    for( unsigned int i = 0; i < fCharsetLen; i++ ) {
        fCharsetUChar[i] = (unsigned char) fCharsetStr[i];
    }

    fInputBase.resize(16);
    std::fill( fInputBase.begin(), fInputBase.end(), 0 );
    fCurrentBaseIdx = 0;
    fInputLen = 0;
}


SHA1Cuda::~SHA1Cuda()
{
    Stop();
}


/*
 *******************************************************************************
 *                  START & STOP                                               *
 *******************************************************************************
 */
void SHA1Cuda::Start() {
    // Start running task.
    // Activate task, create and activate CUDA workers
    fTaskState = TASK_STATE_WAITING_DATA;
    // Create CUDA workers
    for( int i = 0; i < fCudaDeviceCount; i++ ) {
        SHA1CudaSlice *cudaSlice = new SHA1CudaSlice( this );
        fCudaSliceList.push_back(cudaSlice);
        cudaSlice->StartSlice( fCudaDeviceList[i] );
    }
}


void SHA1Cuda::Stop(  )
{
    // Stop running task.
    // Deactive task, stop and destroy all cuda workers

    fTaskState = TASK_STATE_STOPPED;
    // Stop Cuda collaborators
    for( int i = 0; i < fCudaDeviceCount; i++ ) {
        fCudaSliceList[i]->StopSlice();
    }
    fCudaSliceList.clear();
}



/*
 *******************************************************************************
 *                              UTILITIES                                      *
 *******************************************************************************
 */
    void SHA1Cuda::SHA1strToDigest(
        const std::string hashStr,
        std::vector<unsigned char> &hashDigest
        )
{
    // Convert SHA1 ascii string into bitvector form
    // note: original code from @tannd
     char* dummy = NULL;
    //unsigned char bytes[20];
    std::cout << hashStr << std::endl;
    for (int i = 0; i < 20; ++i) {
        unsigned char tmpByte = (unsigned char) strtol(
                hashStr.substr(i * 2, 2).c_str(), &dummy, 16);
        hashDigest[i] = tmpByte;
        //printf("%d %02x\n", i, tmpByte);
    }
 }


void SHA1Cuda::CopyBasicParams(
        std::vector<unsigned char> &hashDigest,
        std::string &charsetStr
        )
{
    // Copy hash vector and charset
    // to be called from collaborator;
    // copy basic params from common work dist-or
    hashDigest = fHashDigest;
    charsetStr = fCharsetStr;
}


/*
 *******************************************************************************
 *                              REQUEST PUSH_&_PULL                            *
 *******************************************************************************
 */
void SHA1Cuda::PushRequest(
        const char *inputBaseStr,
        const unsigned int inputLen,
        const int offset
        )
{
    // Push request
    // Push received data chunk ("request") into the data queue,
    // To be called from task manager.

    while (fTaskState == TASK_STATE_RUNNING) {
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
    fOffset = offset;

    // Update the state flag
    if( fTaskState == TASK_STATE_WAITING_DATA ) {
        fTaskState = TASK_STATE_RUNNING;
    }

    fMutex.unlock();
}


bool SHA1Cuda::PopRequest(
        std::vector<unsigned char> &inputBase,
        size_t &inputLen,
        int &offset
        )
{
    // Pop request
    // To be called from slice workers
    // Pop a data chunk ("request") with size ("offset") suggested by the caller
    // If the task is currently in waiting for more data or result is found
    //      -> return false

    bool ret = false;
    if( fTaskState == TASK_STATE_READY ||
            fTaskState == TASK_STATE_WAITING_DATA ||
            fTaskState == TASK_STATE_FOUND_RESULT ) {
        return ret;
    }

    fMutex.lock();

    uint32_t maxBaseIdx = fOffset - 1;
    // Only pop request if there is request to pop :)))
    if( (fOffset > 0) && (fCurrentBaseIdx != maxBaseIdx) ) {
        ret = true;
        // Get current idx
        uint32_t newBaseIdx = fCurrentBaseIdx + offset;
        if( newBaseIdx >= maxBaseIdx ) {
            // Out of data.
            offset = maxBaseIdx - fCurrentBaseIdx + 1;//newBaseIdx - maxBaseIdx;
            fCurrentBaseIdx = maxBaseIdx;
            fTaskState = TASK_STATE_WAITING_DATA;
        } else {
            fCurrentBaseIdx = newBaseIdx;
        }
        // Assign current base
        inputBase = fInputBase;
        inputLen = fInputLen;
        // Calculate new base
        uint32_t counter = fCurrentBaseIdx;
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
        fTaskState = TASK_STATE_WAITING_DATA;
    }

    fMutex.unlock();


#ifdef _DEBUG
    // Debugging: checkout current base
    std::cout << "Popping request, currentIdx=" << fCurrentBaseIdx << ";offset=" << offset << ";inputbase=";
    for( int i = 0; i < inputLen; i++ ) {
        std::cout << fCharsetStr[ inputBase[i] ];
    }
    std::cout << ";";
    for( int i = 0; i < inputLen; i++ ) {
        std::cout << fCharsetStr[ fInputBase[i] ];
    }
    std::cout << std::endl;
#endif

    return ret;
}


/*
 *******************************************************************************
 *                              RESULT PUSH_&_PULL                            *
 *******************************************************************************
 */
void SHA1Cuda::PushResult( const std::string result )
{
    // Push matching plaintext ("result")
    // To be called by slice worker

    fMutex.lock();

    // Result found, no need of occupying resources
    fTaskState = TASK_STATE_FOUND_RESULT;
    // Push result
    fResultStr = result;
    // only 1 hash to find -> finish
    fTaskState = TASK_STATE_FINISHED;

    fMutex.unlock();

}


int SHA1Cuda::PopResult( char *result )
{
    // To be called by task manager
    fMutex.lock();

    int ret = 0;
    if( !fResultStr.empty() ) {
        size_t resultStrLen = fResultStr.length();
        strncpy(result, fResultStr.c_str(), resultStrLen);
        result[resultStrLen] = '\0';
    } else {
        ret = 1;
    }

    fMutex.unlock();

    return ret;
}

int SHA1Cuda::GetState()
{
     return fTaskState;
}


/*
 *******************************************************************************
 *                              DLL_PUBLIC METHODS                             *
 *******************************************************************************
 */
extern "C" {
    // actual object
    //static class SHA1Cuda *sha1instance = 0;

    static std::map<int, SHA1Cuda *> instanceSet;

    int DLL_PUBLIC SHA1CudaInit(
            const int task_id,
            int n_gpu,
            int* gpu_list,
            const char *hash_str,
            const char *charset_str
            )
    {
        SHA1Cuda *sha1instance = new SHA1Cuda(n_gpu, gpu_list, hash_str, charset_str);
        instanceSet.insert(std::make_pair(task_id, sha1instance));
        return( 0 );
    }


    int DLL_PUBLIC SHA1CudaStart(int task_id)
    {
        SHA1Cuda *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->Start();
            return 0;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CudaGetState(int task_id)
    {
        SHA1Cuda *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->GetState();
            return 0;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CudaPushRequest(
            int task_id,
            const char *input_base_str,
            int input_len,
            unsigned long offset )
    {
        SHA1Cuda *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->PushRequest(input_base_str, input_len, offset);
            return 0;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CudaPopResult(int task_id, char* result)
    {
        SHA1Cuda *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->PopResult(result);
            return 0;
        } else {
            return -1;
        }
    }


    int DLL_PUBLIC SHA1CudaStop(int task_id)
    {
        SHA1Cuda *instancePtr = instanceSet[task_id];
        if (instancePtr != NULL) {
            instancePtr->Stop();
            return 0;
        } else {
            return -1;
        }
    }
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//



