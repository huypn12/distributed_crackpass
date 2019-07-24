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

#include "MD5Cuda.hpp"
#include "MD5CudaSlice.hpp"


/*
 *******************************************************************************
 *                              DEBUGGING MACROS                               *
 *******************************************************************************
 */
#ifdef _DEBUG

#define PRINT_DBG( mesg, pos, level ) { \
    std::cout << "_DEBUG:[" << level << "] @obj(MD5Cuda)." << pos << "MESG=" << mesg << std::endl; \
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
MD5Cuda::MD5Cuda(
        int32_t nCudaDevices,
        int32_t* cudaDevicesList,
        const char* hash,
        const char* charset
        )
{
    fTaskState = TASK_STATE_READY;

    fCudaDeviceCount = nCudaDevices;
    for( int i = 0; i < fCudaDeviceCount; i++ ) {
        fCudaDeviceList.push_back( cudaDevicesList[i] );
    }

    fHashStr = std::string( hash );
    fHashInts.resize( 4 );
    MD5digestToHex( fHashStr, fHashInts );

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


MD5Cuda::~MD5Cuda()
{
    Stop();
}


/*
 *******************************************************************************
 *                  START & STOP                                               *
 *******************************************************************************
 */
void MD5Cuda::Start() {
    // Start running task.
    // Activate task, create and activate CUDA workers
    fTaskState = TASK_STATE_RUNNING;
    // Create CUDA workers
    for( int i = 0; i < fCudaDeviceCount; i++ ) {
        MD5CudaSlice *cudaSlice = new MD5CudaSlice( this );
        fCudaSliceList.push_back(cudaSlice);
        cudaSlice->StartSlice( fCudaDeviceList[i] );
    }
}


void MD5Cuda::Stop(  )
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
void MD5Cuda::MD5digestToHex(
        const std::string hashStr,
        std::vector<unsigned int> &hashInts
        )
{
    // Convert MD5 ascii string into bitvector form
    // note: original code from @tannd
    char* dummy = NULL;
    unsigned char bytes[16];
    for( int i = 0; i < 16; ++i )
        bytes[i] = (unsigned char) strtol(hashStr.substr(i * 2, 2).c_str(), &dummy, 16);
    // old code written by @tannd
    hashInts[0] = bytes[ 0] | bytes[ 1] << 8 | bytes[ 2] << 16 | bytes[ 3] << 24;
    hashInts[1] = bytes[ 4] | bytes[ 5] << 8 | bytes[ 6] << 16 | bytes[ 7] << 24;
    hashInts[2] = bytes[ 8] | bytes[ 9] << 8 | bytes[10] << 16 | bytes[11] << 24;
    hashInts[3] = bytes[12] | bytes[13] << 8 | bytes[14] << 16 | bytes[15] << 24;
}


void MD5Cuda::CopyBasicParams(
        std::vector<unsigned int> &hashInts,
        std::string &charsetStr
        )
{
    // Copy hash vector and charset
    // to be called from collaborator; copy basic params from common work dist-or
     hashInts = fHashInts;
     charsetStr = fCharsetStr;
}


/*
 *******************************************************************************
 *                              REQUEST PUSH_&_PULL                            *
 *******************************************************************************
 */
void MD5Cuda::PushRequest(
        const char *inputBaseStr,
        const unsigned int inputLen,
        const int offset
        )
{
    // Push request
    // Push received data chunk ("request") into the data queue,
    // To be called from task manager.


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


bool MD5Cuda::PopRequest(
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
    if( fTaskState == TASK_STATE_WAITING_DATA ||
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
void MD5Cuda::PushResult( const std::string result )
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


int MD5Cuda::PopResult( char *result )
{
    // To be called by task manager
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

int MD5Cuda::GetState()
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
    static class MD5Cuda *md5instance = 0;

    int DLL_PUBLIC MD5CudaInit(
            int n_gpu, int* gpu_list,
            const char *hash_str,
            const char *charset_str
            )
    {
        // object construction and initialization

        if ( md5instance ) {
            return( -1 );
        }
        md5instance = new MD5Cuda( n_gpu, gpu_list, hash_str, charset_str );
        return( 0 );
    }

    int DLL_PUBLIC MD5CudaStart()
    {
        if( md5instance ) {
            md5instance->Start();
            return( 0 );
        }
        return( -1 );
    }

    int DLL_PUBLIC MD5CudaGetState()
    {
        if( md5instance ) {
            return md5instance->GetState();
        }
        return( -1 );
    }

    int DLL_PUBLIC MD5CudaPushRequest(
            // unsigned char * input_base,
            const char *input_base_str,
            int input_len,
            unsigned long offset )
    {
        if( md5instance ) {
            md5instance->PushRequest( input_base_str, input_len, offset );
            return( 0 );
        }
        return( -1 );
    }

    int DLL_PUBLIC MD5CudaPopResult(
            char* result
            )
    {
        if( md5instance ) {
            return( md5instance->PopResult( result ));
        }
        return( -1 );
    }

    int DLL_PUBLIC MD5CudaStop()
    {
        if( md5instance ) {
            md5instance->Stop();
            delete md5instance;
            return( 0 );
        }
        return( -1 );
    }
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//



