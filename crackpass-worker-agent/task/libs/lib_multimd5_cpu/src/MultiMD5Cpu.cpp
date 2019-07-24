#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "MD5Utils.hpp"
#include "MultiMD5Cpu.hpp"
#include "MultiMD5CpuSlice.hpp"



/**
 * convert hash ascii string to vector repr.
 * @param: std::string      hash string
 * @param: std::vector<uint32_t>& hash vector
 * @return: void
void MD5DigestToHex(
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
 */


/**
 * Convert md5 hex digest to string
void MD5HexToDigest(
        const vec4uint32_t hashvec,
        std::string &digest
        )
{
    std::string tmpStrX = std::to_string(hashvec.x);
    digest.insert(0, tmpStrX);
    std::string tmpStrY = std::to_string(hashvec.y);
    digest.insert(0, tmpStrY);
    std::string tmpStrZ = std::to_string(hashvec.z);
    digest.insert(0, tmpStrZ);
    std::string tmpStrW = std::to_string(hashvec.w);
    digest.insert(0, tmpStrW);
}
 */


/**
 * Compare 2 vectors by the first dim
 */
int compLtVecByX( const vec4uint32_t vec1, const vec4uint32_t vec2 )
{
    return( vec1.x < vec2.x );
}


/**
 * CONSTRUCTOR
 */
MultiMD5Cpu::MultiMD5Cpu(
        const int nCpuCores,
        const char **hashes, const int nHashes,
        const char *charset
        )
{
    // Init resources number & running flag
    fTaskState = TASK_STATE_READY;
    fCpuCores = nCpuCores;

    // Charset
    fCharsetStr  = std::string( charset );
    fCharsetLen = fCharsetStr.size();
    fCharsetUChar.resize( fCharsetLen );
    //@huypn: couldnt get why @tannd need this char->uchar conversion
    for( uint32_t i = 0; i < fCharsetLen; i++ ) {
        fCharsetUChar[i] = (unsigned char) fCharsetStr[i];
    }

    // Hash String
    for( int i = 0; i < nHashes; i++ ) {
        // read digest from array
        const char *szDigest = hashes[i];
        std::string tmpHashStr( szDigest );
        // convert szDigest to binary form
        std::vector<uint32_t> tmpHashInts( 4 );
        MD5DigestToHex( tmpHashStr, tmpHashInts );
        vec4uint32_t tmpHashVec(
                tmpHashInts[0],
                tmpHashInts[1],
                tmpHashInts[2],
                tmpHashInts[3]
                );
        // append to hash list
        fHashList.push_back(tmpHashVec);
    }
    // Sorting hash list by x, for later binary search
    std::sort(fHashList.begin(), fHashList.end(), compLtVecByX);
    for( int i = 0; i < nHashes; i++ )
    {
        std::cout << "hash i X=" << fHashList[i].x << std::endl;
    }

    // Init input base
    //@huypn 2015-11-03T16:49:08
    //      -> critical; without initialization, seg_fault appears
    fInputLen = 0;
    fCurrentBaseIdx = 0;
    fWorkSize = 0;
    std::fill( fInputBase.begin(), fInputBase.end(), 0 );
}


MultiMD5Cpu::~MultiMD5Cpu()
{
    // forces all slaves to stop
    Stop();
}


void MultiMD5Cpu::Start() {
    // trigger running flag
    fTaskState = TASK_STATE_RUNNING;
    // Create CPU workers on start and delete them on stop
    for( int i = 0; i < fCpuCores; i++ ) {
        MultiMD5CpuSlice *cpuSlice = new MultiMD5CpuSlice( this );
        fCpuSliceList.push_back( cpuSlice );
        cpuSlice->StartSlice();
    }
}

void MultiMD5Cpu::Stop(  )
{
    // set state
    if( fTaskState != TASK_STATE_FINISHED ) {
        fTaskState = TASK_STATE_STOPPED;
    }
    // Stop Cpu slaves
    for( int i = 0; i < fCpuCores; i++ ) {
        fCpuSliceList[i]->StopSlice();
    }
    // remove workers
    fCpuSliceList.clear();
}


int MultiMD5Cpu::GetState()
{
     return fTaskState;
}

void MultiMD5Cpu::CopyBasicParams(
        std::vector<vec4uint32_t> & hashList,
        std::string &charsetStr
        )
{
    hashList = fHashList;
    charsetStr = fCharsetStr;
}


void MultiMD5Cpu::PushRequest(
        const char *inputBaseStr,
        const size_t inputLen,
        const uint32_t offset
        )
{
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
    fWorkSize = offset;
    // Update the state flag
    if( fTaskState == TASK_STATE_WAITING_DATA ) {
        fTaskState = TASK_STATE_RUNNING;
    }

    fMutex.unlock();
}


bool MultiMD5Cpu::PopRequest(
        std::vector<unsigned char> &inputBase,
        size_t &inputLen,
        uint32_t &offset
        )
{
    bool ret = false;
    // pre-conditional check:
    //      skip requesting if already requested (waiting for new data)
    //      or finished
    if( fTaskState == TASK_STATE_WAITING_DATA ||
            fTaskState == TASK_STATE_FINISHED ) {
        return ret;
    }

    fMutex.lock();
    // Only pop request if there is request to pop :)))
    if( (fWorkSize > 0) && (fCurrentBaseIdx < fWorkSize - 1) ) {
        ret = true;
        // Get current idx
        uint32_t newOffset = fCurrentBaseIdx + offset;
        //if( newOffset > fWorkSize ) {
        if( newOffset >= fWorkSize ) {
            // Out of data.
            offset = fWorkSize - fCurrentBaseIdx;
            fCurrentBaseIdx = fWorkSize - 1;
            fTaskState = TASK_STATE_WAITING_DATA;
        } else {
            fCurrentBaseIdx += offset;
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
    // Release resources
    fMutex.unlock();

    return ret;
}


void MultiMD5Cpu::PushResult(
        const std::string result,
        const std::string digest
        )
{
    std::cout << "result: " << result << " digest: " << digest << std::endl;
    fMutex.lock();

    // Result found, no need of occupying resources
    fTaskState = TASK_STATE_FOUND_RESULT;
    // Push result
    std::pair<std::string,std::string> newResult(result, digest);
    fResultList.push_back(newResult);
    // finish this task if all results are found
    if( fResultList.size() == fHashList.size() )
    {
        fTaskState = TASK_STATE_FINISHED;
    }
    else
    {
        fTaskState = TASK_STATE_RUNNING;
    }

    fMutex.unlock();
}


int MultiMD5Cpu::PopResult( char **szResult, char **szDigest )
{
    fMutex.lock();

    size_t nResults = fResultList.size();
    if( nResults > 0 ) {
        std::pair<std::string, std::string> result;
        result = fResultList.back();
        fResultList.pop_back();
        // 1st elem: result
        strcpy( *szResult, result.first.c_str() );
        strcpy( *szDigest, result.second.c_str() );
        // not yet done :)))
    }
    fMutex.unlock();

    return nResults;
}



//---------------------------------------------------------------------------//
//-------------- DLL PUBLIC functions ---------------------------------------//
//---------------------------------------------------------------------------//
extern "C" {
    static class MultiMD5Cpu *md5instance = 0;

    int DLL_PUBLIC MultiMD5CpuInit(
            int nCpu,
            const char **szHashList, const int nHashes,
            const char *szCharset
            )
    {
        if ( md5instance ) {
            return( -1 );
        }
        md5instance = new MultiMD5Cpu( nCpu, szHashList, nHashes, szCharset );
        return( 0 );
    }

    int DLL_PUBLIC MultiMD5CpuStart()
    {
        if( md5instance ) {
            md5instance->Start();
            return( 0 );
        }
        return( -1 );
    }

    int DLL_PUBLIC MultiMD5CpuGetState()
    {
        if( md5instance ) {
            return md5instance->GetState();
        }
        return( -1 );
    }

    int DLL_PUBLIC MultiMD5CpuPushRequest(
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

    int DLL_PUBLIC MultiMD5CpuPopResult(
            char** result,
            char** hash
            )
    {
        if( md5instance ) {
            return( md5instance->PopResult( result, hash ));
        }
        return( -1 );
    }

    int DLL_PUBLIC MultiMD5CpuStop()
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



