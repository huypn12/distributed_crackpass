#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "MD5Utils.hpp"
#include "MultiMD5Cpu.hpp"
#include "MultiMD5CpuSlice.hpp"
#include "MultiMD5CpuKernel.hpp"


MultiMD5CpuSlice::MultiMD5CpuSlice (
        MultiMD5Cpu *controller
        )
{
    fSliceState = SLICE_STATE_READY;
    fWorkSize = 1 << 24;

    fController = controller;
    fController->CopyBasicParams( fHashList, fCharsetStr );

    fCharsetLen = fCharsetStr.size();

}

void MultiMD5CpuSlice::StartSlice()
{
    fSliceState = SLICE_STATE_RUNNING;
    fWorkerThread = new boost::thread( boost::bind(&MultiMD5CpuSlice::WorkingLoop, this) );
}

void MultiMD5CpuSlice::StopSlice()
{
    fSliceState = SLICE_STATE_STOPPED;
    fWorkerThread->join();
}


void MultiMD5CpuSlice::WorkingLoop()
{
    while( fSliceState != SLICE_STATE_STOPPED &&
            fSliceState != SLICE_STATE_FINISHED )
    {
        // Prepare data.
        uint32_t requestWorkSize = fWorkSize;
        bool hasData = fController->PopRequest(
                fInputBase, fInputLen, requestWorkSize );
        if (!hasData)
        {
            // IdleState: waiting for data, sleep to void consuming CPU
            boost::this_thread::sleep( boost::posix_time::milliseconds(50) );
            continue;
        }
        int32_t kernelRet = MultiMD5CpuKernel(
                fHashList, fHashList.size(),
                fResultList,
                &fInputBase[0], fInputLen,
                fCharsetStr.c_str(), fCharsetLen,
                requestWorkSize
                );
        if( kernelRet != -1 )
        {
            // Retrieve result
            while( !fResultList.empty() )
            {
                // pop from result list
                vec4uint32_t resultPair = fResultList.back();
                // referencing result data
                uint32_t resultIdx = resultPair.x;
                uint32_t hashIdx = resultPair.y;
                std::string tmpHashStr;
                MD5HexToDigest( fHashList[hashIdx], tmpHashStr );
                // recover plaintext by its index
                std::string tmpResultStr;
                for(int i = 0; i < fInputLen; i++)
                {
                    std::cout << fCharsetStr[fInputBase[i]];
                }
                std::cout << std::endl;
                uint32_t counter = resultIdx;
                for( uint32_t j = 0, a = 0, carry = 0;
                        j < fInputLen;
                        ++j, counter /= fCharsetLen) {
                    a = fInputBase[j]+ counter % fCharsetLen + carry;
                    if ( a >= fCharsetLen ) {
                        carry = 1;
                        a -= fCharsetLen;
                    } else {
                        carry = 0;
                    }
                    tmpResultStr.push_back( fCharsetStr[a] );
                }
                // Announce to the controller
                fController->PushResult( tmpResultStr, tmpHashStr );
                // Clear resultlist -> avoid delta checking for each step
                fResultList.pop_back();
            }
            //break;
        }
    }

}




