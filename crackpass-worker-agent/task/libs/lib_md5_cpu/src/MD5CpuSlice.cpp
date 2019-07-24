#include <iostream>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "MD5Cpu.hpp"
#include "MD5CpuSlice.hpp"
#include "kernel/MD5CpuKernel.hpp"

MD5CpuSlice::MD5CpuSlice (
        MD5Cpu *controller
        )
{

    fSliceState = SLICE_STATE_READY;
    fWorksize = 1 << 16;

    fController = controller;
    fController->CopyBasicParams( fHashInts, fCharsetStr );

    fCharsetLen = fCharsetStr.size();

}


void MD5CpuSlice::StartSlice()
{
    fSliceState = SLICE_STATE_READY;
    fWorkerThread = new boost::thread( boost::bind(&MD5CpuSlice::WorkingLoop, this) );
}


void MD5CpuSlice::StopSlice()
{
    fSliceState = SLICE_STATE_STOPPED;
    fWorkerThread->join();
}


void MD5CpuSlice::WorkingLoop()
{
    fSliceState = SLICE_STATE_RUNNING;
    while( fSliceState != SLICE_STATE_STOPPED ) {
        // Prepare data.
        bool hasData = false;
        int requestWorksize = fWorksize;
        hasData = fController->PopRequest( fInputBase, fInputLen, requestWorksize );
        if (!hasData) {
            // IdleState: waiting for data, sleep to void consuming CPU
            boost::this_thread::sleep( boost::posix_time::milliseconds(50) );
            continue;
        }
        uint32_t resultIdx=0;
        int32_t kernelRet = MD5CpuKernel(
                &fHashInts[0],
                &fInputBase[0], fInputLen,
                fCharsetStr.c_str(), fCharsetLen,
                requestWorksize, resultIdx
                );
        if( kernelRet != -1 ) {
            // Result found
            fFoundResult = 1;
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
                fResultStr.push_back( fCharsetStr[a] );
            }
            // Announce to the controller
            fController->PushResult( fResultStr );
            break;
        }
    }

}




